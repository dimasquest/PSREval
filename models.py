import warnings
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torch import Tensor
import torch.nn as nn
from typing import Optional, Type, Any, List, Union, Callable


def conv_bn_act(
    in_channels, out_channels, pool=False, act_func=nn.Mish, num_groups=None
):
    if num_groups is not None:
        warnings.warn("num_groups has no effect with BatchNorm")
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        act_func(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_gn_act(in_channels, out_channels, pool=False, act_func=nn.Mish, num_groups=32):
    """Conv-GroupNorm-Activation
    """
    if num_groups >= out_channels:
        num_groups = out_channels
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        act_func(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class QuantizedResNet9(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        act_func: nn.Module = nn.ReLU,
        scale_norm: bool = False,
        norm_layer: str = "group",
        num_groups = (32, 32, 32, 32),
    ):
        """9-layer Residual Network. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC
        Args:
            in_channels (int, optional): Channels in the input image. Defaults to 3.
            num_classes (int, optional): Number of classes. Defaults to 10.
            act_func (nn.Module, optional): Activation function to use. Defaults to nn.Mish.
            scale_norm (bool, optional): Whether to add an extra normalisation layer after each residual block. Defaults to False.
            norm_layer (str, optional): Normalisation layer. One of `batch` or `group`. Defaults to "batch".
            num_groups (tuple[int], optional): Number of groups in GroupNorm layers.\
            Must be a tuple with 4 elements, corresponding to the GN layer in the first conv block, \
            the first res block, the second conv block and the second res block. Defaults to (32, 32, 32, 32).
        """
        super().__init__()

        if norm_layer == "batch":
            conv_block = conv_bn_act
        elif norm_layer == "group":
            conv_block = conv_gn_act
        else:
            raise ValueError("`norm_layer` must be `batch` or `group`")

        assert (
            isinstance(num_groups, tuple) and len(num_groups) == 4
        ), "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_block(
            in_channels, 64, act_func=act_func, num_groups=groups[0]
        )
        self.conv2 = conv_block(
            64, 128, pool=True, act_func=act_func, num_groups=groups[0]
        )

        self.res1 = nn.Sequential(
            *[
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
            ]
        )

        self.conv3 = conv_block(
            128, 256, pool=True, act_func=act_func, num_groups=groups[2]
        )
        self.conv4 = conv_block(
            256, 256, pool=True, act_func=act_func, num_groups=groups[2]
        )

        self.res2 = nn.Sequential(
            *[
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
            ]
        )

        self.MP = nn.AdaptiveMaxPool2d((2, 2))
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)

        if scale_norm:
            num_groups = groups[1]
            if groups[1] >= 128:
                num_groups = 128
            self.scale_norm_1 = (
                nn.BatchNorm2d(128)
                if norm_layer == "batch"
                else nn.GroupNorm(num_groups, 128)
            )  # type:ignore
            self.scale_norm_2 = (
                nn.BatchNorm2d(256)
                if norm_layer == "batch"
                else nn.GroupNorm(groups[3], 256)
            )  # type:ignore
        else:
            self.scale_norm_1 = nn.Identity()  # type:ignore
            self.scale_norm_2 = nn.Identity()  # type:ignore
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()


    def forward(self, xb):
        out = self.quant(xb)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.skip_add.add(out, self.res1(out))
        out = self.scale_norm_1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.skip_add.add(out, self.res2(out))
        out = self.scale_norm_2(out)
        out = self.dequant(out)
        
        out = self.MP(out)
        
        out = self.quant(out)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        out = self.dequant(out)
        return out


class QuantizedResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = resnet18(num_classes=num_classes, pretrained=False)
        self.model_fp32.conv1 = torch.nn.Conv2d(in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # Rename relu to relu1
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()
        # Remember to use two independent ReLU for layer fusion.
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Use FloatFunctional for addition for quantization compatibility
        # out += identity
        out = self.skip_add.add(identity, out)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.skip_add.add(identity, out)
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)