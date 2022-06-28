import os
import torch
import models
import data
import torchattacks


def print_size_of_model(model, label=""):
    path = os.path.dirname(__file__)
    torch.save(model.state_dict(), f"{path}/temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size


def load_model(name, in_channels=1, num_classes=3, act_func=torch.nn.ReLU, norm_layer="group"):
    if name == 'resnet18':
        return models.QuantizedResNet18(in_channels, num_classes)
    elif name == 'resnet9':
        return models.QuantizedResNet9(in_channels=in_channels, num_classes=num_classes, act_func=act_func, norm_layer=norm_layer)
    else:
        raise Exception('Model architecture not supported')


def load_dataset(name, batch_size_train, batch_size_test):
    if name == 'cifar10':
        return data.load_cifar_10(batch_size_train, batch_size_test)
    elif name == 'pneumonia':
        return data.load_pneumonia(batch_size_train, batch_size_test)
    elif name == 'derma':
        return data.load_derma(batch_size_train, batch_size_test)
    else:
        raise Exception('Dataset not supported!')


def select_activation_func(name):
    if name == 'ReLU':
        return torch.nn.ReLU
    elif name == 'Mish':
        return torch.nn.Mish
    else:
        raise Exception('Unsupported activation function!')


def generate_tags(adversarial_data, adv_training, attack_name):
    if adv_training:
        defence = 'adv_training_' + str(adversarial_data)
    else:
        defence = 'vanilla'

    if adversarial_data > 0.0 and not adv_training:
        attacker = 'train_time_' + str(adversarial_data)
    else:
        attacker = 'inference_time'

    return [defence, attacker, attack_name]


def select_attack(name, model, eps=2/255, alpha=2/255, steps=20, num_classes=10):
    if name == 'PGD':
        return torchattacks.PGD(model=model, eps=eps, alpha=alpha, steps=steps)
    elif name == 'FGSM':
        return torchattacks.FGSM(model=model, eps=eps)
    elif name == 'FAB':
        return torchattacks.FAB(model=model, eps=eps, alpha_max=alpha, steps=steps, num_classes=num_classes)
    else:
        raise Exception('Unsupported attack type!')