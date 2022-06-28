import torch 
import train
import utilities as utils
from copy import deepcopy

def quantize_static(model, test_loader, name, path):

    model_quant = deepcopy(model).to('cpu')
    model_quant.eval()
    model_quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # N.B. fusion is disabled by default, because very model specific, but examples include:
    # model_quant = torch.quantization.fuse_modules(model_quant,
    #     [['conv1','relu1'], ['conv2', 'relu2'], ['fc1', 'relu3'], ['fc2', 'relu4']])
    model_quant = torch.quantization.prepare(model_quant)
    train.evaluate(model_quant, test_loader, 'cpu', name=name)
    torch.quantization.convert(model_quant, inplace=True)
    train.evaluate(model_quant, test_loader, 'cpu', name=name + '_quantized')

    size_vanilla = utils.print_size_of_model(model,"Original")
    size_quant = utils.print_size_of_model(model_quant, 'Quantized')
    print("{0:.3f}% smaller".format(100 - size_quant/size_vanilla*100))
    torch.save(model_quant.state_dict(), path)
    return model_quant