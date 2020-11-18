import torch
from torch import _VF

VF_lstm = _VF.lstm

def ipex_lstm(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first):
    if input.device == torch.device('dpcpp') and (dropout == 0 or training == False):
        return torch.ops.torch_ipex.lstm(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)
    else:
        return VF_lstm(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)

def lstm(*args):
    if isinstance(args[1], torch.Tensor):
        return VF_lstm(*args)
    else:
        return ipex_lstm(*args)

_VF.lstm = lstm
