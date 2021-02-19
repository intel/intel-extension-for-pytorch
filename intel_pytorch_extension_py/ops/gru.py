import math
import torch
from torch.nn.modules.rnn import RNNBase
from torch.nn.utils.rnn import PackedSequence
from torch import _VF

VF_gru = _VF.gru

def ipex_gru(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first):
    if input.device.type == 'xpu' and (dropout == 0 or training == False):
        return torch.ops.torch_ipex.gru(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)
    else:
        return VF_gru(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)

def gru(*args):
    if isinstance(args[2], torch.Tensor):
        return VF_gru(*args)
    else:
        return ipex_gru(*args)

_VF.gru = gru