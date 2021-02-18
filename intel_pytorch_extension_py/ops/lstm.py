import torch
from torch import _VF

VF_lstm = _VF.lstm

def ipex_lstm(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first, device):
    # For LSTM training with dropout, fallback to cpu due to performance issue in oneDNN mode
    if training and dropout != 0:
        return fallback_lstm(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first, device=device)
    else:
        return torch.ops.torch_ipex.lstm(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)

# users may only transfer the data but not the module to IPEX device, need to check if every item in the args is on "cpu" device
def get_device(*args):
    for item in args:
        if isinstance(item, (tuple, list)):
            for x in item:
                if x.device.type != "cpu":
                    return x.device.type
        elif isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                return item.device.type
    return "cpu"

def fallback_lstm(*args, device):
    # move args to cpu device
    args_cpu  = []
    # args is a tuple which does not support item assignment
    for item in args:
        if isinstance(item, (tuple, list)):
            item_cpu = [x.to("cpu") for x in item]
        elif isinstance(item, torch.Tensor):
            item_cpu = item.to("cpu")
        else:
            item_cpu = item
        args_cpu.append(item_cpu)
    
    output = VF_lstm(*args_cpu)
    
    # move output to the original device
    output_device = []
    # output is a tuple which does not support item assignment
    for item in output:
        item_device = item.to(device)
        output_device.append(item_device)
    return tuple(output_device)

def lstm(*args):
    device = get_device(*args)
    if device == "cpu":
        return VF_lstm(*args)
    
    # For LSTM with pack_padded_sequence as input, fallback to cpu due to performance issue in oneDNN mode
    if isinstance(args[1], torch.Tensor):
        return fallback_lstm(*args, device=device)
    else:
        return ipex_lstm(*args, device=device)

_VF.lstm = lstm
