import torch
import copy
from torch._six import string_classes as _string_classes
import copyreg
import pickle
import pathlib

DEFAULT_PROTOCOL = 2

torch_save = torch.save

def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=False):
    def to_cpu(obj):
        for k in obj.keys():
            if isinstance(obj[k], dict):
                to_cpu(obj[k])
            elif torch.is_tensor(obj[k]) and obj[k].device.type == 'dpcpp':
                obj[k] = obj[k].to('cpu')

    if isinstance(obj, dict):
        obj_copy = copy.deepcopy(obj)
        to_cpu(obj_copy)
    elif torch.is_tensor(obj) and obj.device.type == 'dpcpp':
        obj_copy = copy.deepcopy(obj).to('cpu')
    elif isinstance(obj, torch.nn.Module): 
        obj_copy = copy.deepcopy(obj).to('cpu')
    else:
        obj_copy = obj
    return torch_save(obj_copy, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

torch.save = save