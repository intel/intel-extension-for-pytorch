import torch
from .fx import *

def _replace_dropout_with_identity(model):
    # replace dropout with identity during inference, so that aten::dropout won't be on the JIT graph.
    # This optimization may provide more fusion opportunites on the graph.
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                setattr(model, child_name, torch.nn.Identity())
            else:
                _replace_dropout_with_identity(child)

def convert_module_data_type(module, dtype):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        weight_data = module.weight.detach().clone().to(dtype)
        module.weight.data = weight_data
        if module.bias is not None:
            bias_data = module.bias.detach().clone().to(dtype)
            module.bias.data = bias_data
    for child in module.children():
        convert_module_data_type(child, dtype)
    return module

def optimize(model, dtype=torch.bfloat16, level='O1'):
    optimized_model = conv_bn_fuse(model)
    if dtype == torch.bfloat16:
        optimized_model = convert_module_data_type(optimized_model, torch.bfloat16)
    return optimized_model
