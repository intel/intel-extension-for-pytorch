import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.optimize_transformers.modules.transformer_modules.Linear import INT4Linear

def convert_qmodel(model, dtype, group_size):
    def convert_qmodel_recursive(module):
        for name, child in module.named_children():
            if type(child) == torch.nn.Linear:
                qmodule = INT4Linear(in_features=child.in_features, out_features=child.out_features,
                        group_size=group_size, bias=True if child.bias is not None else False, dtype=dtype)
                setattr(module, name, qmodule)
            else:
                convert_qmodel_recursive(child)
    convert_qmodel_recursive(model)
    return model
