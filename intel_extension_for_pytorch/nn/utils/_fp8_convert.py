import torch
from intel_extension_for_pytorch.xpu.fp8.module import FP8Linear


def convert_fp8_model(model):
    def convert_qmodel_recursive(module):
        for name, child in module.named_children():
            if type(child) == torch.nn.Linear:
                qmodule = FP8Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=True if child.bias is not None else False,
                )
                setattr(module, name, qmodule)
            else:
                convert_qmodel_recursive(child)

    print("original model:\n", model)
    convert_qmodel_recursive(model)
    print("converted FP8 model:\n", model)
    return model
