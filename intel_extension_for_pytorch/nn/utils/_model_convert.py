import torch
from ._parameter_wrapper import get_shared_parameter_status, patch_state_dict


def replace_customized_linear_with_linear(model):
    if isinstance(model, torch.jit.ScriptModule):
        return
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Linear) and child.__class__.__name__ in [
                "FalconLinear",
                "Linear",
            ]:
                new_m = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=False if child.bias is None else True,
                )
                new_m.weight = child.weight
                if child.bias is not None:
                    new_m.bias = child.bias
                setattr(model, child_name, new_m)
            else:
                replace_customized_linear_with_linear(child)


def replace_dropout_with_identity(model):
    # replace dropout with identity during inference, so that aten::dropout won't be on the JIT graph.
    # This optimization may provide more fusion opportunites on the graph.
    if isinstance(model, torch.jit.ScriptModule):
        return
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                setattr(model, child_name, torch.nn.Identity())
            else:
                replace_dropout_with_identity(child)


def convert_model_data_type(model, dtype):
    # convert weights(bias) of model to dtype to reduce dtype reorder
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], "model convert only support bf16 and fp16"

    params_attr = {}
    get_shared_parameter_status(model, params_attr)

    for _, param in model.named_parameters():
        if param is None:
            continue
        if params_attr[param].can_cast_inference(dtype):
            params_attr[param].cast_for_inference(dtype)

    patch_state_dict(model, params_attr, "inference")
    return params_attr, model
