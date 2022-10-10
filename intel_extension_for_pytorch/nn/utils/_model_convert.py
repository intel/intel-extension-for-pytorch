import torch
import warnings


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


def convert_module_data_type(module, dtype):
    # convert weights(bias) of module to dtype to reduce dtype reorder
    module_convert_list = [torch.nn.Conv2d,
                           torch.nn.Conv3d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d,
                           torch.nn.Linear,
                           torch.nn.Embedding,
                           torch.nn.LSTM]
    for module_cls in module_convert_list:
        if isinstance(module, module_cls):
            if module_cls is torch.nn.LSTM:
                for name, param in module.named_parameters():
                    ori_data = getattr(getattr(module, name), "data") # noqa B010
                    ori_data_dtype = ori_data.dtype
                    if ori_data_dtype == torch.float or ori_data_dtype == torch.bfloat16:
                        casted_data = ori_data.detach().clone().to(dtype)
                        setattr(getattr(module, name), "data", casted_data) # noqa B010
                    else:
                        warnings.warn(
                            f"WARNING: Can't convert model's parameters dtyep from {ori_data_dtype} to {dtype}")
                        break
            else:
                ori_data_dtype = module.weight.dtype
                # Assume weight and bias have same dtype, only need check weight dtype here.
                if ori_data_dtype == torch.float or ori_data_dtype == torch.bfloat16 or ori_data_dtype == torch.float16:
                    weight_data = module.weight.detach().clone().to(dtype)
                    module.weight.data = weight_data
                    if hasattr(module, 'bias') and module.bias is not None:
                        bias_data = module.bias.detach().clone().to(dtype)
                        module.bias.data = bias_data
                else:
                    warnings.warn(f"WARNING: Can't convert model's parameters dtype from {ori_data_dtype} to {dtype}")
            break
    for child in module.children():
        convert_module_data_type(child, dtype)
    return module
