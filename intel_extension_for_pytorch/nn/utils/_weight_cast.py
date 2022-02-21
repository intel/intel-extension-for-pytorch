import torch
import torch.nn as nn
from intel_extension_for_pytorch.optim import _optimizer_utils, _lamb
import types

# IPEX does not cast all module parameters for acc reason, such as BN
IPEX_WEIGHT_CAST_MODULE = {
    # align with auto cast white list
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    # ipex support
    torch.nn.EmbeddingBag,
    torch.nn.Embedding,
}

def _save_to_state_dict(self, destination, prefix, keep_vars):
    # cast weight
    temp_weight = self.weight
    if self.master_weight_split:
        self.weight = torch.nn.Parameter(
            torch.ops.torch_ipex.cat_bfloat16_float(self.weight.data, self.weight_trail),
            requires_grad=temp_weight.requires_grad)
    else:
        self.weight = torch.nn.Parameter(
            self.master_weight,
            requires_grad=temp_weight.requires_grad)
    # cast bias
    if hasattr(self, 'bias') and self.bias is not None:
        temp_bias = self.bias
        if self.master_weight_split:
            self.bias = torch.nn.Parameter(
                torch.ops.torch_ipex.cat_bfloat16_float(self.bias.data, self.bias_trail),
                requires_grad=temp_bias.requires_grad)
        else:
            self.bias = torch.nn.Parameter(
                self.master_bias,
                requires_grad=temp_bias.requires_grad)
    super(type(self), self)._save_to_state_dict(destination, prefix, keep_vars)
    self.weight = temp_weight
    if hasattr(self, 'bias') and self.bias is not None:
        self.bias = temp_bias

def weight_dtype_convert_with_ipex(module, optimizer, params_attr, master_weight_split):

    def cast_attr(m, attr, master_weight_split, params_attr, optimizer):
        # cast weight/bias for BF16 dtype
        float_param = getattr(m, attr)
        params_attr[float_param] = {}
        if master_weight_split:
            top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(float_param.data)
            setattr(m, attr + '_trail', bot_half)
            setattr(m, attr, nn.Parameter(top_half.detach(), requires_grad=float_param.requires_grad))
            params_attr[float_param]['trail'] = getattr(m, attr + '_trail')
        else:
            setattr(m, 'master_' + attr, float_param.data)
            setattr(m, attr, nn.Parameter(float_param.detach().bfloat16(), requires_grad=float_param.requires_grad))
            params_attr[float_param]['bf16_param'] = getattr(m, attr)
        # update attr entry, always use params in optimzer as "key"
        # while master weight split, key is m.weight/bias, if not split, key is m.master_weight/master_bias
        attr_name = attr if master_weight_split else 'master_' + attr
        params_attr[getattr(m, attr_name)] = params_attr.pop(float_param)
        _optimizer_utils.refresh_optimizer_params_after_cast(m, attr, float_param, master_weight_split, optimizer)

    def convert(m):
        if type(m) in IPEX_WEIGHT_CAST_MODULE:
            setattr(m, 'master_weight_split', master_weight_split)
            # replace weight
            cast_attr(m, 'weight', master_weight_split, params_attr, optimizer)
            if hasattr(m, 'bias') and m.bias is not None:
                # replace bias
                cast_attr(m, 'bias', master_weight_split, params_attr, optimizer)
            # for resume training reason, we always save float tensors
            # replace module method to ensure return float params while call "state_dict()"
            setattr(m, '_save_to_state_dict', types.MethodType(_save_to_state_dict, m))
        return m

    def convert_rec(m):
        new_m = convert(m)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m))
        return new_m

    casted_model, casted_optimizer, params_attr = convert_rec(module), optimizer, params_attr

    if optimizer is not None:
        _optimizer_utils.patch_load_state_dict(casted_optimizer)
        setattr(casted_optimizer, 'params_attr', params_attr)
        if not master_weight_split:
            _optimizer_utils.patch_step_for_master_weight_training(casted_optimizer)
            _optimizer_utils.patch_zero_grad_for_master_weight_training(casted_optimizer)

    return casted_model, casted_optimizer, params_attr
