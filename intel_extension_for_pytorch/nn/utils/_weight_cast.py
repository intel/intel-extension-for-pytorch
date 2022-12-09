import torch
import torch.nn as nn
from intel_extension_for_pytorch.optim import _optimizer_utils, _lamb # noqa F401
import types
from ._model_convert import _LSTM

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
    _LSTM,
}


def _save_to_state_dict(self, destination, prefix, keep_vars):
    param_dict = {}
    for name, para in self.named_parameters():
        temp = para
        param_dict.update({name: para})
        if self.master_weight_split:
            temp_para = torch.nn.Parameter(
                torch.ops.torch_ipex.cat_bfloat16_float(para.data, getattr(self, name + '_trail')),
                requires_grad=temp.requires_grad)
            setattr(self, name, temp_para) # noqa B010
        else:
            temp_para = torch.nn.Parameter(
                getattr(self, 'master_' + name),
                requires_grad=temp.requires_grad)
            setattr(self, name, temp_para) # noqa B010

    super(type(self), self)._save_to_state_dict(destination, prefix, keep_vars)
    for p in param_dict:
        origin_param = param_dict[p]
        setattr(self, p, origin_param) # noqa B010


def weight_dtype_convert_with_ipex(module, optimizer, params_attr, master_weight_split, convert_dtype=torch.bfloat16):

    def cast_attr(m, attr, master_weight_split, params_attr, optimizer):
        # cast weight/bias for BF16 or FP16 dtype
        float_param = getattr(m, attr)
        params_attr[float_param] = {}
        if master_weight_split:
            assert convert_dtype == torch.bfloat16, "master_weight_split is only support for bf16 now"
            top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(float_param.data)
            setattr(m, attr + '_trail', bot_half)
            setattr(m, attr, nn.Parameter(top_half.detach(), requires_grad=float_param.requires_grad))
            params_attr[float_param]['trail'] = getattr(m, attr + '_trail')
        else:
            setattr(m, 'master_' + attr, float_param.data)
            if convert_dtype == torch.bfloat16:
                setattr(m, attr, nn.Parameter(float_param.detach().bfloat16(), requires_grad=float_param.requires_grad))
                params_attr[float_param]['bf16_param'] = getattr(m, attr)
            else:
                assert convert_dtype == torch.half, "Only bf16 and fp16 are supported"
                setattr(m, attr, nn.Parameter(float_param.detach().half(), requires_grad=float_param.requires_grad))
                params_attr[float_param]['fp16_param'] = getattr(m, attr)
        # update attr entry, always use params in optimzer as "key"
        # while master weight split, key is m.weight/bias, if not split, key is m.master_weight/master_bias
        attr_name = attr if master_weight_split else 'master_' + attr
        params_attr[getattr(m, attr_name)] = params_attr.pop(float_param)
        _optimizer_utils.refresh_optimizer_params_after_cast(m, attr, float_param, master_weight_split, optimizer)

    def convert(m):
        if type(m) in IPEX_WEIGHT_CAST_MODULE:
            setattr(m, 'master_weight_split', master_weight_split) # noqa B010
            # replace weight/bias
            for name, para in m.named_parameters():
                cast_attr(m, name, master_weight_split, params_attr, optimizer)
            # for resume training reason, we always save float tensors
            # replace module method to ensure return float params while call "state_dict()"
            setattr(m, '_save_to_state_dict', types.MethodType(_save_to_state_dict, m)) # noqa B010
        return m

    def convert_rec(m):
        new_m = convert(m)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m)) # noqa B010
        return new_m

    casted_model, casted_optimizer, params_attr = convert_rec(module), optimizer, params_attr

    if optimizer is not None:
        _optimizer_utils.patch_load_state_dict(casted_optimizer)
        setattr(casted_optimizer, 'params_attr', params_attr) # noqa B010
        if not master_weight_split:
            _optimizer_utils.patch_step_for_master_weight_training(casted_optimizer)
            _optimizer_utils.patch_zero_grad_for_master_weight_training(casted_optimizer)

    return casted_model, casted_optimizer, params_attr
