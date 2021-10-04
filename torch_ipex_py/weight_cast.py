import torch
import torch.nn as nn
import intel_extension_for_pytorch._C as core
import warnings
from .weight_prepack import _IPEXConvNd, _IPEXConv2d, _IPEXLinear
from .optimizer_utils import IPEX_OPTIMIZER_MAPPING, _ipex_optimizer
from collections import OrderedDict, namedtuple
import types
import copy

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
}

def _save_to_state_dict(self, destination, prefix, keep_vars):
    # cast weight
    temp_weight = self.weight
    if self.master_weight_split:
        self.weight =  torch.nn.Parameter(torch.ops.torch_ipex.cat_bfloat16_float(self.weight.data, self.weight_trail))
    else:
        self.weight = torch.nn.Parameter(self.master_weight)
    # cast bias
    if hasattr(self, 'bias') and self.bias is not None:
        temp_bias = self.bias
        if self.master_weight_split:
            self.bias =  torch.nn.Parameter(torch.ops.torch_ipex.cat_bfloat16_float(self.bias.data, self.bias_trail))
        else:
            self.bias =  torch.nn.Parameter(self.master_bias)
    super(type(self), self)._save_to_state_dict(destination, prefix, keep_vars)
    self.weight = temp_weight
    if hasattr(self, 'bias') and self.bias is not None:
        self.bias = temp_bias

def _weight_dtype_convert_with_ipex(module, optimizer, params_attr):
    master_weight_split = type(optimizer) in IPEX_OPTIMIZER_MAPPING
    def cast_optimizer_params_and_states(m, attr, float_param, master_weight_split):
        if optimizer is None:
            return
        # update params
        for group in optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p is float_param:
                    if master_weight_split:
                        group['params'][i] = getattr(m, attr)
                    else:
                        group['params'][i] = getattr(m, 'master_' + attr)
                    # update optimizer's state.
                    new_param = group['params'][i]
                    if p in optimizer.state:
                        optimizer.state[new_param] = optimizer.state.pop(p)

    def cast_attr(m, attr, master_weight_split, params_attr, optimizer):
        # cast weight/bias for BF16 dtype
        float_param = getattr(m, attr)
        params_attr[float_param] = {}
        if master_weight_split:
            top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(float_param.data)
            setattr(m, attr + '_trail', bot_half)
            setattr(m, attr, nn.Parameter(top_half.detach()))
            params_attr[float_param]['trail'] = getattr(m, attr + '_trail')
        else:
            setattr(m, 'master_' + attr, float_param.data)
            setattr(m, attr, nn.Parameter(float_param.detach().bfloat16()))
            params_attr[float_param]['master_param'] = getattr(m, 'master_' + attr)
        # update attr entry
        params_attr[getattr(m, attr)] = params_attr.pop(float_param)
        cast_optimizer_params_and_states(m, attr, float_param, master_weight_split)

    def convert(m):
        if type(m) in IPEX_WEIGHT_CAST_MODULE:
            setattr(m, 'master_weight_split', master_weight_split)
            # replace weight
            cast_attr(m, 'weight', master_weight_split, params_attr, optimizer)
            if hasattr(m, 'bias') and m.bias != None:
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
        casted_optimizer = _ipex_optimizer(casted_optimizer, params_attr)
    return casted_model, casted_optimizer, params_attr
