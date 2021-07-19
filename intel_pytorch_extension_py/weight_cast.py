import torch
import torch.nn as nn
import _torch_ipex as core
import warnings
from .weight_prepack import _IPEXConvNd, _IPEXConv2d, _IPEXLinear
from .optimizer_utils import IPEX_OPTIMIZER_MAPPING


class _IPEXLinearWeightCastWrapper(_IPEXLinear):
    def __init__(self, float_module, master_weight_split):
        super(_IPEXLinear, self).__init__()
        self.dtype = float_module.dtype
        self.out_features = float_module.out_features
        self.in_features = float_module.in_features
        self.weight_transposed = float_module.weight_transposed
        self.weight = float_module.weight
        if float_module.bias is not None:
            self.bias = nn.Parameter(float_module.bias.detach())
        else:
            self.register_parameter('bias', None)
        self.master_weight_split = master_weight_split
        if self.master_weight_split:
            top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(float_module.weight)
            self.trail = bot_half
            self.weight = nn.Parameter(top_half.detach())
        else:
            self.master_weight = float_module.weight.data
            self.weight = nn.Parameter(self.master_weight.detach().bfloat16())

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        temp_weight = self.weight
        if self.master_weight_split:
            self.weight =  torch.nn.Parameter(torch.ops.torch_ipex.cat_bfloat16_float(self.weight.data, self.trail))
        else:
            self.weight = torch.nn.Parameter(self.master_weight)
        super(_IPEXLinearWeightCastWrapper, self)._save_to_state_dict(destination, prefix, keep_vars)
        self.weight = temp_weight
    

class _IPEXConv2dWeightCastWrapper(_IPEXConv2d):
    def __init__(self, float_module, master_weight_split):
        super(_IPEXConvNd, self).__init__()
        self.out_channels = float_module.out_channels
        self.in_channels = float_module.in_channels
        self.kernel_size = float_module.kernel_size
        self.stride = float_module.stride
        self.padding = float_module.padding
        self.dilation = float_module.dilation
        self.groups = float_module.groups

        self.dtype = float_module.dtype
        self.weight_channels_last = float_module.weight_channels_last
        self.weight_prepacked = float_module.weight_prepacked
        self.weight = float_module.weight
        if float_module.bias is not None:
            self.bias = float_module.bias
        else:
            self.register_parameter('bias', None)
        self.master_weight_split = master_weight_split
        if self.master_weight_split:
            top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(float_module.weight)
            self.trail = bot_half
            self.weight = nn.Parameter(top_half.detach())
        else:
            self.master_weight = float_module.weight.data
            self.weight = nn.Parameter(self.master_weight.detach().bfloat16())

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        temp_weight = self.weight
        if self.master_weight_split:
            self.weight =  torch.nn.Parameter(torch.ops.torch_ipex.cat_bfloat16_float(self.weight.data, self.trail))
        else:
            self.weight = torch.nn.Parameter(self.master_weight)
        super(_IPEXConv2dWeightCastWrapper, self)._save_to_state_dict(destination, prefix, keep_vars)
        self.weight = temp_weight

IPEX_WEIGHT_CAST_MODULE_MAPPING = {
  _IPEXLinear: _IPEXLinearWeightCastWrapper,
  _IPEXConv2d: _IPEXConv2dWeightCastWrapper,
}

def _optimizer_convert_for_weight_cast(optimizer, origin_model, prepacked_model, attrs):
    """
    Convert user's optimizer state to expected state, for example, some optimizer has
    momentum_buffer, need make sure the momentum_buffer is also prepacked if the corresponding
    parameter has been prepacked.
    """
    if optimizer is None:
        return
    master_weight_split = type(optimizer) in IPEX_OPTIMIZER_MAPPING
    dict_param = {}
    for k, value in zip(origin_model.parameters(), prepacked_model.parameters()):
        dict_param[k] = value
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            if p in dict_param:
                new_model_param = dict_param[p]
                if new_model_param in attrs and not master_weight_split:
                    new_param = attrs[new_model_param]['master_weight']
                else:
                    new_param = new_model_param
                group['params'][i] = new_param
                # copy optimizer's state.
                if p in optimizer.state:
                    optimizer.state[new_param] = optimizer.state.pop(p)
                    if new_param in attrs:
                        attr = attrs[new_param]
                        state = optimizer.state[new_param]

def _weight_dtype_convert_with_ipex(module, optimizer, weight_params_attr = {}):
    master_weight_split = type(optimizer) in IPEX_OPTIMIZER_MAPPING
    def convert(m):
        if type(m) in IPEX_WEIGHT_CAST_MODULE_MAPPING.keys():
            new_module_cls = IPEX_WEIGHT_CAST_MODULE_MAPPING[type(m)]
            new_model = new_module_cls(m, master_weight_split)
            if master_weight_split:
                weight_params_attr[m.weight]['trail'] = new_model.trail
            else:
                weight_params_attr[m.weight]['master_weight'] = new_model.master_weight
            # update attr entry
            weight_params_attr[new_model.weight] = weight_params_attr.pop(m.weight)
            _optimizer_convert_for_weight_cast(optimizer, m, new_model, weight_params_attr)
            return new_model
        else:
            return m

    def convert_rec(m):
        new_m = convert(m)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m))
        return new_m

    return convert_rec(module), optimizer, weight_params_attr
