import torch
import torch.nn as nn
import _torch_ipex as core
import warnings
import types
import copy

class _IPEXConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'out_channels', 'kernel_size']

    def __init__(self, dense_module):
        super(_IPEXConvNd, self).__init__()
        self.out_channels = dense_module.out_channels
        self.in_channels = dense_module.in_channels
        self.kernel_size = dense_module.kernel_size
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups

    def forward(self, x):
        return torch.ops.torch_ipex.convolution_forward(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size,
            self.groups,
            self.out_channels,
            self.weight_channels_last,
            self.weight_prepacked)
class _IPEXConv2d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv2d, self).__init__(dense_module)
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last)
        self.weight_prepacked = True

        # TODO: ".clone()" will make weight shared by multiple module not shared anymore
        # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
        self.weight = nn.Parameter(torch.ops.torch_ipex.conv2d_weight_prepack(
            dense_module.weight.detach().clone(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = torch.ops.torch_ipex.conv2d_weight_prepack(
                dense_module.master_weight.detach().clone(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                self.weight.dtype)
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = torch.ops.torch_ipex.conv2d_weight_prepack(
                dense_module.weight_trail.detach().clone(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups)
        if dense_module.bias is not None:
            self.bias = nn.Parameter(dense_module.bias.detach().clone())
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        unpack_dtype = self.weight.dtype
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConv2d's parameters"
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias =  torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()
        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight =  torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = torch.ops.torch_ipex.conv2d_weight_unpack(
            weight.detach(),
            self.padding,
            self.stride,
            self.dilation,
            self.kernel_size,
            self.groups,
            self.out_channels,
            self.in_channels,
            self.weight_channels_last,
            unpack_dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXConv2d does not support _load_from_state_dict method"

class _IPEXLinear(torch.nn.Module):
    def __init__(self, dense_module):
        super(_IPEXLinear, self).__init__()
        # use in_features, out features and weight_transposed to restore origin 2D weight
        self.out_features = dense_module.out_features
        self.in_features = dense_module.in_features
        self.weight_transposed = (
          dense_module.weight.stride()[0] == 1 and
          dense_module.weight.stride()[1] == dense_module.weight.size()[0]
        )

        # TODO:".clone()" will make weight shared by multiple module not shared anymore
        # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
        self.weight = torch.nn.Parameter(
            torch.ops.torch_ipex.linear_weight_prepack(dense_module.weight.detach().clone())
        )
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = torch.ops.torch_ipex.linear_weight_prepack(
                dense_module.master_weight.detach().clone(),
                self.weight.dtype)
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = torch.ops.torch_ipex.linear_weight_prepack(
                dense_module.weight_trail.detach().clone())
  
        if dense_module.bias is not None:
            self.bias = nn.Parameter(dense_module.bias.detach().clone())
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail

    def forward(self, x):
        return torch.ops.torch_ipex.ipex_linear(
          x, self.weight, self.out_features, self.in_features, self.bias
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXLinear's parameters"
        unpack_dtype = self.weight.dtype
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias =  torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()

        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight =  torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = torch.ops.torch_ipex.linear_weight_unpack(
            weight.detach(),
            self.out_features,
            self.in_features,
            self.weight_transposed, 
            unpack_dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXLinear does not support _load_from_state_dict method"

IPEX_WEIGHT_PREPACK_MODULE = {
  torch.nn.Linear,
  torch.nn.Conv2d,
}

def optimizer_state_dict(self):
    r"""Returns the state of the optimizer as a :class:`dict`.
    This optimizer_state_dict is used to 
    (1) Unpack params/related state
    It contains two entries:

    * state - a dict holding current optimization state. Its content
        differs between optimizer classes.
    * param_groups - a dict containing all parameter groups
    """
    # remove ipex optimizer weight cast wrapper under bf16 case
    if hasattr(self, 'optimizer'):
        opt = self.optimizer
    else:
        opt = self
    
    opt_temp = copy.deepcopy(opt)
    params_attr_ = {}
    is_bf16 = hasattr(self, 'master_weight_split')
    unpack_dtype = torch.bfloat16 if is_bf16 else torch.float
    if is_bf16 and self.master_weight_split == False:
        for _, values in self.params_attr.items():
            master_weight = values['master_param']
            params_attr_[master_weight] = values
    else:
        params_attr_ = self.params_attr
    for (k1, _), (_, v2) in zip(opt.state.items(), opt_temp.state.items()):
        if k1 in params_attr_:
            params_attr = params_attr_[k1]
            for state_key, state_value in v2.items():
              if isinstance(state_value, torch.Tensor):
                    if 'op' in params_attr:
                    # Secondly, unpack releated states
                        if params_attr['op'] is torch.nn.Conv2d:
                            state_value = torch.ops.torch_ipex.conv2d_weight_unpack(
                                state_value,
                                params_attr['padding'],
                                params_attr['stride'],
                                params_attr['dilation'],
                                params_attr['kernel_size'],
                                params_attr['groups'],
                                params_attr['out_channels'],
                                params_attr['in_channels'],
                                params_attr['weight_channels_last'],
                                unpack_dtype)
                        elif params_attr['op'] is torch.nn.Linear:
                            state_value = torch.ops.torch_ipex.linear_weight_unpack(
                                state_value,
                                params_attr['out_features'],
                                params_attr['in_features'],
                                params_attr['weight_transposed'],
                                unpack_dtype)
                        else:
                            assert False, "unsupported op to unpack"
                    v2[state_key] = state_value
    return opt_temp.state_dict()


def _optimizer_convert_for_weight_prepack(optimizer, weight_pair, bias_pair, attrs, attr_key):
    """
    1. convert user's optimizer weights and related states to packed format
    While optimizer is maintain "master weight", the key in attrs is "weight", 
    Need pass "weight" here as attr_key visit attr.
    2. convert user's optimizer bias to new model's bias since there is a "clone"
    """
    if optimizer is None:
        return
    para_pair = {}
    para_pair.update(weight_pair)
    if bias_pair is not None:
        para_pair.update(bias_pair)
    if hasattr(optimizer, 'optimizer'):
        # remove ipex weight cast optimizer wrapper
        optimizer = optimizer.optimizer
    pack_dtype = attr_key.dtype
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            if p in para_pair:
                new_param = para_pair[p]
                group['params'][i] = new_param
                # copy optimizer's state.
                if p in optimizer.state:
                    optimizer.state[new_param] = optimizer.state.pop(p)
                    if p in weight_pair:
                        # Prepack the state according to the prepacked weight.
                        # it covers both conv and linear now. TODO: LSTM or other ops.
                        if attr_key in attrs:
                            attr = attrs[attr_key]
                            state = optimizer.state[new_param]
                            for state_key, state_value in state.items():
                                if isinstance(state_value, torch.Tensor):
                                    assert state_value.size() == p.size(), \
                                        "Only support the optimizer state's size has the same shape with model's parameter."
                                    if attr['op'] is torch.nn.Conv2d:
                                        value_temp = state_value.to(memory_format=torch.channels_last) \
                                            if attr['weight_channels_last'] else state_value
                                        state[state_key] = torch.ops.torch_ipex.conv2d_weight_prepack(
                                            value_temp,
                                            attr['padding'],
                                            attr['stride'],
                                            attr['dilation'],
                                            attr['groups'],
                                            pack_dtype)
                                    elif attr['op'] is torch.nn.Linear:
                                        state[state_key] = torch.ops.torch_ipex.linear_weight_prepack(
                                            state_value,
                                            pack_dtype)

def _weight_prepack_with_ipex(module, optimizer, params_attr):
    def convert(m):
        if type(m) in IPEX_WEIGHT_PREPACK_MODULE:
            if (m.weight not in params_attr):
                params_attr[m.weight] = {}
            if isinstance(m, torch.nn.Conv2d):
                new_m = _IPEXConv2d(m)
                params_attr[m.weight].update({'op': torch.nn.Conv2d, \
                                              'padding': new_m.padding, 'stride': new_m.stride, \
                                              'dilation': new_m.dilation, 'kernel_size': new_m.kernel_size, \
                                              'groups': new_m.groups, 'out_channels': new_m.out_channels, \
                                              'in_channels': new_m.in_channels, \
                                              'weight_channels_last': new_m.weight_channels_last})
            elif isinstance(m, torch.nn.Linear):
              try:
                  new_m = _IPEXLinear(m)
                  params_attr[m.weight].update({'op': torch.nn.Linear,
                                                        'out_features': new_m.out_features,
                                                        'in_features': new_m.in_features,
                                                        'weight_transposed': new_m.weight_transposed})
              except:
                  warnings.warn(m.__str__()  + " not be packed because weight is not transposed or contiguous")
                  new_m = m
            if 'master_param' in params_attr[m.weight]:
                params_attr[m.weight]['master_param'] = new_m.master_weight
            elif 'trail' in params_attr[m.weight]:
                params_attr[m.weight]['trail'] = new_m.weight_trail
            # update entry from origin weight to packed weight, from origin bias to cloned bias
            params_attr[new_m.weight] = params_attr.pop(m.weight)
            if hasattr(m, 'bias') and m.bias != None and m.bias in params_attr:
                params_attr[new_m.bias] = params_attr.pop(m.bias)
            # replace optimizer's param with prepacked param, also prepack its state.
            bias_pair = None
            if hasattr(optimizer, 'master_weight_split') and optimizer.master_weight_split == False:
                weight_pair = {m.master_weight: new_m.master_weight}
                # under this case, optimizer is maintail master weight instead of weight
                if hasattr(m, 'bias') and m.bias != None:
                      bias_pair = {m.master_bias: new_m.master_bias}
            else:
                weight_pair = {m.weight: new_m.weight}
                if hasattr(m, 'bias') and m.bias != None:
                      bias_pair = {m.bias: new_m.bias}
            _optimizer_convert_for_weight_prepack(
                optimizer, weight_pair, bias_pair, params_attr, new_m.weight)
            return new_m
        else:
            return m

    def convert_rec(m):
        new_m = convert(m)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m))
        return new_m
    
    opt_model, opt_optmizer, params_attr = convert_rec(module), optimizer, params_attr
    if optimizer is not None:
        setattr(optimizer, 'params_attr', params_attr)
        setattr(optimizer, 'state_dict', types.MethodType(optimizer_state_dict, optimizer))
    return opt_model, opt_optmizer, params_attr
