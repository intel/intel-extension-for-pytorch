import torch
import torch.nn as nn
import _torch_ipex as core
import warnings

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
    def __init__(self, dense_module, dtype):
        super(_IPEXConv2d, self).__init__(dense_module)
        self.dtype = dtype
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last)
        self.weight_prepacked = True
        self.weight = nn.Parameter(torch.ops.torch_ipex.conv2d_weight_prepack(
            # TODO: ".clone()" will make weight shared by multiple module not shared anymore
            # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
            dense_module.weight.detach().clone(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.dtype))
        if dense_module.bias is not None:
            self.bias = nn.Parameter(dense_module.bias.detach().clone())
        else:
            self.register_parameter('bias', None)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConv2d's parameters"
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()
        destination[prefix + 'weight'] = torch.ops.torch_ipex.conv2d_weight_unpack(
            self.weight.detach().clone(),
            self.padding,
            self.stride,
            self.dilation,
            self.kernel_size,
            self.groups,
            self.out_channels,
            self.in_channels,
            self.weight_channels_last,
            self.dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXConv2d does not support _load_from_state_dict method"

class _IPEXLinear(torch.nn.Module):
    def __init__(self, dense_module, dtype):
        super(_IPEXLinear, self).__init__()
        # use in_features, out features and weight_transposed to restore origin 2D weight
        self.dtype = dtype
        self.out_features = dense_module.out_features
        self.in_features = dense_module.in_features
        self.weight_transposed = (
          dense_module.weight.stride()[0] == 1 and
          dense_module.weight.stride()[1] == dense_module.weight.size()[0]
        )
        self.weight = torch.nn.Parameter(
            # TODO:".clone()" will make weight shared by multiple module not shared anymore
            # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
            torch.ops.torch_ipex.linear_weight_prepack(dense_module.weight.detach().clone(), self.dtype)
        )
        if dense_module.bias is not None:
            self.bias = torch.nn.Parameter(dense_module.bias.detach().clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return torch.ops.torch_ipex.ipex_linear(
          x, self.weight, self.out_features, self.in_features, self.bias
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXLinear's parameters"
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()
        destination[prefix + 'weight'] = torch.ops.torch_ipex.linear_weight_unpack(
            self.weight.detach().clone(),
            self.out_features,
            self.in_features,
            self.weight_transposed,
            self.dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXLinear does not support _load_from_state_dict method"

def _weight_prepack_with_ipex(module, dtype=None):
    weight_params_attr = {}
    def convert(m, dtype):
        if isinstance(m, torch.nn.Conv2d):
            new_model = _IPEXConv2d(m, dtype)
            weight_params_attr[new_model.weight] = {'op': torch.nn.Conv2d, 'padding': new_model.padding, \
                                                    'stride': new_model.stride, 'dilation': new_model.dilation, \
                                                    'kernel_size': new_model.kernel_size, 'groups': new_model.groups, \
                                                    'out_channels': new_model.out_channels, 'in_channels': new_model.in_channels, \
                                                    'weight_channels_last': new_model.weight_channels_last, 'dtype': new_model.dtype}
            return new_model
        elif isinstance(m, torch.nn.Linear):
            try:
                new_model = _IPEXLinear(m, dtype)
                weight_params_attr[new_model.weight] = {'op': torch.nn.Linear,
                                                        'out_features': new_model.out_features,
                                                        'in_features': new_model.in_features,
                                                        'weight_transposed': new_model.weight_transposed,
                                                        'dtype': new_model.dtype}
                return new_model
            except:
                warnings.warn(m.__str__()  + " not be packed because weight is not transposed or contiguous")
                return m
        else:
            return m

    def convert_rec(m, dtype):
        new_m = convert(m, dtype)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m, dtype))
        return new_m

    return convert_rec(module, dtype), weight_params_attr
