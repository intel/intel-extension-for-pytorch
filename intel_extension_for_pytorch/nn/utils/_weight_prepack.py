import torch
import torch.nn as nn
import warnings

from intel_extension_for_pytorch import optim

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
        self.weight_packed = True
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last) \
            or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)

        # TODO: ".clone()" will make weight shared by multiple module not shared anymore
        # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
        self.weight = nn.Parameter(torch.ops.torch_ipex.convolution_weight_pack(
            dense_module.weight.detach().clone(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups), requires_grad = dense_module.weight.requires_grad)

        if hasattr(dense_module, 'master_weight'):
            self.master_weight = torch.ops.torch_ipex.convolution_weight_pack(
                dense_module.master_weight.detach().clone(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                self.weight.dtype)
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = torch.ops.torch_ipex.convolution_weight_pack(
                dense_module.weight_trail.detach().clone(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups)
        if dense_module.bias is not None:
            self.bias = nn.Parameter(
                dense_module.bias.detach().clone(),
                requires_grad = dense_module.bias.requires_grad)
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        unpack_dtype = self.weight.dtype
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConvNd's parameters"
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()
        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = torch.ops.torch_ipex.convolution_weight_unpack(
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
        assert False, "_IPEXConvNd does not support _load_from_state_dict method"

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
            self.weight_packed)

class _IPEXConv2d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv2d, self).__init__(dense_module)

class _IPEXConv3d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv3d, self).__init__(dense_module)

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
            torch.ops.torch_ipex.linear_weight_pack(dense_module.weight.detach().clone()),
            requires_grad = dense_module.weight.requires_grad)

        if hasattr(dense_module, 'master_weight'):
            self.master_weight = torch.ops.torch_ipex.linear_weight_pack(
                dense_module.master_weight.detach().clone(),
                self.weight.dtype)
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = torch.ops.torch_ipex.linear_weight_pack(
                dense_module.weight_trail.detach().clone())

        if dense_module.bias is not None:
            self.bias = nn.Parameter(
                dense_module.bias.detach().clone(),
                requires_grad = dense_module.bias.requires_grad)
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)

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
                bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()

        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
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

class _IPEXConvTransposeNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'out_channels', 'kernel_size', 'output_padding']

    def __init__(self, dense_module):
        super(_IPEXConvTransposeNd, self).__init__()
        self.out_channels = dense_module.out_channels
        self.in_channels = dense_module.in_channels
        self.kernel_size = dense_module.kernel_size
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        self.output_padding = dense_module.output_padding

    def forward(self, x):
        return torch.ops.torch_ipex.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
            self.kernel_size,
            self.out_channels,
            self.weight_channels_last,
            self.weight_packed)

class _IPEXConvTranspose2d(_IPEXConvTransposeNd):
    def __init__(self, dense_module):
        super(_IPEXConvTranspose2d, self).__init__(dense_module)
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last)
        self.weight_packed = True

        self.weight = nn.Parameter(torch.ops.torch_ipex.conv_transpose2d_weight_pack(
            dense_module.weight.detach().clone(),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation), requires_grad = dense_module.weight.requires_grad)

        if hasattr(dense_module, 'master_weight'):
            self.master_weight = torch.ops.torch_ipex.conv_transpose2d_weight_pack(
                dense_module.master_weight.detach().clone(),
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
                self.weight.dtype)
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = torch.ops.torch_ipex.conv_transpose2d_weight_pack(
                dense_module.weight_trail.detach().clone(),
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation)
        if dense_module.bias is not None:
            self.bias = nn.Parameter(
                dense_module.bias.detach().clone(),
                requires_grad = dense_module.bias.requires_grad)
            if hasattr(dense_module, 'master_bias'):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, 'bias_trail'):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter('bias', None)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        unpack_dtype = self.weight.dtype
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConvTranspose2d's parameters"
        if self.bias is not None:
            if hasattr(self, 'master_bias'):
                bias = self.master_bias
            elif hasattr(self, 'bias_trail'):
                bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
            else:
                bias = self.bias
            destination[prefix + 'bias'] = bias.detach()
        if hasattr(self, 'master_weight'):
            weight = self.master_weight
        elif hasattr(self, 'weight_trail'):
            weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
        else:
            weight = self.weight
        destination[prefix + 'weight'] = torch.ops.torch_ipex.conv_transpose2d_weight_unpack(
            weight.detach(),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
            self.kernel_size,
            self.out_channels,
            self.in_channels,
            self.weight_channels_last,
            unpack_dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXConvTranspose2d does not support _load_from_state_dict method"

IPEX_WEIGHT_PREPACK_MODULE = {
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.ConvTranspose2d,
    torch.nn.Conv3d,
}

def _should_prepack(module, auto_kernel_selection):
    if type(module) not in IPEX_WEIGHT_PREPACK_MODULE:
        return False
    elif isinstance(module, torch.nn.Linear) and not auto_kernel_selection and module.weight.dtype is torch.float:
        # For now we simply distinguish "mkl" and "mkldnn" backend by "weight prepack"
        # Does not prepack Linear for FP32 to choose "mkl" backend
        return False
    return True

def weight_prepack_with_ipex(module, optimizer, params_attr, auto_kernel_selection):
    def convert(m, auto_kernel_selection):
        if _should_prepack(m, auto_kernel_selection) and (m.weight.dtype == torch.float32 or m.weight.dtype == torch.bfloat16):
            weight = m.master_weight if hasattr(m, "master_weight") else m.weight
            if weight not in params_attr:
                params_attr[weight] = {}
            if isinstance(m, torch.nn.Conv2d):
                new_m = _IPEXConv2d(m)
                params_attr[weight].update({
                    'op': torch.nn.Conv2d, 'padding': new_m.padding,
                    'dilation': new_m.dilation, 'stride': new_m.stride,
                    'kernel_size': new_m.kernel_size, 'groups': new_m.groups,
                    'in_channels': new_m.in_channels, 'out_channels': new_m.out_channels,
                    'weight_channels_last': new_m.weight_channels_last})
            elif isinstance(m, torch.nn.Conv3d):
                new_m = _IPEXConv3d(m)
                params_attr[weight].update({
                    'op': torch.nn.Conv3d, 'padding': new_m.padding,
                    'dilation': new_m.dilation, 'stride': new_m.stride,
                    'kernel_size': new_m.kernel_size, 'groups': new_m.groups,
                    'in_channels': new_m.in_channels, 'out_channels': new_m.out_channels,
                    'weight_channels_last': new_m.weight_channels_last})
            elif isinstance(m, torch.nn.Linear):
                try:
                    new_m = _IPEXLinear(m)
                    params_attr[weight].update({
                        'op': torch.nn.Linear,
                        'out_features': new_m.out_features,
                        'in_features': new_m.in_features,
                        'weight_transposed': new_m.weight_transposed})
                except RuntimeError:
                    new_m = m
            elif isinstance(m, torch.nn.ConvTranspose2d):
                if m.padding[0] - m.output_padding[0] + m.stride[0] <= 0 or m.padding[1] - m.output_padding[1] + m.stride[1] <= 0:
                    new_m = m
                else:
                    new_m = _IPEXConvTranspose2d(m)
                    params_attr[weight].update({'op': torch.nn.ConvTranspose2d, \
                                                'padding': new_m.padding, 'stride': new_m.stride, \
                                                'dilation': new_m.dilation, 'kernel_size': new_m.kernel_size, \
                                                'groups': new_m.groups, 'out_channels': new_m.out_channels, \
                                                'in_channels': new_m.in_channels, \
                                                'output_padding': new_m.output_padding, \
                                                'weight_channels_last': new_m.weight_channels_last})
            if 'bf16_param' in params_attr[weight]:
                params_attr[weight]['bf16_param'] = new_m.weight
            elif 'trail' in params_attr[weight]:
                params_attr[weight]['trail'] = new_m.weight_trail
            # update entry from origin weight to packed weight, from origin bias to cloned bias
            new_weight = new_m.master_weight if hasattr(m, "master_weight") else new_m.weight
            params_attr[new_weight] = params_attr.pop(weight)
            params_pair = {weight: new_weight}
            if hasattr(m, 'bias') and m.bias is not None:
                bias = m.master_bias if hasattr(m, "master_bias") else m.bias
                new_bias = new_m.master_bias if hasattr(m, "master_bias") else new_m.bias
                params_pair.update({bias: new_bias})
                if bias in params_attr:
                    if 'bf16_param' in params_attr[bias]:
                        params_attr[bias]['bf16_param'] = new_m.bias
                    elif 'trail' in params_attr[bias]:
                        params_attr[bias]['trail'] = new_m.bias_trail
                    if bias in params_attr:
                        params_attr[new_bias] = params_attr.pop(bias)
            # replace optimizer's param with prepacked param, also prepack its state.
            optim._optimizer_utils.pack_optimizer_params_and_states(
                optimizer, params_pair, params_attr, m.weight.dtype)
            return new_m
        else:
            return m

    def convert_rec(m, auto_kernel_selection):
        new_m = convert(m, auto_kernel_selection)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m, auto_kernel_selection))
        return new_m

    opt_model, opt_optmizer, params_attr = convert_rec(module, auto_kernel_selection), optimizer, params_attr
    if optimizer is not None:
        setattr(optimizer, 'params_attr', params_attr)
        optim._optimizer_utils.patch_load_state_dict(optimizer)
        optim._optimizer_utils.patch_state_dict(optimizer)
    return opt_model, opt_optmizer, params_attr
