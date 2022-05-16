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
        self.prepack_input_shape = dense_module.input_shape if hasattr(dense_module, "input_shape") else []
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last) \
            or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)

        # TODO: ".clone()" will make weight shared by multiple module not shared anymore
        # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
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
        # create conv op context
        self.ctx = torch.ops.ipex_prepack.convolution_prepack(
            dense_module.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
            self.weight_channels_last, self.prepack_input_shape
        )

        self.weight = nn.Parameter(self.ctx.get_weight(), requires_grad = dense_module.weight.requires_grad)

        # pack master_weight or weight_trail if needed
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = self.ctx.pack(
                dense_module.master_weight.detach().clone()
            )
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = self.ctx.pack(
                dense_module.weight_trail.detach().clone(),
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
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
        destination[prefix + 'weight'] = self.ctx.to_public(weight.detach())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXConvNd does not support _load_from_state_dict method"

    def forward(self, x):
        return torch.ops.torch_ipex.convolution_forward(x, self.weight, self.bias, self.ctx.get_data_handle())

class _IPEXConv1d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv1d, self).__init__(dense_module)

class _IPEXConv2d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv2d, self).__init__(dense_module)

class _IPEXConv3d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv3d, self).__init__(dense_module)

class _IPEXLinear(torch.nn.Module):
    def __init__(self, dense_module):
        super(_IPEXLinear, self).__init__()

        # prepare batch size
        self.batch_size_collapsed = None
        if hasattr(dense_module, "input_shape"):
            self.batch_size_collapsed = 1
            for i in range(len(dense_module.input_shape) - 1):
                self.batch_size_collapsed *= dense_module.input_shape[i]

        # TODO:".clone()" will make weight shared by multiple module not shared anymore
        # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
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

        # create linear op context
        self.ctx = torch.ops.ipex_prepack.linear_prepack(
            dense_module.weight, self.bias, self.batch_size_collapsed
        )

        self.weight = nn.Parameter(self.ctx.get_weight(), requires_grad = dense_module.weight.requires_grad)

        # pack master_weight or weight_trail if needed
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = self.ctx.pack(
                dense_module.master_weight.detach().clone()
            )
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = self.ctx.pack(
                dense_module.weight_trail.detach().clone(),
            )

    def forward(self, x):
        return torch.ops.torch_ipex.ipex_linear(
            x, self.weight, self.bias, self.ctx.get_data_handle()
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXLinear's parameters"
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
        destination[prefix + 'weight'] = self.ctx.to_public(weight.detach())

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
        self.groups = dense_module.groups
        self.dilation = dense_module.dilation
        self.output_padding = dense_module.output_padding
        self.prepack_input_shape = dense_module.input_shape if hasattr(dense_module, "input_shape") else []
        self.weight_channels_last = dense_module.weight.is_contiguous(memory_format=torch.channels_last) \
            or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)

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
        # create conv op context
        self.ctx = torch.ops.ipex_prepack.conv_transpose_prepack(
            dense_module.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation,
            self.weight_channels_last, self.prepack_input_shape
        )

        self.weight = nn.Parameter(self.ctx.get_weight(), requires_grad = dense_module.weight.requires_grad)

        # pack master_weight or weight_trail if needed
        if hasattr(dense_module, 'master_weight'):
            self.master_weight = self.ctx.pack(
                dense_module.master_weight.detach().clone())
        elif hasattr(dense_module, 'weight_trail'):
            self.weight_trail = self.ctx.pack(
                dense_module.weight_trail.detach().clone()
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert not keep_vars, "can not using keep_vars true when to save _IPEXConvTransposeNd's parameters"
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
        destination[prefix + 'weight'] = self.ctx.to_public(weight.detach())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        assert False, "_IPEXConvTransposeNd does not support _load_from_state_dict method"

    def forward(self, x):
        return torch.ops.torch_ipex.conv_transpose(
            x,
            self.weight,
            self.bias,
            self.ctx.get_data_handle())

class _IPEXConvTranspose2d(_IPEXConvTransposeNd):
    def __init__(self, dense_module):
        super(_IPEXConvTranspose2d, self).__init__(dense_module)

class _IPEXConvTranspose3d(_IPEXConvTransposeNd):
    def __init__(self, dense_module):
        super(_IPEXConvTranspose3d, self).__init__(dense_module)

IPEX_WEIGHT_PREPACK_MODULE = {
    torch.nn.Linear: _IPEXLinear,
    torch.nn.Conv2d: _IPEXConv2d,
    torch.nn.Conv3d: _IPEXConv3d,
    torch.nn.Conv1d: _IPEXConv1d,
    torch.nn.ConvTranspose2d: _IPEXConvTranspose2d,
    torch.nn.ConvTranspose3d: _IPEXConvTranspose3d,
}

def _should_prepack(module, auto_kernel_selection):
    if type(module) not in IPEX_WEIGHT_PREPACK_MODULE:
        return False
    # If hook is on `weight` or `bias`, will not prepack.
    if module._forward_pre_hooks is not None:
        for _, hook in module._forward_pre_hooks.items():
            if hook.name == 'weight' or hook.name == 'bias':
                return False
    if module._forward_hooks is not None:
        for _, hook in module._forward_hooks.items():
            if hook.name == 'weight' or hook.name == 'bias':
                return False
    if module._backward_hooks is not None:
        for _, hook in module._backward_hooks.items():
            if hook.name == 'weight' or hook.name == 'bias':
                return False
    if isinstance(module, torch.nn.Linear) and not auto_kernel_selection and module.weight.dtype is torch.float:
        # For now we simply distinguish "mkl" and "mkldnn" backend by "weight prepack"
        # Does not prepack Linear for FP32 to choose "mkl" backend
        return False
    if isinstance(module, torch.nn.ConvTranspose2d):
        if module.padding[0] - module.output_padding[0] + module.stride[0] <= 0:
            return False
        if module.padding[1] - module.output_padding[1] + module.stride[1] <= 0:
            return False
    if isinstance(module, torch.nn.ConvTranspose3d):
        if module.padding[0] - module.output_padding[0] + module.stride[0] <= 0:
            return False
        if module.padding[1] - module.output_padding[1] + module.stride[1] <= 0:
            return False
        if module.padding[2] - module.output_padding[2] + module.stride[2] <= 0:
            return False
    return True

def weight_prepack_with_ipex(module, optimizer, params_attr, auto_kernel_selection):
    def convert(m, optimizer, params_attr, auto_kernel_selection):
        if _should_prepack(m, auto_kernel_selection) and (m.weight.dtype == torch.float32 or m.weight.dtype == torch.bfloat16):
            weight = m.master_weight if hasattr(m, "master_weight") else m.weight
            if weight not in params_attr:
                params_attr[weight] = {}
            new_m = IPEX_WEIGHT_PREPACK_MODULE[type(m)](m)
            params_attr[weight].update({
                'op': type(m),
                'ctx': new_m.ctx})
            if hasattr(new_m, "weight_channels_last"):
                params_attr[weight]['weight_channels_last'] = new_m.weight_channels_last,
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

    def convert_rec(m, optimizer, params_attr, auto_kernel_selection):
        new_m = convert(m, optimizer, params_attr, auto_kernel_selection)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m, optimizer, params_attr, auto_kernel_selection)[0])
        return new_m, optimizer, params_attr

    opt_model, opt_optmizer, params_attr = convert_rec(module, optimizer, params_attr, auto_kernel_selection)
    if opt_optmizer is not None:
        setattr(opt_optmizer, 'params_attr', params_attr)
        optim._optimizer_utils.patch_load_state_dict(opt_optmizer)
        optim._optimizer_utils.patch_state_dict(opt_optmizer)
    return opt_model, opt_optmizer, params_attr

def record_input_shape_for_prepack(module, sample_input):

    def hook_function(self, input):
        # input for linear/conv/transpose conv received here will be Tuple[Tensor]
        self.input_shape = input[0].shape

    def register_hook_function(module):
        if type(module) in [torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            module.register_forward_pre_hook(hook_function)

    def register_hook_function_rec(module):
        register_hook_function(module)
        for child in module.children():
            register_hook_function_rec(child)

    hook_function.name = "input_shape"
    register_hook_function_rec(module)
    module(*sample_input)
