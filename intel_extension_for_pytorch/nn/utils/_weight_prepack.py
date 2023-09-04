import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import os
import pkg_resources
from intel_extension_for_pytorch import optim

logger = logging.getLogger(__name__)


def may_import_deepspeed_modules():
    try:
        # import deepspeed in a global space will raise circular import error
        # intel-extension-for-deepspeed imports both IPEX and deepspeed
        from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

        return LinearAllreduce, LinearLayer
    except ImportError:
        return None


# installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
# if "deepspeed" in installed_pkg:
#     from deepspeed import comm

#     def _all_reduce(self, reduceOp, tag, ranks, group_size):
#         comm.all_reduce(self, async_op=False)
#         return self

#     ds_comm = torch.library.Library("deepspeed_comm", "DEF")
#     ds_comm.define(
#         "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor"
#     )
#     ds_comm_lib_cpu = torch.library.Library("deepspeed_comm", "IMPL", "CPU")
#     ds_comm_lib_cpu.impl("all_reduce", _all_reduce)


def _ipex_module_load_from_state_dict_(self, state_dict, prefix):
    w_name = prefix + "weight"
    b_name = prefix + "bias"
    loaded_weight = state_dict[w_name]
    if b_name in state_dict:
        loaded_bias = state_dict[b_name]
        self.bias_wrapper.load(self, loaded_bias)
    self.weight_wrapper.load(self, loaded_weight)


class _IPEXConvNd(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "out_channels",
        "kernel_size",
    ]

    def __init__(self):
        super(_IPEXConvNd, self).__init__()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert (
            not keep_vars
        ), "can not using keep_vars true when to save _IPEXConvNd's parameters"
        super(_IPEXConvNd, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        with torch.no_grad():
            _ipex_module_load_from_state_dict_(self, state_dict, prefix)

    def forward(self, x):
        if self.padding_mode != "zeros":
            return torch.ops.torch_ipex.convolution_forward(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.ctx.get_data_handle(),
                self.weight_size,
                self._real_padding,
                self.stride,
                self.dilation,
                self.weight_channels_last,
            )
        return torch.ops.torch_ipex.convolution_forward(
            x,
            self.weight,
            self.bias,
            self.ctx.get_data_handle(),
            self.weight_size,
            self._real_padding,
            self.stride,
            self.dilation,
            self.weight_channels_last,
        )


class _IPEXConv1d(_IPEXConvNd):
    def __init__(self):
        super(_IPEXConv1d, self).__init__()


class _IPEXConv2d(_IPEXConvNd):
    def __init__(self):
        super(_IPEXConv2d, self).__init__()


class _IPEXConv3d(_IPEXConvNd):
    def __init__(self):
        super(_IPEXConv3d, self).__init__()


class _IPEXLinear(torch.nn.Module):
    def __init__(self):
        super(_IPEXLinear, self).__init__()

    def post_ipex_gemm(self, output):
        return output

    def forward(self, x):
        if self.use_dnnl:
            output = torch.ops.torch_ipex.ipex_linear(
                x, self.weight, self.bias, self.ctx.get_data_handle(), self.out_features
            )
        else:
            output = torch.ops.torch_ipex.ipex_MKLSGEMM(
                x, self.weight, self.bias, self.ctx.get_data_handle(), self.out_features
            )

        return self.post_ipex_gemm(output)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert (
            not keep_vars
        ), "can not using keep_vars true when to save _IPEXLinear's parameters"
        super(_IPEXLinear, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        with torch.no_grad():
            _ipex_module_load_from_state_dict_(self, state_dict, prefix)


class _IPEXLinearAllreduce(_IPEXLinear):
    def __init__(self):
        super(_IPEXLinearAllreduce, self).__init__()

    def post_ipex_gemm(self, output):
        if self.mp_group is not None:
            torch.ops.deepspeed_comm.all_reduce(
                output,
                "sum",
                "",
                list(torch.arange(int(os.environ["WORLD_SIZE"]))),
                int(os.environ["WORLD_SIZE"]),
            )
        if self.module_bias is not None:
            output += self.module_bias
        return output


class _IPEXConvTransposeNd(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "out_channels",
        "kernel_size",
        "output_padding",
    ]

    def __init__(self):
        super(_IPEXConvTransposeNd, self).__init__()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert (
            not keep_vars
        ), "can not using keep_vars true when to save _IPEXConvTransposeNd's parameters"
        super(_IPEXConvTransposeNd, self)._save_to_state_dict(
            destination, prefix, keep_vars
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        with torch.no_grad():
            _ipex_module_load_from_state_dict_(self, state_dict, prefix)

    def forward(self, x):
        return torch.ops.torch_ipex.conv_transpose(
            x,
            self.weight,
            self.bias,
            self.ctx.get_data_handle(),
            self.weight_size,
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
            self.groups,
            self.weight_channels_last,
        )


class _IPEXConvTranspose2d(_IPEXConvTransposeNd):
    def __init__(self):
        super(_IPEXConvTranspose2d, self).__init__()


class _IPEXConvTranspose3d(_IPEXConvTransposeNd):
    def __init__(self):
        super(_IPEXConvTranspose3d, self).__init__()


def is_with_hook_on_weight_or_bias(module):
    # If hook is on `weight` or `bias`, will not prepack.
    if module._forward_pre_hooks is not None:
        for _, hook in module._forward_pre_hooks.items():
            if hasattr(hook, "name") and (hook.name == "weight" or hook.name == "bias"):
                return True
    if module._forward_hooks is not None:
        for _, hook in module._forward_hooks.items():
            if hasattr(hook, "name") and (hook.name == "weight" or hook.name == "bias"):
                return True
    if module._backward_hooks is not None:
        for _, hook in module._backward_hooks.items():
            if hasattr(hook, "name") and (hook.name == "weight" or hook.name == "bias"):
                return True


def weight_prepack_with_ipex(model, optimizer, params_attr, device_type="cpu"):
    from ._parameter_wrapper import (
        patch_state_dict,
        get_shared_parameter_status,
        IPEX_WEIGHT_PREPACK_MODULE_CPU,
    )

    is_training = optimizer is not None
    if len(params_attr) == 0:
        get_shared_parameter_status(model, params_attr)

    def found_wrapper(parameter, params_attr):
        for _, v in params_attr.items():
            if parameter is v.parameter:
                return v
        return None

    def convert(m, optimizer, params_attr):
        # already packed for reentrancy test
        if m.__class__ in IPEX_WEIGHT_PREPACK_MODULE_CPU().values():
            return m
        # pre check module class
        if m.__class__ not in IPEX_WEIGHT_PREPACK_MODULE_CPU().keys():
            return m
        if not hasattr(m, "weight"):
            return m
        if m.weight is None:
            return m
        if is_with_hook_on_weight_or_bias(m):
            return m
        if hasattr(m, "bias") and m.bias is not None:
            if m.bias in params_attr:
                param_wrapper = params_attr[m.bias]
            else:
                assert (
                    m.bias.dtype in [torch.bfloat16, torch.half]
                    and not m.master_weight_split
                )
                param_wrapper = found_wrapper(m.bias, params_attr)
                assert param_wrapper is not None
            bias_wrapper = param_wrapper
        else:
            bias_wrapper = None
        if m.weight in params_attr:
            param_wrapper = params_attr[m.weight]
        else:
            assert (
                m.weight.dtype in [torch.bfloat16, torch.half]
                and not m.master_weight_split
            )
            param_wrapper = found_wrapper(m.weight, params_attr)
            assert param_wrapper is not None
        if param_wrapper.can_prepack(m, is_training):
            new_m = IPEX_WEIGHT_PREPACK_MODULE_CPU()[m.__class__]()
            all_reduce_bias = m.bias
            if isinstance(new_m, _IPEXLinearAllreduce):
                m.bias = None
            param_wrapper.prepack(m, is_training)
            new_m.__dict__ = m.__dict__
            if isinstance(new_m, _IPEXLinearAllreduce):
                new_m.module_bias = all_reduce_bias
            new_m.ctx = param_wrapper.op_ctx
            setattr(new_m, "weight_wrapper", param_wrapper)  # noqa: B010
            setattr(new_m, "bias_wrapper", bias_wrapper)  # noqa: B010
            optimizer_para = param_wrapper.parameter
            if param_wrapper.master_parameter is not None:
                optimizer_para = param_wrapper.master_parameter
            optim._optimizer_utils.pack_optimizer_states(
                optimizer, optimizer_para, param_wrapper
            )
            return new_m
        else:
            return m

    def convert_rec(m, optimizer, params_attr):
        new_m = convert(m, optimizer, params_attr)
        for name, sub_m in m.named_children():
            setattr(new_m, name, convert_rec(sub_m, optimizer, params_attr)[0])
        return new_m, optimizer, params_attr

    if device_type == "cpu":
        opt_model, opt_optmizer, params_attr = convert_rec(
            model, optimizer, params_attr
        )

        patch_state_dict(opt_model, params_attr, "prepack")
        setattr(opt_model, "params_attr", params_attr)  # noqa: B010
        if opt_optmizer is not None:
            setattr(opt_optmizer, "params_attr", params_attr)  # noqa: B010
            optim._optimizer_utils.patch_load_state_dict(opt_optmizer)
            optim._optimizer_utils.patch_state_dict(opt_optmizer)
        return opt_model, opt_optmizer, params_attr


def record_input_shape_for_prepack(module, sample_input):
    def hook_function(self, input):
        # input for linear/conv/transpose conv received here will be Tuple[Tensor]
        if (
            self in [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]
            and self.padding_mode != "zeros"
        ):
            self.input_shape = F.pad(
                input[0], self._reversed_padding_repeated_twice, mode=self.padding_mode
            ).shape
        else:
            self.input_shape = input[0].shape

    def register_hook_function(module):
        if type(module) in [
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
        ]:
            module.register_forward_pre_hook(hook_function)

    def register_hook_function_rec(module):
        register_hook_function(module)
        for child in module.children():
            register_hook_function_rec(child)

    origin_state_dict = copy.deepcopy(module.state_dict())
    register_hook_function_rec(module)
    module(*sample_input)
    module.load_state_dict(origin_state_dict)
