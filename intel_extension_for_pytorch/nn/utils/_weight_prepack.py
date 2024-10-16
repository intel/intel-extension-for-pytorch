import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import pkg_resources
from intel_extension_for_pytorch import optim
from intel_extension_for_pytorch.cpu.tpp.utils.blocked_layout import (
    BlockedParameter,
    get_vnni_blocking,
)

from intel_extension_for_pytorch.cpu._auto_kernel_selection import _using_tpp

logger = logging.getLogger(__name__)

USE_LOW_PREC_PARAMS = True


def TPPLinear_weight_prepack(m, bk=None, bc=None, layer_dtype=torch.float32):
    m.__class__ = _IPEXLinear
    m.weight = BlockedParameter(m.weight.data)
    m.weight.set_blocking_param(
        (
            [bk, bc],
            [0, 2, 3, 1],
        )
    )
    m.weight_for_large_batch = None
    layer_use_low_prec = layer_dtype != torch.float32
    if layer_use_low_prec is True and USE_LOW_PREC_PARAMS:
        low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
        m.weight.set_blocking_param(
            (
                [
                    bk,
                    [
                        bc // low_prec_vnni_blocking,
                        low_prec_vnni_blocking,
                    ],
                ],
                [0, 2, 3, 1, 4],
                layer_dtype,
            )
        )

    if m.bias is not None:
        m.bias = BlockedParameter(m.bias.data)
        m.bias.set_blocking_param((None, None, layer_dtype))
    return m


# For below shapes where ic is 2x bigger than oc, oneDNN may perform better when mb is also big
# (while TPP still better when mb is small). However, mb is not able to be aware during IPEX
# weight prepack stage.
#
# For short term, we add the ENV flag ("BF16_OPTIMIZED_THROUGHPUT") to specify when we are running
#  with big mb like throughput usecase, and avoiding regression when mb is small like latency usecase.
#
# For long term, mark as TODO, we will tune TPP block layout/loop order to make it on par with oneDNN.

fallback_ic_shape_list = [13824, 11008]
fallback_oc_shape_list = [4096, 5120]


def Apply_TPPLinear_weight_prepack(m, dtype, device="cpu"):
    BF16_OPTIMIZED_THROUGHPUT = int(os.environ.get("BF16_OPTIMIZED_THROUGHPUT", 0))
    if (m.weight.size()[0] == 50400 or m.weight.size()[0] == 32000) and m.weight.size()[
        1
    ] % 64 == 0:
        m = TPPLinear_weight_prepack(m, 100, 64, dtype)
    elif (
        m.weight.size()[0] % 16 == 0
        and m.weight.size()[1] % 64 == 0
        and (
            not (
                BF16_OPTIMIZED_THROUGHPUT
                and (
                    m.weight.size()[1] in fallback_ic_shape_list
                    and m.weight.size()[0] in fallback_oc_shape_list
                )
            )
        )
    ):
        m = TPPLinear_weight_prepack(m, 16, 64, dtype)
    else:
        m.tpp_fallback = True
        return
    m.tpp_fallback = False

    block(m)


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()


def may_import_deepspeed_modules():
    try:
        # import deepspeed in a global space will raise circular import error
        # intel-extension-for-deepspeed imports both IPEX and deepspeed
        from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

        ds_layers = [LinearAllreduce, LinearLayer]

        # TODO: remove this logic once deepspeed LmHeadLinearAllreduce change has been upstream-ed.
        try:
            from deepspeed.module_inject.layers import LmHeadLinearAllreduce

            ds_layers.append(LmHeadLinearAllreduce)
            return ds_layers
        except ImportError:
            return ds_layers
    except ImportError:
        return None


installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if "deepspeed" in installed_pkg:
    from deepspeed import comm

    def _all_reduce(self):
        comm.inference_all_reduce(self, async_op=False)
        return self

    ds_comm = torch.library.Library("deepspeed_comm", "DEF")
    ds_comm.define("all_reduce(Tensor self) -> Tensor")
    ds_comm_lib_cpu = torch.library.Library("deepspeed_comm", "IMPL", "CPU")
    ds_comm_lib_cpu.impl("all_reduce", _all_reduce)


def _all_reduce_and_bias_add(mp_group, original_bias, output):
    if mp_group is not None:
        torch.ops.deepspeed_comm.all_reduce(output)
    if original_bias is not None:
        output += original_bias

    return output


def _pre_ipex_gemm(input, world_size, rank):
    assert "deepspeed" in installed_pkg, "_pre_ipex_gemm requires deepspeed installed"
    try:
        from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
    except ImportError:
        from deepspeed.utils.tp_shard import get_shard_size, get_shard_size_list
    input_shard_size = get_shard_size(input.shape[-1], world_size, "lm_head")
    input_shard_offset = sum(
        get_shard_size_list(input.shape[-1], world_size, "lm_head")[0:rank]
    )
    return input[:, :, input_shard_offset : input_shard_offset + input_shard_size]


def _ipex_module_load_from_state_dict_(self, state_dict, prefix):
    w_name = prefix + "weight"
    b_name = prefix + "bias"
    loaded_weight = state_dict[w_name]
    if b_name in state_dict:
        loaded_bias = state_dict[b_name]
        self.bias_wrapper.load(self, loaded_bias)
    self.weight_wrapper.load(self, loaded_weight)


class _IPEXPrepackModule(nn.Module):
    def _get_forward_weight(self):
        return self.weight if self.training else self._ipex_module_empty_weight_tensor

    def _get_forward_bias(self):
        return self.bias if self.training else self._ipex_module_empty_bias_tensor


class _IPEXConvNd(_IPEXPrepackModule):
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
        # [ Note -- Fix the size of the saved TorchScript model ]
        # In inference case:
        # We pass empty tensors for weight and bias in forward to solve the size increase issue of the saved TorchScript model.
        # For runtime memory usage, we don't have concern to use real tensors since
        # self.weight and self.bias shares the storage with the tensors in self.ctx,
        # thus the runtime memory won't grow.
        # However, for the saved TorchScript model, after torch.jit.trace and torch.jit.freeze,
        # self.ctx (with weight and bias inside), self.weight and self.bias will all be converted to prim::Constant on the graph
        # and they will all be serialized which makes the saved model size grow.
        # For inference, we pass in empty tensors in the forward function for weight and bias,
        # since self.weight and self.bias are not used,
        # they won't be on the traced graph, thus won't be saved later.
        # In training case:
        # Since autograd requires that grad shape to match the input tensor shape in the forward func,
        # we can't use empty tensor here.
        if self.padding_mode != "zeros":
            return torch.ops.torch_ipex.convolution_forward(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self._get_forward_weight(),
                self._get_forward_bias(),
                self.ctx.get_data_handle(),
                self.weight_size,
                self._real_padding,
                self.stride,
                self.dilation,
                self.weight_channels_last,
            )
        return torch.ops.torch_ipex.convolution_forward(
            x,
            self._get_forward_weight(),
            self._get_forward_bias(),
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


@torch.library.impl("torch_ipex::choose_tpp_linear_weight", "CPU")
def choose_tpp_linear_weight(x, weight, weight_for_large_batch):
    M = x.numel() // x.size(-1)
    return (
        weight_for_large_batch
        if weight_for_large_batch is not None and M >= 256
        else weight
    )


torch.library.define(
    "torch_ipex::choose_tpp_linear_weight",
    "(Tensor x, Tensor weight, Tensor? weight_for_large_batch) -> Tensor",
)


class _IPEXLinear(_IPEXPrepackModule):
    def __init__(self):
        super(_IPEXLinear, self).__init__()
        self.weight_for_large_batch = None  # for LLM large batch/first token inference

    def maybe_block_params(self):
        self.weight.block()
        if self.bias is not None:
            self.bias.block()

    def pre_ipex_gemm(self, input):
        return input

    def post_ipex_gemm(self, output):
        return output

    def forward(self, x):
        x = self.pre_ipex_gemm(x)

        if self.use_dnnl:
            output = torch.ops.torch_ipex.ipex_linear(
                x,
                self._get_forward_weight(),
                self._get_forward_bias(),
                self.ctx.get_data_handle(),
                self.out_features,
            )
        elif self.use_tpp:
            if self.tpp_fallback:
                output = torch.nn.functional.linear(x, self.weight, self.bias)
            else:
                x = x.to(self.weight.dtype).contiguous()
                weight_for_large_batch = (
                    self.weight_for_large_batch
                    if hasattr(self, "weight_for_large_batch")
                    else None
                )
                w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                    x, self.weight, weight_for_large_batch
                )
                if self.bias is not None:
                    output = torch.ops.torch_ipex.tpp_linear_bias(
                        x, w.detach(), self.bias.detach(), self.out_features
                    )
                else:
                    output = torch.ops.torch_ipex.tpp_linear(
                        x, w.detach(), self.out_features
                    )
        else:
            output = torch.ops.torch_ipex.ipex_MKLSGEMM(
                x,
                self._get_forward_weight(),
                self._get_forward_bias(),
                self.ctx.get_data_handle(),
                self.out_features,
            )

        return self.post_ipex_gemm(output)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.use_tpp:
            blocked_params = []
            for p in self.parameters(recurse=False):
                if isinstance(p, BlockedParameter) and p.is_blocked():
                    p.unblock()
                    blocked_params.append(p)
            super(_IPEXLinear, self)._save_to_state_dict(destination, prefix, keep_vars)
            for p in blocked_params:
                p.block()
        else:
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
        if self.use_tpp:
            blocked_params = []
            for p in self.parameters(recurse=False):
                if isinstance(p, BlockedParameter) and p.is_blocked():
                    p.unblock()
                    blocked_params.append(p)
            super(_IPEXLinear, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for p in blocked_params:
                p.block()
        else:
            with torch.no_grad():
                _ipex_module_load_from_state_dict_(self, state_dict, prefix)


class _IPEXLinearAllreduce(_IPEXLinear):
    def __init__(self):
        super(_IPEXLinearAllreduce, self).__init__()

    def post_ipex_gemm(self, output):
        return _all_reduce_and_bias_add(self.mp_group, self.original_bias, output)


class _IPEXLmHeadLinearAllreduce(_IPEXLinear):
    def __init__(self):
        super(_IPEXLmHeadLinearAllreduce, self).__init__()

    def pre_ipex_gemm(self, input):
        return _pre_ipex_gemm(input, self.world_size, self.rank)

    def post_ipex_gemm(self, output):
        return _all_reduce_and_bias_add(self.mp_group, self.original_bias, output)


class _IPEXConvTransposeNd(_IPEXPrepackModule):
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
            self._get_forward_weight(),
            self._get_forward_bias(),
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
        found_wrapper,
    )

    is_training = optimizer is not None
    if len(params_attr) == 0:
        get_shared_parameter_status(model, params_attr)

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
            if isinstance(new_m, (_IPEXLinearAllreduce, _IPEXLmHeadLinearAllreduce)):
                m.bias = None
            if _using_tpp() and hasattr(m, "tpp_fallback"):
                weight_key = m.weight
                param_wrapper.prepack(m, is_training)
                if m.tpp_fallback:
                    new_m.tpp_fallback = True
                else:
                    new_m.tpp_fallback = False
                    params_attr[m.weight] = params_attr.pop(weight_key)
                    del weight_key

            else:
                param_wrapper.prepack(m, is_training)

            new_m.__dict__ = m.__dict__

            if isinstance(new_m, (_IPEXLinearAllreduce, _IPEXLmHeadLinearAllreduce)):
                new_m.original_bias = all_reduce_bias
            new_m.ctx = param_wrapper.op_ctx
            setattr(new_m, "weight_wrapper", param_wrapper)  # noqa: B010
            setattr(new_m, "bias_wrapper", bias_wrapper)  # noqa: B010
            optimizer_para = param_wrapper.parameter
            if param_wrapper.master_parameter is not None:
                optimizer_para = param_wrapper.master_parameter
            optim._optimizer_utils.pack_optimizer_states(
                optimizer, optimizer_para, param_wrapper
            )
            new_m.training = is_training
            # _ipex_module_empty_weight_tensor and _ipex_module_empty_bias_tensor
            # have to be a Parameter so that dynamo could convert it into FakeTensor
            # These empty tensors will only be used during inference but we'll set
            # it in both training and eval mode to supprt the use case of the below
            # workflow:
            # model.train() -> ipex.optimize(model) -> model.eval()
            new_m._ipex_module_empty_weight_tensor = torch.nn.Parameter(
                torch.Tensor().to(dtype=new_m.weight.dtype), requires_grad=False
            )
            if new_m.bias is None:
                new_m.register_parameter("_ipex_module_empty_bias_tensor", None)
            else:
                new_m._ipex_module_empty_bias_tensor = torch.nn.Parameter(
                    torch.Tensor().to(dtype=new_m.bias.dtype), requires_grad=False
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
            hook = module.register_forward_pre_hook(hook_function)
            hooks.append(hook)

    def register_hook_function_rec(module):
        register_hook_function(module)
        for child in module.children():
            register_hook_function_rec(child)

    hooks = []
    module_is_train = module.training
    module.eval()
    register_hook_function_rec(module)
    module(*sample_input)
    if module_is_train:
        module.train()
    for hook in hooks:
        hook.remove()
