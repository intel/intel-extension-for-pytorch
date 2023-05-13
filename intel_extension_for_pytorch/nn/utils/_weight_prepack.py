import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import os

from intel_extension_for_pytorch import optim, frontend
from intel_extension_for_pytorch.cpu._auto_kernel_selection import _using_dnnl
import intel_extension_for_pytorch._C as core

logger = logging.getLogger(__name__)


def may_import_deepspeed_modules():
    try:
        # import deepspeed in a global space will raise circular import error
        # intel-extension-for-deepspeed imports both IPEX and deepspeed
        import deepspeed
        from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

        return LinearAllreduce, LinearLayer
    except ImportError:
        return None


if may_import_deepspeed_modules() is not None:
    # register ds comm as the kernel of aten.all_reduce
    # to align the _IPEX_LinearAllreduce with LinearAllreduce
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    from deepspeed import comm

    def _all_reduce(self, reduceOp, tag, ranks, group_size):
        prefer_deepspeed_comm = os.environ.get("PREFER_DEEPSPEED_COMM")
        if prefer_deepspeed_comm:
            comm.all_reduce(self, async_op=False)
        else:
            reduceOp = reduceOp.upper()
            op = dist.ReduceOp.RedOpType.__members__.get(reduceOp)
            if op is None:
                raise ValueError(f"Invalid reduce operation {reduceOp}")
            group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
            assert group is not None
            comm.all_reduce(self, group=group, op=op, async_op=False)
        return self

    ds_comm = torch.library.Library("deepspeed_comm", "DEF")
    ds_comm.define(
        "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor"
    )
    ds_comm_lib_cpu = torch.library.Library("deepspeed_comm", "IMPL", "CPU")
    ds_comm_lib_cpu.impl("all_reduce", _all_reduce)


def _save_weight_bias_to_state_dict(self, destination, prefix):
    if self.bias is not None:
        if hasattr(self, "master_bias"):
            bias = self.master_bias
        elif hasattr(self, "bias_trail"):
            bias = torch.ops.torch_ipex.cat_bfloat16_float(self.bias, self.bias_trail)
        else:
            bias = self.bias.float()
        destination[prefix + "bias"] = bias.detach()
    if hasattr(self, "master_weight"):
        weight = self.master_weight
    elif hasattr(self, "weight_trail"):
        weight = torch.ops.torch_ipex.cat_bfloat16_float(self.weight, self.weight_trail)
    else:
        weight = self.weight.float()
    destination[prefix + "weight"] = self.ctx.to_public(weight.detach())


def _load_from_state_dict_pre_hook(self, state_dict, prefix):
    w_name = prefix + "weight"
    b_name = prefix + "bias"
    fp32_loaded_weight = state_dict[w_name]
    weight_trail = None
    if hasattr(self, "master_weight"):
        loaded_weight = fp32_loaded_weight.bfloat16()
    elif hasattr(self, "weight_trail"):
        loaded_weight, weight_trail = torch.ops.torch_ipex.split_float_bfloat16(
            fp32_loaded_weight
        )
    else:
        loaded_weight = fp32_loaded_weight.to(self.weight.dtype)
    if b_name in state_dict:
        loaded_bias = state_dict[b_name]
        if hasattr(self, "master_bias"):
            self.master_bias.copy_(loaded_bias)
            loaded_bias = loaded_bias.bfloat16()
        elif hasattr(self, "bias_trail"):
            loaded_bias, bias_trail = torch.ops.torch_ipex.split_float_bfloat16(
                loaded_bias
            )
            self.bias_trail.copy_(bias_trail)
        else:
            loaded_bias = loaded_bias.to(self.bias.dtype)
    else:
        loaded_bias = None
    return loaded_weight, loaded_bias, fp32_loaded_weight, weight_trail


def _load_from_state_dict_post_hook(self, loaded_ctx, fp32_loaded_weight, weight_trail):
    _load_from_state_dict_pre_hook
    self.ctx.load_from_ctx(loaded_ctx)
    if hasattr(self, "master_weight"):
        self.master_weight.copy_(self.ctx.pack(fp32_loaded_weight))
    elif hasattr(self, "weight_trail"):
        self.weight_trail.copy_(self.ctx.pack(weight_trail))


class _IPEXConvNd(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "out_channels",
        "kernel_size",
    ]

    def __init__(self, dense_module):
        super(_IPEXConvNd, self).__init__()
        self.out_channels = dense_module.out_channels
        self.in_channels = dense_module.in_channels
        self.kernel_size = dense_module.kernel_size
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        self.padding_mode = dense_module.padding_mode
        self._reversed_padding_repeated_twice = (
            dense_module._reversed_padding_repeated_twice
        )
        self.prepack_input_shape = (
            dense_module.input_shape if hasattr(dense_module, "input_shape") else []
        )
        self.weight_channels_last = dense_module.weight.is_contiguous(
            memory_format=torch.channels_last
        ) or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)
        self.weight_size = dense_module.weight.size()
        self._real_padding = (
            self.padding
            if self.padding_mode == "zeros"
            else tuple([0] * (len(self.weight_size) - 2))
        )

        # TODO: ".clone()" will make weight shared by multiple module not shared anymore
        # related issues: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/65
        if dense_module.bias is not None:
            self.bias = nn.Parameter(
                dense_module.bias.detach().clone(),
                requires_grad=dense_module.bias.requires_grad,
            )
            if hasattr(dense_module, "master_bias"):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, "bias_trail"):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter("bias", None)
        # create conv op context
        self.ctx = torch.ops.ipex_prepack.convolution_prepack(
            dense_module.weight,
            self.bias,
            self.stride,
            self._real_padding,
            self.dilation,
            self.groups,
            self.weight_channels_last,
            self.prepack_input_shape,
        )

        self.weight = nn.Parameter(
            self.ctx.get_weight(), requires_grad=dense_module.weight.requires_grad
        )

        # pack master_weight or weight_trail if needed
        if hasattr(dense_module, "master_weight"):
            self.master_weight = self.ctx.pack(
                dense_module.master_weight.detach().clone()
            )
        elif hasattr(dense_module, "weight_trail"):
            self.weight_trail = self.ctx.pack(
                dense_module.weight_trail.detach().clone(),
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert (
            not keep_vars
        ), "can not using keep_vars true when to save _IPEXConvNd's parameters"
        _save_weight_bias_to_state_dict(self, destination, prefix)

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
            (
                loaded_weight,
                loaded_bias,
                fp32_loaded_weight,
                weight_trail,
            ) = _load_from_state_dict_pre_hook(self, state_dict, prefix)
            loaded_ctx = torch.ops.ipex_prepack.convolution_prepack(
                loaded_weight,
                loaded_bias,
                self.stride,
                self._real_padding,
                self.dilation,
                self.groups,
                self.weight_channels_last,
                self.prepack_input_shape,
            )
            _load_from_state_dict_post_hook(
                self, loaded_ctx, fp32_loaded_weight, weight_trail
            )

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
    def __init__(self, dense_module):
        super(_IPEXConv1d, self).__init__(dense_module)


class _IPEXConv2d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv2d, self).__init__(dense_module)


class _IPEXConv3d(_IPEXConvNd):
    def __init__(self, dense_module):
        super(_IPEXConv3d, self).__init__(dense_module)


class _IPEXLinear(torch.nn.Module):
    def __init__(self, dense_module, use_dnnl):
        super(_IPEXLinear, self).__init__()

        self.use_dnnl = use_dnnl
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
                requires_grad=dense_module.bias.requires_grad,
            )
            if hasattr(dense_module, "master_bias"):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, "bias_trail"):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter("bias", None)

        self.out_features = (
            dense_module.out_features
            if hasattr(dense_module, "out_features")
            else dense_module.weight.size()[0]
        )

        # create linear op context
        if self.use_dnnl:
            self.ctx = torch.ops.ipex_prepack.linear_prepack(
                dense_module.weight, self.bias, self.batch_size_collapsed
            )
        else:
            self.ctx = torch.ops.ipex_prepack.mkl_sgemm_prepack(
                dense_module.weight, self.bias, self.batch_size_collapsed
            )

        self.weight = nn.Parameter(
            self.ctx.get_weight(), requires_grad=dense_module.weight.requires_grad
        )

        # pack master_weight or weight_trail if needed
        if hasattr(dense_module, "master_weight"):
            self.master_weight = self.ctx.pack(
                dense_module.master_weight.detach().clone()
            )
        elif hasattr(dense_module, "weight_trail"):
            self.weight_trail = self.ctx.pack(
                dense_module.weight_trail.detach().clone(),
            )

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
        _save_weight_bias_to_state_dict(self, destination, prefix)

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
            (
                loaded_weight,
                loaded_bias,
                fp32_loaded_weight,
                weight_trail,
            ) = _load_from_state_dict_pre_hook(self, state_dict, prefix)
            pack_fn = (
                torch.ops.ipex_prepack.linear_prepack
                if self.use_dnnl
                else torch.ops.ipex_prepack.mkl_sgemm_prepack
            )
            loaded_ctx = pack_fn(loaded_weight, loaded_bias, self.batch_size_collapsed)
            _load_from_state_dict_post_hook(
                self, loaded_ctx, fp32_loaded_weight, weight_trail
            )


class _IPEXLinearAllreduce(_IPEXLinear):
    def __init__(self, dense_module, use_dnnl):
        # _IPEXLinear __init__ func will save the bias value and then use it during forward.
        # deepspeed LinearAllreduce will firstly calculate torch.matmul(x, w), then call the all_reduce and finally add the bias to the result.
        # reference: https://github.com/microsoft/DeepSpeed/blob/f1d2a15b50fa83beb8fb8076fae883853f83b5ad/deepspeed/module_inject/layers.py#L19-L25
        # Thus we need to save the original bias here and use None as the bias during the __init__ func
        module_bias = dense_module.bias
        dense_module.bias = None

        super(_IPEXLinearAllreduce, self).__init__(dense_module, use_dnnl)

        self.module_bias = module_bias
        self.mp_group = dense_module.mp_group

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
        self.prepack_input_shape = (
            dense_module.input_shape if hasattr(dense_module, "input_shape") else []
        )
        self.weight_channels_last = dense_module.weight.is_contiguous(
            memory_format=torch.channels_last
        ) or dense_module.weight.is_contiguous(memory_format=torch.channels_last_3d)
        self.weight_size = dense_module.weight.size()

        if dense_module.bias is not None:
            self.bias = nn.Parameter(
                dense_module.bias.detach().clone(),
                requires_grad=dense_module.bias.requires_grad,
            )
            if hasattr(dense_module, "master_bias"):
                self.master_bias = dense_module.master_bias
            elif hasattr(dense_module, "bias_trail"):
                self.bias_trail = dense_module.bias_trail
        else:
            self.register_parameter("bias", None)
        # create conv op context
        self.ctx = torch.ops.ipex_prepack.conv_transpose_prepack(
            dense_module.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
            self.weight_channels_last,
            self.prepack_input_shape,
        )

        self.weight = nn.Parameter(
            self.ctx.get_weight(), requires_grad=dense_module.weight.requires_grad
        )

        # pack master_weight or weight_trail if needed
        if hasattr(dense_module, "master_weight"):
            self.master_weight = self.ctx.pack(
                dense_module.master_weight.detach().clone()
            )
        elif hasattr(dense_module, "weight_trail"):
            self.weight_trail = self.ctx.pack(
                dense_module.weight_trail.detach().clone()
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert (
            not keep_vars
        ), "can not using keep_vars true when to save _IPEXConvTransposeNd's parameters"
        _save_weight_bias_to_state_dict(self, destination, prefix)

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
            (
                loaded_weight,
                loaded_bias,
                fp32_loaded_weight,
                weight_trail,
            ) = _load_from_state_dict_pre_hook(self, state_dict, prefix)
            loaded_ctx = torch.ops.ipex_prepack.conv_transpose_prepack(
                loaded_weight,
                loaded_bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
                self.weight_channels_last,
                self.prepack_input_shape,
            )
            _load_from_state_dict_post_hook(
                self, loaded_ctx, fp32_loaded_weight, weight_trail
            )

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
    def __init__(self, dense_module):
        super(_IPEXConvTranspose2d, self).__init__(dense_module)


class _IPEXConvTranspose3d(_IPEXConvTransposeNd):
    def __init__(self, dense_module):
        super(_IPEXConvTranspose3d, self).__init__(dense_module)


@functools.lru_cache(None)
def IPEX_WEIGHT_PREPACK_MODULE_CPU():
    torch_modules = {
        torch.nn.Linear: _IPEXLinear,
        torch.nn.Conv2d: _IPEXConv2d,
        torch.nn.Conv3d: _IPEXConv3d,
        torch.nn.Conv1d: _IPEXConv1d,
        torch.nn.ConvTranspose2d: _IPEXConvTranspose2d,
        torch.nn.ConvTranspose3d: _IPEXConvTranspose3d,
    }

    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        LinearAllreduce, LinearLayer = deepspeed_modules
        deepspeed_modules = {
            LinearLayer: _IPEXLinear,
            LinearAllreduce: _IPEXLinearAllreduce,
        }
        torch_modules.update(deepspeed_modules)

    return torch_modules


@functools.lru_cache(None)
def IPEX_GEMM_MODULE_CPU():
    torch_modules = [torch.nn.Linear]

    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        torch_modules.extend(deepspeed_modules)

    return torch_modules


def _should_prepack(module, is_training, is_xpu=False):
    if type(module) not in IPEX_WEIGHT_PREPACK_MODULE_CPU() and not is_xpu:
        return False
    # If hook is on `weight` or `bias`, will not prepack.
    if module._forward_pre_hooks is not None:
        for _, hook in module._forward_pre_hooks.items():
            if hasattr(hook, "name") and (hook.name == "weight" or hook.name == "bias"):
                return False
    if module._forward_hooks is not None:
        for _, hook in module._forward_hooks.items():
            if hasattr(hook, "name") and (hook.name == "weight" or hook.name == "bias"):
                return False
    if module._backward_hooks is not None:
        for _, hook in module._backward_hooks.items():
            if hasattr(hook, "name") and (hook.name == "weight" or hook.name == "bias"):
                return False

    # for training, if auto_kernel_selection(onednn) is off, IPEX won't prepack FP32 linear.
    if (
        isinstance(module, torch.nn.Linear)
        and not _using_dnnl()
        and is_training
        and module.weight.dtype is torch.float
    ):
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
    # Conv1d backward is not implemented, will not prepack.
    if isinstance(module, torch.nn.Conv1d) and module.training:
        return False
    return True


def weight_prepack_with_ipex(
    module, optimizer, params_attr, inplace=False, device_type="cpu"
):
    def convert(m, optimizer, params_attr):
        if _should_prepack(m, is_training=(optimizer != None)) and (
            m.weight.dtype == torch.float32
            or m.weight.dtype == torch.bfloat16
            or (
                m.weight.dtype == torch.half
                and type(m) not in [torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d]
            )
        ):
            weight = m.master_weight if hasattr(m, "master_weight") else m.weight
            if weight not in params_attr:
                params_attr[weight] = {}
            if type(m) in IPEX_GEMM_MODULE_CPU():
                if m.weight.dtype == torch.half:
                    new_m = IPEX_WEIGHT_PREPACK_MODULE_CPU()[type(m)](m, use_dnnl=True)
                elif (
                    m.weight.dtype == torch.float32
                    and optimizer is None
                    and frontend.get_fp32_math_mode(device="cpu")
                    == frontend.FP32MathMode.FP32
                    and not _using_dnnl()
                ):
                    new_m = IPEX_WEIGHT_PREPACK_MODULE_CPU()[type(m)](m, use_dnnl=False)
                else:
                    assert m.weight.dtype in [
                        torch.float32,
                        torch.bfloat16,
                    ], "Only float, bf16 and fp16 are supported"
                    new_m = IPEX_WEIGHT_PREPACK_MODULE_CPU()[type(m)](m, use_dnnl=True)
            else:
                new_m = IPEX_WEIGHT_PREPACK_MODULE_CPU()[type(m)](m)

            # move original layer info to new prepacked layer
            if hasattr(m, "master_weight_split"):
                setattr(new_m, "master_weight_split", m.master_weight_split)

            params_attr[weight].update({"op": type(m), "ctx": new_m.ctx})
            if hasattr(new_m, "weight_channels_last"):
                params_attr[weight]["weight_channels_last"] = (
                    new_m.weight_channels_last,
                )
            if "bf16_param" in params_attr[weight]:
                params_attr[weight]["bf16_param"] = new_m.weight
            elif "trail" in params_attr[weight]:
                params_attr[weight]["trail"] = new_m.weight_trail
            if "fp16_param" in params_attr[weight]:
                params_attr[weight]["fp16_param"] = new_m.weight
            # update entry from origin weight to packed weight, from origin bias to cloned bias
            new_weight = (
                new_m.master_weight if hasattr(m, "master_weight") else new_m.weight
            )
            params_attr[new_weight] = params_attr.pop(weight)
            params_pair = {weight: new_weight}
            if hasattr(m, "bias") and m.bias is not None:
                bias = m.master_bias if hasattr(m, "master_bias") else m.bias
                new_bias = (
                    new_m.master_bias if hasattr(m, "master_bias") else new_m.bias
                )
                params_pair.update({bias: new_bias})
                if bias in params_attr:
                    if "bf16_param" in params_attr[bias]:
                        params_attr[bias]["bf16_param"] = new_m.bias
                    elif "trail" in params_attr[bias]:
                        params_attr[bias]["trail"] = new_m.bias_trail
                    if "fp16_param" in params_attr[bias]:
                        params_attr[bias]["fp16_param"] = new_m.bias
                    if bias in params_attr:
                        params_attr[new_bias] = params_attr.pop(bias)
            # replace optimizer's param with prepacked param, also prepack its state.
            optim._optimizer_utils.pack_optimizer_params_and_states(
                optimizer, params_pair, params_attr, m.weight.dtype
            )
            if inplace:
                del m.weight
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
            module, optimizer, params_attr
        )
        if opt_optmizer is not None:
            setattr(opt_optmizer, "params_attr", params_attr)
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
