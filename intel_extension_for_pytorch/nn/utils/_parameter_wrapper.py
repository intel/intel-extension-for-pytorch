import torch
from typing import Set
import functools
import contextlib
import types
from ...utils._logger import logger, WarningType
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _using_dnnl,
    _using_tpp,
)
from intel_extension_for_pytorch import frontend
import intel_extension_for_pytorch._C as core
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    _IPEXLinear,
    _IPEXConv1d,
    _IPEXConv2d,
    _IPEXConv3d,
    _IPEXConvTranspose2d,
    _IPEXConvTranspose3d,
    _IPEXLinearAllreduce,
    _IPEXLmHeadLinearAllreduce,
    may_import_deepspeed_modules,
)


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
        LinearAllreduce, LinearLayer = deepspeed_modules[:2]
        deepspeed_modules_mapping = {
            LinearLayer: _IPEXLinear,
            LinearAllreduce: _IPEXLinearAllreduce,
        }
        if len(deepspeed_modules) > 2:
            LmHeadLinearAllreduce = deepspeed_modules[2]
            deepspeed_modules_mapping.update(
                {LmHeadLinearAllreduce: _IPEXLmHeadLinearAllreduce}
            )
        torch_modules.update(deepspeed_modules_mapping)

    return torch_modules


@functools.lru_cache(None)
def IPEX_GEMM_MODULE_CPU():
    torch_modules = [torch.nn.Linear]

    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        torch_modules.extend(deepspeed_modules)

    return torch_modules


@functools.lru_cache(None)
def IPEX_WEIGHT_CONVERT_MODULE_CPU(inference: bool, dtype: torch.bfloat16):
    from ._lstm_convert import _LSTM
    from intel_extension_for_pytorch.nn.modules import (
        MergedEmbeddingBag,
        MergedEmbeddingBagWithCat,
    )

    module_convert_list_bf16_inference = [
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.LSTM,
        MergedEmbeddingBagWithCat,
        torch.nn.ParameterList,
    ]

    try:
        from transformers.models import albert

        module_convert_list_bf16_inference.append(albert.modeling_albert.AlbertMLMHead)
    except ImportError:
        pass

    module_convert_list_bf16_training = [
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.LSTM,
        # TODO: why different with inference
        MergedEmbeddingBag,
        torch.nn.EmbeddingBag,
        _LSTM,
        torch.nn.ParameterList,
    ]

    module_convert_list_fp16_inference = [
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
        torch.nn.Embedding,
        MergedEmbeddingBagWithCat,
        torch.nn.ParameterList,
    ]

    module_convert_list_fp16_training = [
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.EmbeddingBag,
        torch.nn.ParameterList,
    ]

    if dtype == torch.float16:
        if inference:
            return module_convert_list_fp16_inference
        else:
            return module_convert_list_fp16_training
    elif dtype == torch.bfloat16:
        if inference:
            return module_convert_list_bf16_inference
        else:
            return module_convert_list_bf16_training


def _should_prepack(module, is_training, is_xpu=False):
    if is_xpu:
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

    if module.weight.dtype not in (
        torch.float,
        torch.float32,
        torch.bfloat16,
        torch.half,
    ):
        return False
    return True


def get_shared_parameter_status(module, shared_p):
    visited_wrapper = []

    # TODO: weight and bias of deepspeed modules are no longer
    # nn.Parameter starting from deepspeed commit 94c7233.
    # Add a workaround here to convert them to Parameter.
    # Need to check if we can upstream this fix to deepspeed
    # and remove this workaround in IPEX later.
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        LinearAllreduce, LinearLayer = deepspeed_modules[:2]

        if isinstance(module, (LinearLayer, LinearAllreduce)):
            module.weight = torch.nn.Parameter(module.weight, requires_grad=False)
            if module.bias is not None:
                module.bias = torch.nn.Parameter(module.bias, requires_grad=False)

    for _, param in module._parameters.items():
        if param is None:
            continue
        if param not in shared_p:
            shared_p[param] = ParameterWrapper()
        shared_p[param].modules_cls.add(module.__class__)
        shared_p[param].num_modules += 1
        shared_p[param].parameter = param
        visited_wrapper.append(shared_p[param])
    # Special handle for nn.ParameterList since ParameterList is also a child module
    # Use the father's module class
    for _, sub_m in module.named_children():
        if isinstance(sub_m, torch.nn.ParameterList):
            for _, param in sub_m.named_parameters():
                if param is None:
                    continue
                if param not in shared_p:
                    shared_p[param] = ParameterWrapper()
                shared_p[param].parameter = param
                shared_p[param].modules_cls.add(module.__class__)
                visited_wrapper.append(shared_p[param])
    # If Linear's weight is shared by some module cannot be casted
    # Linear's bias should not be casted too
    union_set = set()
    for param_w in visited_wrapper:
        union_set = union_set | param_w.modules_cls
    for param_w in visited_wrapper:
        param_w.modules_cls = union_set
    del visited_wrapper
    for _, child in module._modules.items():
        get_shared_parameter_status(child, shared_p)


def remove_empty_tensor(out):
    empty_tensor_key = [
        key
        for key in out.keys()
        if key.endswith(
            ("_ipex_module_empty_weight_tensor", "_ipex_module_empty_bias_tensor")
        )
    ]

    for key in empty_tensor_key:
        del out[key]
    return out


def found_wrapper(parameter, params_attr):
    for _, v in params_attr.items():
        if parameter is v.parameter:
            return v
    return None


def patch_state_dict(model, params_attr, mode):
    def get_parammeter_from_model(model, name_list):
        if name_list[0] == "module" and not hasattr(model, "module"):
            # for DDP model, there is an extra module
            name_list = name_list[1:]
        model_or_param = model
        for attr in name_list:
            model_or_param = getattr(model_or_param, attr)
        return model_or_param

    def to_public_fp32(model, state_dict, params_attr):
        data_ptr_dict = {}
        for k, v in state_dict.items():
            v_ptr = v.data_ptr()
            if v_ptr in data_ptr_dict:
                # use cached tensor for multiple parameters share same tensor data
                state_dict[k] = data_ptr_dict[v_ptr]
                continue
            # k = "submodule_name.submodule_name.attr_name"
            # for example, "attn.linear.weight"
            name_list = k.split(".")
            param = get_parammeter_from_model(model, name_list)
            param_wrapper = found_wrapper(param, params_attr)
            if param_wrapper:
                if mode == "inference" and param_wrapper.original_dtype is not None:
                    state_dict[k] = v.to(param_wrapper.original_dtype)
                elif mode == "training":
                    state_dict[k] = param_wrapper._training_cast_to_fp32()
                else:
                    assert mode == "prepack"
                    state_dict[k] = param_wrapper._unpack_cast_to_fp32()
                data_ptr_dict[v_ptr] = state_dict[k]
        return state_dict

    def cast_back_state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        with torch.no_grad(), contextlib.ExitStack() as stack:
            state_dict = self._original_state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )
            # We don't save the _ipex_module_empty_weight_tensor or _ipex_module_empty_bias_tensor Parameter in the state dict
            state_dict = remove_empty_tensor(state_dict)
            state_dict = to_public_fp32(self, state_dict, params_attr)
        return state_dict

    if not hasattr(model, "_original_state_dict"):
        setattr(model, "_original_state_dict", model.state_dict)  # noqa: B010
    model.state_dict = types.MethodType(cast_back_state_dict, model)


class ParameterWrapper(object):
    def __init__(self):
        # Holding module class with the same Parameter
        # Used to inferring whether this Parameter can be cast or prepacked
        self.modules_cls: Set[type] = set()
        # We will only pack weight if there is only 1 module are using this Parameter
        self.num_modules: int = 0
        # Master parameter for low precision training (bf16/fp16)
        self.master_parameter: torch.nn.Parameter = None
        # Parameter in the module, for example, module.weight
        self.parameter: torch.nn.Parameter = None
        # Parameter trail for split optimization
        self.parameter_trail: torch.Tensor = None
        # The original dtype for Paramter
        self.original_dtype: torch.dtype = None
        # The caseted dtype by ipex.optimize
        self.casted_dtype: torch.dtype = None
        # Whether using split optimization
        self.split: bool = None
        # Whether weight is channels last for Conv/Conv_transpose
        self.weight_channels_last: bool = None
        # op context for prepacked weight
        self.op_ctx = None
        # shape before weight prepack, need this to check
        # whether we should pack state in optimizers
        self.plain_format_shape: torch.Size = None

    def can_cast_inference(self, dtype):
        if self.casted_dtype is not None:
            # already casted
            assert dtype == self.casted_dtype
            return True
        ori_dtype = self.parameter.dtype
        if ori_dtype not in (torch.float, torch.float32, torch.bfloat16, torch.float16):
            logger.warning(
                f"Can't convert model's parameters dtype from {ori_dtype} to {dtype}",
                _type=WarningType.NotSupported,
            )
            return False
        module_cls = IPEX_WEIGHT_CONVERT_MODULE_CPU(True, dtype)
        return all(cls in module_cls for cls in self.modules_cls)

    def cast_for_inference(self, dtype):
        if self.original_dtype is not None:
            # current parameter is casted
            return
        self.casted_dtype = dtype
        self.original_dtype = self.parameter.dtype
        casted_param = self.parameter.to(dtype)
        with torch.no_grad():
            self.parameter.data = casted_param

    def can_cast_training(self, dtype):
        if self.casted_dtype is not None:
            # already casted
            assert dtype == self.casted_dtype
            return True
        ori_dtype = self.parameter.dtype
        if ori_dtype not in (
            torch.float,
            torch.float32,
        ):
            logger.warning(
                f"Can't convert model's parameters dtype from {ori_dtype} to {dtype}",
                _type=WarningType.NotSupported,
            )
            return False
        module_cls = IPEX_WEIGHT_CONVERT_MODULE_CPU(False, dtype)
        return all(cls in module_cls for cls in self.modules_cls)

    def cast_for_training(self, dtype, split):
        if self.original_dtype is not None:
            # current parameter is casted
            return
        self.original_dtype = self.parameter.dtype
        self.casted_dtype = dtype
        self.split = split
        if split:
            assert (
                dtype == torch.bfloat16
            ), "master_weight_split is only support for bf16 now"
            top, self.parameter_trail = torch.ops.torch_ipex.split_float_bfloat16(
                self.parameter.data
            )
            with torch.no_grad():
                self.parameter.data = top
        else:
            # for non-split case, module use different parameter with optimizer
            self.master_parameter = self.parameter
            self.parameter = torch.nn.Parameter(
                self.master_parameter.data.to(dtype),
                requires_grad=self.master_parameter.requires_grad,
            )

    def _training_cast_to_fp32(self):
        if self.original_dtype is None:
            return self.parameter.data.detach()
        assert self.original_dtype in (
            torch.float,
            torch.float32,
        )
        if self.split:
            assert self.parameter_trail is not None
            fp32_param = torch.ops.torch_ipex.cat_bfloat16_float(
                self.parameter.data, self.parameter_trail
            )
            return fp32_param.detach()
        else:
            return self.master_parameter.data.detach()

    def _unpack_cast_to_fp32(self):
        fp32_param = self.parameter.data
        if self.split is not None:
            fp32_param = self._training_cast_to_fp32()
        elif self.original_dtype is not None:
            fp32_param = self.parameter.to(self.original_dtype)
        if self.op_ctx is None:
            return fp32_param
        else:
            return self.op_ctx.to_public(fp32_param)

    def can_prepack(self, module, is_training):
        if self.num_modules != 1:
            return False
        return _should_prepack(module, is_training)

    def prepack(self, module, is_training):
        self.plain_format_shape = module.weight.shape
        if module.__class__ not in IPEX_WEIGHT_PREPACK_MODULE_CPU():
            raise ValueError(
                "Cannot prepack module with class {}".format(module.__class__)
            )
        target_module = IPEX_WEIGHT_PREPACK_MODULE_CPU()[module.__class__]
        if target_module in (
            _IPEXConv1d,
            _IPEXConv2d,
            _IPEXConv3d,
        ):
            self.conv_prepack(module)
        elif target_module in (
            _IPEXConvTranspose2d,
            _IPEXConvTranspose3d,
        ):
            self.conv_transpose_prepack(module)
        else:
            assert target_module in (
                _IPEXLinear,
                _IPEXLinearAllreduce,
                _IPEXLmHeadLinearAllreduce,
            )
            self.linear_prepack(module, is_training)

    def pack_weight(self, use_dnnl=True):
        if not use_dnnl:
            # TODO: Haozhe, LinWei
            # weired case that cannot override ".data" for mkl here
            # The op_ctx seems not hold the original plain format weight
            self.parameter = self.op_ctx.get_weight()
        else:
            with torch.no_grad():
                self.parameter.data = self.op_ctx.get_weight()
        if self.master_parameter is not None:
            with torch.no_grad():
                self.master_parameter.data = self.op_ctx.pack(
                    self.master_parameter.data
                )
        if self.parameter_trail is not None:
            self.parameter_trail = self.op_ctx.pack(self.parameter_trail)

    def conv_prepack(self, module):
        module.prepack_input_shape = (
            module.input_shape if hasattr(module, "input_shape") else []
        )
        module.weight_channels_last = module.weight.is_contiguous(
            memory_format=torch.channels_last
        ) or module.weight.is_contiguous(memory_format=torch.channels_last_3d)
        self.weight_channels_last = module.weight_channels_last
        module.weight_size = module.weight.size()
        if module.padding_mode == "zeros":
            if module.padding == "same":
                padding: List[int] = []
                dilation = module.dilation
                kernel_size = module.kernel_size

                def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt):
                    if isinstance(func, torch.nn.Conv1d):
                        return conv1dOpt
                    if isinstance(func, torch.nn.Conv2d):
                        return conv2dOpt
                    else:
                        assert isinstance(func, torch.nn.Conv3d)
                        return conv3dOpt

                def get_dilation(i):
                    return dilation[i] if isinstance(dilation, tuple) else dilation

                def conv_padding_for_same(dilation, kernel_size):
                    total_pad = dilation * (kernel_size - 1)
                    left_pad = total_pad // 2
                    right_pad = total_pad - left_pad
                    return left_pad, right_pad

                for i in range(conv_picker(module, 0, 1, 2), -1, -1):
                    padding += conv_padding_for_same(get_dilation(i), kernel_size[i])
                module._real_padding = padding[: len(module.weight_size) - 2]
            elif module.padding == "valid":
                module._real_padding = tuple([0] * (len(module.weight_size) - 2))
            else:
                module._real_padding = module.padding
        else:
            module._real_padding = tuple([0] * (len(module.weight_size) - 2))
        self.op_ctx = torch.ops.ipex_prepack.convolution_prepack(
            module.weight,
            module.bias,
            module.stride,
            module._real_padding,
            module.dilation,
            module.groups,
            module.weight_channels_last,
            module.prepack_input_shape,
        )
        self.pack_weight()

    def conv_transpose_prepack(self, module):
        module.prepack_input_shape = (
            module.input_shape if hasattr(module, "input_shape") else []
        )
        module.weight_channels_last = module.weight.is_contiguous(
            memory_format=torch.channels_last
        ) or module.weight.is_contiguous(memory_format=torch.channels_last_3d)
        self.weight_channels_last = module.weight_channels_last
        module.weight_size = module.weight.size()
        module._real_padding = (
            module.padding
            if module.padding_mode == "zeros"
            else tuple([0] * (len(module.weight_size) - 2))
        )
        self.op_ctx = torch.ops.ipex_prepack.conv_transpose_prepack(
            module.weight,
            module.bias,
            module.stride,
            module.padding,
            module.output_padding,
            module.groups,
            module.dilation,
            module.weight_channels_last,
            module.prepack_input_shape,
        )
        self.pack_weight()

    def linear_prepack(self, module, is_training):
        if module.__class__ in IPEX_GEMM_MODULE_CPU():
            if (
                module.weight.dtype == torch.float32
                and not is_training
                and frontend.get_fp32_math_mode(device="cpu")
                == frontend.FP32MathMode.FP32
                and not _using_dnnl()
            ):
                use_dnnl = False
            else:
                assert module.weight.dtype in [
                    torch.float32,
                    torch.bfloat16,
                    torch.float16,
                ], "Only float, bf16 and fp16 are supported"
                use_dnnl = True

        module.use_tpp = _using_tpp() and (
            module.weight.dtype != torch.float16 or core.isa_has_amx_fp16_support()
        )
        if not hasattr(module, "out_features"):
            setattr(module, "out_features", module.weight.shape[0])  # noqa: B010

        module.tpp_fallback = False
        if module.use_tpp:
            from intel_extension_for_pytorch.nn.utils import (
                Apply_TPPLinear_weight_prepack,
            )

            Apply_TPPLinear_weight_prepack(module, dtype=module.weight.dtype)
            if module.tpp_fallback:
                module.use_tpp = False
            else:
                self.parameter.data = module.weight.data
                self.parameter = module.weight

        module.use_dnnl = use_dnnl if not module.use_tpp else False
        if not module.use_tpp:
            # prepare batch size
            module.batch_size_collapsed = None
            if hasattr(module, "input_shape"):
                module.batch_size_collapsed = 1
                for i in range(len(module.input_shape) - 1):
                    module.batch_size_collapsed *= module.input_shape[i]
            # create linear op context
            if module.use_dnnl:
                self.op_ctx = torch.ops.ipex_prepack.linear_prepack(
                    module.weight, module.bias, module.batch_size_collapsed
                )
            else:
                self.op_ctx = torch.ops.ipex_prepack.mkl_sgemm_prepack(
                    module.weight, module.bias, module.batch_size_collapsed
                )
            self.pack_weight(use_dnnl)

    def load_cast_and_prepack(self, module, param):
        # load from state dict
        if self.split is not None:
            if self.split:
                (
                    to_pack,
                    self.parameter_trail,
                ) = torch.ops.torch_ipex.split_float_bfloat16(param)
            else:
                to_pack = param.to(torch.bfloat16)
        elif self.casted_dtype is not None:
            to_pack = param.to(self.casted_dtype)
        else:
            to_pack = param
        if module.__class__ in IPEX_WEIGHT_PREPACK_MODULE_CPU():
            m_cls = IPEX_WEIGHT_PREPACK_MODULE_CPU()[module.__class__]
        else:
            m_cls = module.__class__
        if m_cls in (
            _IPEXConv1d,
            _IPEXConv2d,
            _IPEXConv3d,
        ):
            loaded_ctx = torch.ops.ipex_prepack.convolution_prepack(
                to_pack,
                module.bias,
                module.stride,
                module._real_padding,
                module.dilation,
                module.groups,
                module.weight_channels_last,
                module.prepack_input_shape,
            )
        elif m_cls in (
            _IPEXConvTranspose2d,
            _IPEXConvTranspose3d,
        ):
            loaded_ctx = torch.ops.ipex_prepack.conv_transpose_prepack(
                to_pack,
                module.bias,
                module.stride,
                module.padding,
                module.output_padding,
                module.groups,
                module.dilation,
                module.weight_channels_last,
                module.prepack_input_shape,
            )
        else:
            assert m_cls in (
                _IPEXLinear,
                _IPEXLinearAllreduce,
            )
            if module.use_dnnl:
                loaded_ctx = torch.ops.ipex_prepack.linear_prepack(
                    to_pack, module.bias, module.batch_size_collapsed
                )
            else:
                loaded_ctx = torch.ops.ipex_prepack.mkl_sgemm_prepack(
                    to_pack, module.bias, module.batch_size_collapsed
                )
        self.op_ctx.load_from_ctx(loaded_ctx)
        self.parameter.data = self.op_ctx.get_weight()
        if self.parameter_trail is not None:
            self.parameter_trail = self.op_ctx.pack(self.parameter_trail)
        if self.master_parameter is not None:
            self.master_parameter.data = self.op_ctx.pack(param)

    def load_cast(self, param):
        if self.split is not None:
            if self.split:
                (
                    self.parameter.data,
                    self.parameter_trail,
                ) = torch.ops.torch_ipex.split_float_bfloat16(param)
            else:
                self.parameter.data = param.to(self.casted_dtype)
                self.master_parameter.data = param
        elif self.casted_dtype is not None:
            self.parameter.data = param.to(self.casted_dtype)
        else:
            self.parameter.data = param

    def load(self, module, param):
        # load from state dict
        if self.op_ctx is not None:
            self.load_cast_and_prepack(module, param)
        else:
            self.load_cast(param)
