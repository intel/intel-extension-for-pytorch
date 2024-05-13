import dataclasses
from typing import Callable, Tuple, Any, List, Optional, Dict
import torch
import torch.nn.functional as F
import torch.nn.quantized.dynamic as nnqd
from intel_extension_for_pytorch.nn.functional import interaction
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithCat


functions_supported_by_quantization = set(
    [
        torch.Tensor.add,
        torch.add,
        torch.Tensor.relu,
        # torch.Tensor.sigmoid,  # TODO
        torch.flatten,
        torch.Tensor.flatten,
        F.adaptive_avg_pool2d,
        F.adaptive_avg_pool3d,
        F.avg_pool2d,
        F.avg_pool3d,
        F.max_pool2d,
        F.max_pool3d,
        F.conv2d,
        F.conv3d,
        torch.conv2d,
        torch.conv3d,
        F.conv_transpose2d,
        F.conv_transpose3d,
        torch.conv_transpose2d,
        torch.conv_transpose3d,
        torch.relu,
        F.relu,
        # torch.sigmoid,  # TODO
        # F.sigmoid,  # TODO
        # F.gelu, # TODO
        F.linear,
        torch._C._nn.linear,
        torch.matmul,
        torch.bmm,
        torch.Tensor.matmul,
        torch.Tensor.bmm,
        F.embedding_bag,
        torch.embedding_bag,
    ]
)

# ipex customer function
functions_supported_by_quantization_ipex = set(
    [
        interaction,
        torch.ops.torch_ipex.interaction_forward,
        torch.ops.torch_ipex.merged_embeddingbag_cat_forward,
    ]
)

module_types_supported_by_quantization = set(
    [
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.ReLU,
        # torch.nn.Sigmoid,  # TODO
        # torch.nn.GELU,     # TODO
        torch.nn.EmbeddingBag,
        MergedEmbeddingBagWithCat,
        torch.nn.Flatten,
        torch.nn.LSTM,
        # dynamic quantization module
        nnqd.Linear,
        nnqd.LSTM,
    ]
)

may_inplace_module = set(
    [
        torch.nn.ReLU,
    ]
)


a_related_to_b = (
    (str(torch.add), str(torch.Tensor.add)),
    (str(torch.Tensor.add), str(torch.add)),
    (str(torch.nn.Linear), str(nnqd.Linear)),
    (str(nnqd.Linear), str(torch.nn.Linear)),
    (str(torch.nn.LSTM), str(nnqd.LSTM)),
    (str(nnqd.LSTM), str(torch.nn.LSTM)),
)

conv_linear_ops = [
    # F.conv1d, # it will be enabled at next step.
    str(F.conv2d),
    str(F.conv3d),
    str(torch.conv2d),
    str(torch.conv3d),
    str(F.conv_transpose2d),
    str(F.conv_transpose3d),
    str(torch.conv_transpose2d),
    str(torch.conv_transpose3d),
    str(F.linear),
    str(torch._C._nn.linear),
]

conv_linear_modules = [
    # str(torch.nn.Conv1d) # it will be enabled at next step.
    str(torch.nn.Conv2d),
    str(torch.nn.Conv3d),
    str(torch.nn.ConvTranspose2d),
    str(torch.nn.ConvTranspose3d),
    str(torch.nn.Linear),
]

embedding_op = [
    str(F.embedding_bag),
    str(torch.embedding_bag),
]


def op_needs_quantization(op: Callable) -> bool:
    if (
        op in functions_supported_by_quantization
        or op in functions_supported_by_quantization_ipex
    ):
        return True
    elif type(op) in module_types_supported_by_quantization:
        if op in may_inplace_module and op.inplace:
            return False
        return True
    else:
        return False


def ops_are_related(
    cur_op: Callable,
    expected_op_type: str,
    type_is_module: bool,
) -> bool:
    r"""
    This function is to check whether the cur_op is align with the saved op_type, which make sure
    the model doesn't have dynamic workflow.
    """
    if type_is_module:
        cur_op = type(cur_op)
    return (
        str(cur_op) == expected_op_type
        or (str(cur_op), expected_op_type) in a_related_to_b
    )


def _raise_obs_not_found_error(func):
    raise RuntimeError(
        f"Encountered arithmetic operation {torch.typename(func)} but we have "
        f"encountered fewer arithmetic operations in previous calibration runs. "
        f"This likely indicates that the program contains dynamic control flow. "
        f" Quantization is not defined over dynamic control flow!"
    )


def _raise_obs_op_mismatch(func, prev_op):
    raise RuntimeError(
        f"Encountered arithmetic operation {torch.typename(func)} but previously "
        f"recorded operation was {prev_op}!. This likely indicates "
        f"that the program contains dynamic control flow. Quantization is not "
        f"defined over dynamic control flow!"
    )


@dataclasses.dataclass
class QTensorInfo:
    id: int  # tensor ID
    orig_dtype: torch.dtype  # dtype seen while tracing with example input
    inf_dtype: torch.dtype  # dtype at inference


@dataclasses.dataclass
class SeenQOpInfo:
    idx: int
    # Python type of the seen op. For modules, this is str(type(mod)). For
    # functions, this is the target function(str).
    type: str
    # True if the type is a module, False otherwise (for functions/methods).
    type_is_module: bool
    # Note: FQN refers to the current module for modules and to the parent
    # module for functions
    fqn: str
    # Information about the input tensors
    # Non-tensor inputs are represented with None.
    input_tensor_infos: List[Optional[QTensorInfo]]
    # We use input_tensor_infos's inf_dtype to check whether we need add fake quant
    # at convert step, but sometimes, the QTensorInfo's infor may used by many
    # operators, and one operator may set QTensorInfo' inf dtype to fp32, which hope
    # use fp32 kernel, but the cur op hope use low-precison op, so we introduce this flag
    # to fix the multi-use case: if input_tensor_force_inf_dtype has low-precison, we will
    # ignore the related QTensorInfo's inf dtype even QTensorInfo's inf dtype is fp32 dtype.
    # Note: the inint value of the QTensorInfo's  is orig dtype.
    input_tensor_force_inf_dtype: List[Optional[torch.dtype]]
    # Information about the output tensors
    # Non-tensor outputs are represented with None.
    output_tensor_infos: List[QTensorInfo]
    # Some operator only support INT8->INT8, if post operator is non-quantized op,
    # the output_tensor_infos's inf dtype always same as orig dtype, we can set the output_tensor_infos's
    # inf dtype to int8, and do a check whether add fake quant after output according to the inf dtype,
    # but if the post operator is quantized op, we will add two fake quant if we only check the inf dtype.
    # so we introduce insert_fake_quant_after_output to fix this issue: if insert_fake_quant_after_output is true,
    # and the the inf dtype is int8, we will add fake quant after the output, otherwise, we will not insert fake quant
    # after the output(if inf dtype is int8, but insert_fake_quant_after_output is False, the post op will insert
    # fake quant, if inf dtype is not int8, the output hopes a orig dtype, we don't need to add fake quant).
    # Note: the init value of the insert_fake_quant_after_output's is False.
    # Our Quant param binding algorithm (binding info used to decide whether to add q/dq at runtime) is that:
    # 1. Bind input tensors by default for all quantized ops.
    # 2. Bind output tensor if any of downstream ops is not quantized.
    insert_fake_quant_after_outputs: List[Optional[bool]]
    weight_tensor_infos: List[Optional[QTensorInfo]]
    qconfig: torch.ao.quantization.QConfig

    def __repr__(self) -> str:
        s = f"(type): {self.type}\n"
        s += f"     (fqn): {self.fqn}\n"
        s += f"     (input_tensor_infos): {self.input_tensor_infos}\n"
        s += f"     (input_tensor_force_inf_dtype): {self.input_tensor_force_inf_dtype}\n"
        s += f"     (output_tensor_infos): {self.output_tensor_infos}\n"
        s += f"     (insert_fake_quant_after_outputs): {self.insert_fake_quant_after_outputs}\n"
        s += f"     (weight_tensor_infos): {self.weight_tensor_infos}\n"
        s += f"     (qconfig): {self.qconfig}"
        return s


@dataclasses.dataclass
class SeenNonQOpInfo:
    # Python type of the seen op. For modules, this is str(type(mod)). For
    # functions, this is the target function.
    type: str
    # Note: FQN refers to the current module for modules and to the parent
    # module for functions
    fqn: str
    # Information about the input tensors
    # Non-tensor inputs are represented with None.
    input_tensor_infos: List[Optional[QTensorInfo]]
    # Information about the output tensors
    # Non-tensor outputs are represented with None.
    output_tensor_infos: List[QTensorInfo]


def get_input_observed_arg_idxs(
    op_type: str,
    op_type_is_module: bool,
) -> Optional[List[int]]:
    if op_type_is_module and op_type not in (
        str(torch.nn.EmbeddingBag),
        str(MergedEmbeddingBagWithCat),
    ):
        # TODO(future PR): handle RNNs
        return [0]
    elif op_type in conv_linear_ops:
        return [0, 1]
    elif op_type in embedding_op:
        return [1]
    # None means "observe all Tensor args"
    return None


def get_weight_arg_idx(op: str) -> Optional[int]:
    if op in conv_linear_ops:
        return 1
    return None


def set_tensor_info_dtype(tensor_info: QTensorInfo, observer):
    """
    This function is expected to be called on the prepare step which is tensor_info's
    inf_dtype is not same as observe's dtype when user load a changed configure json file.
    """
    quantized_dtype = [torch.quint8, torch.qint8]
    if (
        tensor_info.inf_dtype in quantized_dtype
        and tensor_info.inf_dtype != tensor_info.orig_dtype
        and tensor_info.inf_dtype != observer.dtype
    ):
        tensor_info.inf_dtype = observer.dtype


def iterate_and_apply(
    args: Any,
    flattened_tensor_infos: List[Optional[QTensorInfo]],
    func: Callable,
    flattened_tensor_infos_idx=None,
) -> Any:
    """
    Inputs:
      `args`: arguments to a function, may contain nested types, for example:
        ([torch.Tensor, torch.Tensor], int, (int, int))
      `flattened_tensor_infos`: tensor information containers for each tensor
        in `args`, flattened, for example corresponding with above:
        ({...}, {...}, None, None, None)
      `func`: function to apply to each tensor in `args` to create `new_args`
    Returns `new_args`, where each tensor has been transformed by `func`.
    """
    if flattened_tensor_infos_idx is None:
        flattened_tensor_infos_idx = [0]

    if isinstance(args, tuple):
        new_args = []
        for arg in args:
            new_arg = iterate_and_apply(
                arg, flattened_tensor_infos, func, flattened_tensor_infos_idx
            )
            new_args.append(new_arg)
        return tuple(new_args)
    elif isinstance(args, list):
        for idx in range(len(args)):
            new_arg = iterate_and_apply(
                args[idx], flattened_tensor_infos, func, flattened_tensor_infos_idx
            )
            args[idx] = new_arg
        return args
    else:
        # individual element
        cur_flattened_tensor_info = flattened_tensor_infos[
            flattened_tensor_infos_idx[0]
        ]
        flattened_tensor_infos_idx[0] += 1

        if cur_flattened_tensor_info is not None:
            return func(args, cur_flattened_tensor_info)
        else:
            return args


def iterate_and_apply_convert(
    args: Any,
    quant_infos: List[Optional[Tuple[float, int, torch.dtype]]],
    quant_or_dequant_needed: List[bool],
    op: Callable,
    flattened_tensor_infos_idx=None,
) -> Any:
    """
    Inputs:
      `args`: arguments to a function, may contain nested types, for example:
        ([torch.Tensor, torch.Tensor], int, (int, int))
      `quant_infos`: tensor information containers for each tensor
        in `args`, flattened, for example corresponding with above:
        ({...}, {...}, None, None, None)
       `quant_or_dequant_needed`: tensor information about whether do quantization
        containers for each tensorin `args`,
      `op`: cur quantizable op
    Returns `new_args`, where each tensor has been transformed by `func`.
    """

    if flattened_tensor_infos_idx is None:
        flattened_tensor_infos_idx = [0]
    if isinstance(args, tuple):
        new_args = []
        for arg in args:
            new_arg = iterate_and_apply_convert(
                arg,
                quant_infos,
                quant_or_dequant_needed,
                op,
                flattened_tensor_infos_idx,
            )
            new_args.append(new_arg)
        return tuple(new_args)
    elif isinstance(args, list):
        new_args = []
        for arg in args:
            new_arg = iterate_and_apply_convert(
                arg,
                quant_infos,
                quant_or_dequant_needed,
                op,
                flattened_tensor_infos_idx,
            )
            new_args.append(new_arg)
        return new_args
    else:
        # individual element
        cur_quant_infos = quant_infos[flattened_tensor_infos_idx[0]]
        cur_quant_or_dequant_needed = quant_or_dequant_needed[
            flattened_tensor_infos_idx[0]
        ]
        if (
            cur_quant_infos is not None
            and cur_quant_or_dequant_needed
            and isinstance(args, torch.Tensor)
        ):
            scale, zp, dtype = cur_quant_infos
            # For F.Linear, F.conv, the weight's may use per_channel.
            if (
                str(op) in conv_linear_ops
                and get_weight_arg_idx(str(op)) == flattened_tensor_infos_idx[0]
                and isinstance(scale, torch.Tensor)
                and scale.numel() > 1
            ):
                ch_axis = 0
                # conv_transpose's weight is iohw or iodhw
                if str(op) in [
                    str(F.conv_transpose2d),
                    str(torch.conv_transpose2d),
                    str(F.conv_transpose3d),
                    str(torch.conv_transpose3d),
                ]:
                    ch_axis = 1
                if (
                    torch.is_autocast_enabled("cpu")
                    and torch.get_autocast_dtype("cpu") == torch.bfloat16
                ):
                    # do autocast in Python side
                    if args.dtype == torch.float32:
                        args = args.to(dtype=torch.float32)
                    args = torch.quantize_per_channel(args, scale, zp, ch_axis, dtype)
                    args = args.dequantize()
                    args = args.to(dtype=torch.bfloat16)
                else:
                    args = torch.quantize_per_channel(args, scale, zp, ch_axis, dtype)
                    args = args.dequantize()
            else:
                # white list, conv, linear, matmul, we always convert it's input to bflat16 firstly, and then inser q+dq
                if (
                    str(op)
                    in conv_linear_ops
                    + [
                        str(torch.matmul),
                        str(torch.Tensor.matmul),
                        str(torch.bmm),
                        str(torch.Tensor.bmm),
                    ]
                    + embedding_op
                    or str(type(op)) in conv_linear_modules
                ):
                    if (
                        torch.is_autocast_enabled("cpu")
                        and torch.get_autocast_dtype("cpu") == torch.bfloat16
                    ):
                        if args.dtype == torch.bfloat16:
                            args = args.to(dtype=torch.float32)
                        args = torch.quantize_per_tensor(
                            args, scale.item(), zp.item(), dtype
                        )
                        args = args.dequantize()
                        args = args.to(dtype=torch.bfloat16)
                    else:
                        args = torch.quantize_per_tensor(
                            args, scale.item(), zp.item(), dtype
                        )
                        args = args.dequantize()
                else:
                    # fall through
                    args_is_bfloat16 = False
                    if args.dtype == torch.bfloat16:
                        args_is_bfloat16 = True
                        args = args.to(dtype=torch.float32)
                    args = torch.quantize_per_tensor(
                        args, scale.item(), zp.item(), dtype
                    )
                    args = args.dequantize()
                    if args_is_bfloat16:
                        args = args.to(dtype=torch.bfloat16)
        flattened_tensor_infos_idx[0] += 1
        return args


def get_input_args_quant_dequant_info(
    seen_q_op_info: SeenQOpInfo,
    tensor_id_to_scale_zp: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[List[Optional[Tuple[float, int, torch.dtype]]], List[bool], bool]:
    """
    Returns a list of information about the tensor inputs to the current op.
    Quant list:
    For each tensor input:
    * if the tensor input needs a quant, the list will contain
      (scale, zero_point)
    * if the tensor input does not need a quant, the list will contain None
    """
    quant_infos: List[Optional[Tuple[float, int, torch.dtype]]] = []
    quantized_dtype = [torch.quint8, torch.qint8]
    any_arg_quant_or_dequant_needed = []
    if len(seen_q_op_info.input_tensor_infos) > 0:
        for i, input_arg in enumerate(seen_q_op_info.input_tensor_infos):
            if input_arg is not None:
                if input_arg.id in tensor_id_to_scale_zp:
                    tensor_id = input_arg.id
                    inf_dtype = input_arg.inf_dtype
                    # force_inf_dtype always should be same as input_arg.inf_dtype, but some time,
                    # the input arg may be used by many other operators, and it may have been
                    # changed by other operators, so for cur op, twe check whether input_arg.inf_dtype
                    # is same as the origin force_inf_dtype, if not same use force_inf_dtype as new
                    # inf dtype, if same, we can say the input_arg.inf_dtype is not changed or the cur op
                    # changed input_arg.inf_dtype and force_inf_dtype at get default recipe step.
                    if (
                        seen_q_op_info.input_tensor_force_inf_dtype[i]
                        != input_arg.inf_dtype
                    ):
                        inf_dtype = seen_q_op_info.input_tensor_force_inf_dtype[i]

                    scale, zp = tensor_id_to_scale_zp[tensor_id]
                    quant_infos.append((scale, zp, inf_dtype))  # type: ignore[arg-type]
                    # only support float to int8.
                    if (
                        input_arg.orig_dtype == torch.float32
                        and inf_dtype in quantized_dtype
                    ):
                        any_arg_quant_or_dequant_needed.append(True)
                    else:
                        any_arg_quant_or_dequant_needed.append(False)
                else:
                    quant_infos.append(None)
                    any_arg_quant_or_dequant_needed.append(False)
            else:
                quant_infos.append(None)
                any_arg_quant_or_dequant_needed.append(None)
    return quant_infos, any_arg_quant_or_dequant_needed


def get_weight_args_quant_dequant_info(
    seen_q_op_info: SeenQOpInfo,
    weight_tensor_id_to_scale_zp: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[List[Optional[Tuple[float, int, torch.dtype]]], List[bool], bool]:
    """
    Returns a list of information about the tensor inputs to the current op.
    """
    quant_infos: List[Optional[Tuple[float, int, torch.dtype]]] = []
    any_arg_quant_or_dequant_needed = []
    for _, input_arg in enumerate(seen_q_op_info.weight_tensor_infos):
        if input_arg is not None:
            tensor_id = str(seen_q_op_info.idx) + "_" + str(input_arg.id)
            if tensor_id in weight_tensor_id_to_scale_zp:
                scale, zp = weight_tensor_id_to_scale_zp[tensor_id]
                output_dtype = input_arg.inf_dtype
                quant_infos.append((scale, zp, output_dtype))  # type: ignore[arg-type]
                if input_arg.orig_dtype == torch.float32 and input_arg.inf_dtype in [
                    torch.quint8,
                    torch.qint8,
                ]:
                    any_arg_quant_or_dequant_needed.append(True)
                else:
                    any_arg_quant_or_dequant_needed.append(False)
            else:
                quant_infos.append(None)
                any_arg_quant_or_dequant_needed.append(False)
        else:
            quant_infos.append(None)
            any_arg_quant_or_dequant_needed.append(None)
    return quant_infos, any_arg_quant_or_dequant_needed
