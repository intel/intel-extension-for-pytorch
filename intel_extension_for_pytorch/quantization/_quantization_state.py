from typing import Callable, List, Tuple, Any, Optional, Dict
import torch
import torch.nn.functional as F
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithCat

from ._utils import (
    OpQuantizeabilityType,
    is_leaf,
    get_fqn_valid_for_module_dict_key,
    quantized_modules_has_weights,
    int8_int8_ops,
)
from ._quantization_state_utils import (
    SeenQOpInfo,
    SeenNonQOpInfo,
    QTensorInfo,
    op_needs_quantization,
    get_input_observed_arg_idxs,
    get_weight_arg_idx,
    iterate_and_apply,
    get_input_args_quant_dequant_info,
    _raise_obs_not_found_error,
    get_weight_args_quant_dequant_info,
    _raise_obs_op_mismatch,
    ops_are_related,
    iterate_and_apply_convert,
    set_tensor_info_dtype,
)
from ._smooth_quant import SmoothQuantActivationObserver, SmoothQuantWeightObserver


OpConvertInfo = Tuple[
    # quantized equivalent of original op (None means keep original)
    # Optional[Callable],
    # arg_quant_infos, each element is (scale, zp, dtype) for quantized and None otherwise
    List[Optional[Tuple[float, int, torch.dtype]]],
    List[bool],
]


# TODO(future PR): maybe better name
# TODO(future PR): add serialization support
class AutoQuantizationState(torch.nn.Module):
    """
    Contains state necessary to perform auto quantization on the parent
    `nn.Module` instance.
    """

    idx: int

    def __init__(self, fqn: str, qconfig: torch.ao.quantization.QConfig):
        super().__init__()
        self.idx = 0
        self.qconfig = qconfig
        self.fqn = fqn
        # this is a ModuleDict in order to properly register observers
        # to be within the module hierarchy.
        self.tensor_id_to_observer = torch.nn.ModuleDict()
        self.weight_tensor_id_to_observer = torch.nn.ModuleDict()

        # TODO(future PR): include kwargs
        # Note: seen quantizeable ops are recorded with an index,
        # because we enforce order of execution. However, seen
        # unquantizeable ops are recorded without an index, because
        # we do not enforce order of execution.
        self.idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo] = {}
        self.seen_nonq_op_infos: List[SeenNonQOpInfo] = []

        # qtensor_info objects of tensor outputs of the module, specified
        # in order of iteration through the output type. Non-tensor outputs
        # are represented with `None`.
        self.output_qtensor_infos: List[Optional[QTensorInfo]] = []
        # note: this is filled out right before convert
        self.tensor_id_to_scale_zp: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.idx_to_op_convert_info: Dict[int, OpConvertInfo] = {}
        self.weight_tensor_id_to_scale_zp: Dict[
            str, Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.idx_to_op_weight_convert_info: Dict[int, OpConvertInfo] = {}
        self.tensor_id_to_smooth_quant_scaling_factor: Dict[int, torch.Tensor] = {}
        self.weight_tensor_id_to_smooth_quant_scaling_factor: Dict[
            int, torch.Tensor
        ] = {}
        self.idx_to_smooth_quant_scaling_factor: Dict[str, torch.Tensor] = {}
        self.idx_to_weight_updated_for_smooth_quant: set[str] = set()

    def get_extra_state(self):
        return {"tensor_id_to_scale_zp": self.tensor_id_to_scale_zp}

    def set_extra_state(self, state):
        self.tensor_id_to_scale_zp = state["tensor_id_to_scale_zp"]
        for _, seen_q_op_info in self.idx_to_seen_q_op_infos.items():
            self.idx_to_op_convert_info[seen_q_op_info.idx] = (
                self.calculate_op_convert_info(seen_q_op_info)
            )

    def has_at_least_one_seen_q_op_info(self) -> bool:
        return len(self.idx_to_seen_q_op_infos) > 0

    def validate_is_at_last_seen_idx(self) -> None:
        is_at_last_seen_idx = len(self.idx_to_seen_q_op_infos) == 0 or self.idx == len(
            self.idx_to_seen_q_op_infos
        )
        if not is_at_last_seen_idx:
            raise AssertionError(
                f"Cur idx: {self.idx}, expected idx: {len(self.idx_to_seen_q_op_infos)}"
            )

    def extra_repr(self) -> str:
        s = ""
        # idx_to_seen_q_op_infos
        if len(self.idx_to_seen_q_op_infos):
            s += "(seen_q_op_infos): {\n"
            for k, v in self.idx_to_seen_q_op_infos.items():
                s += f"  {k}: {v}\n"
            s += "}\n"
        else:
            s += "(seen_q_op_infos): {}\n"
        if len(self.seen_nonq_op_infos):
            s += "(seen_nonq_op_infos): {\n"
            for n in self.seen_nonq_op_infos:
                s += f"  {n}\n"
            s += "}\n"
        else:
            s += "(seen_nonq_op_infos): {}\n"
        # output_qtensor_infos
        s += "(output_qtensor_infos): ["
        for i in self.output_qtensor_infos:
            s += f"{i} "
        s += "]\n"
        if len(self.tensor_id_to_scale_zp):
            s += "(tensor_id_to_scale_zp): {\n"
            for k, v in self.tensor_id_to_scale_zp.items():  # type: ignore[assignment]
                s += f"  {k}: {v}\n"
            s += "}\n"
        if len(self.weight_tensor_id_to_scale_zp):
            s += "(weight_tensor_id_to_scale_zp): {\n"
            for k, v in self.weight_tensor_id_to_scale_zp.items():  # type: ignore[assignment]
                s += f"  {k}: {v}\n"
            s += "}"

        return s

    def _get_cur_seen_q_op_info(self):
        return self.idx_to_seen_q_op_infos[self.idx]

    def get_cur_output_inf_dtype(self):
        return self._get_cur_seen_q_op_info().output_tensor_infos[0].inf_dtype

    def reset_to_new_call(self):
        """
        Resets the internal op counter to start a new top level module call
        """
        # torch.nn.Module __setattr__ has overhead,
        # this code is the explicit fast path for `self.idx = 0`
        object.__setattr__(self, "idx", 0)

    def cur_op_needs_hooks(self, cur_op: Callable) -> bool:
        return op_needs_quantization(cur_op)

    def validate_cur_op(self, cur_op: Callable) -> None:
        """
        This function is expected to be called before any new function or
        module call which needs hooks. It validates that the new function or
        module is of the expected type based on the order of execution.
        """
        try:
            seen_q_op_info = self._get_cur_seen_q_op_info()
            expected_op = seen_q_op_info.type
        except IndexError:
            _raise_obs_not_found_error(cur_op)
        if not ops_are_related(cur_op, expected_op, seen_q_op_info.type_is_module):
            _raise_obs_op_mismatch(cur_op, expected_op)

    def mark_cur_op_complete(self, cur_op: Callable) -> None:
        """
        This function is expected to be called after a function or module
        processing is complete.
        """
        # torch.nn.Module __setattr__ has overhead,
        # this code is the explicit fast path for `self.idx += 1`
        object.__setattr__(self, "idx", self.idx + 1)

    def first_call_outputs_prepare_hook(
        self,
        outputs: Any,
        qtensor_id: List[int],
    ) -> Any:
        """
        This function is expected to be called on the outputs of a prepared
        module right before they are returned to the parent, during tracing.
        """
        outputs = self._first_call_assign_qtensor_infos_to_mod_outputs(
            outputs, qtensor_id
        )
        return outputs

    def outputs_prepare_hook(
        self,
        outputs: Any,
    ) -> Any:
        """
        This function is expected to be called on the outputs of a prepared
        module right before they are returned to the parent.
        """
        return outputs

    def outputs_convert_hook(
        self,
        outputs: Any,
    ) -> Any:
        """
        This function is expected to be called on the outputs of a converted
        module right before they are returned to the parent.
        """
        # outputs = self._maybe_mod_outputs_dtype_transform(outputs)
        return outputs

    def first_call_op_prepare_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        qtensor_id: List[int],
        fqn: str,
        root_module: torch.nn.Module,
        op_quantizeability_type: OpQuantizeabilityType,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is expected to be called on args and kwargs of
        `op` directly before `op` is executed, during tracing.
        We record the type of `op`
        and the IDs of its tensor inputs. Note: we add a placeholder for IDs
        of tensor outputs, the placeholder will be filled out during the
        `op_prepare_after_hook`.
        The function returns modified `args` and `kwargs`.
        """
        return self._first_call_op_prepare_before_hook_create_subgraphs(
            op, args, kwargs, qtensor_id, fqn, root_module, op_quantizeability_type
        )

    def op_prepare_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is expected to be called on args and kwargs of
        `op` directly before `op` is executed.
        We do the following:
        * pass the inputs through observers, if needed
        The function returns modified `args` and `kwargs`.
        """
        seen_q_op_info = self._get_cur_seen_q_op_info()

        def _maybe_observe(arg, tensor_info):
            tensor_id = tensor_info.id
            # TODO: do not run this twice on input and output
            if str(tensor_id) in self.tensor_id_to_observer:
                observer = self.tensor_id_to_observer[str(tensor_id)]
                if isinstance(arg, torch.Tensor) and arg.dtype != torch.float32:
                    dtype = arg.dtype
                    out = observer(arg.float())
                    return out.to(dtype)
                return observer(arg)
            else:
                return arg

        # If user changes observer's dtype and re-do calibration, we need to update
        # the tensor_info.inf_dtype  and force dtype with the new oberver's dtype.
        quantized_dtype = [torch.quint8, torch.qint8]
        for i, tensor_info in enumerate(seen_q_op_info.input_tensor_infos):
            if (
                tensor_info is not None
                and str(tensor_info.id) in self.tensor_id_to_observer
            ):
                tensor_id = tensor_info.id
                observer = self.tensor_id_to_observer[str(tensor_id)]
                set_tensor_info_dtype(tensor_info, observer)
                force_dtype = seen_q_op_info.input_tensor_force_inf_dtype[i]
                if (
                    force_dtype in quantized_dtype
                    and force_dtype != tensor_info.orig_dtype
                    and force_dtype != observer.dtype
                ):
                    seen_q_op_info.input_tensor_force_inf_dtype[i] = observer.dtype
        args = iterate_and_apply(
            args, seen_q_op_info.input_tensor_infos, _maybe_observe
        )
        # works for nn.module case
        weight_tensor_info = seen_q_op_info.weight_tensor_infos
        for i, tensor_info in enumerate(weight_tensor_info):
            if tensor_info is None:
                continue
            tensor_id = tensor_info.id
            if (
                str(seen_q_op_info.idx) + "_" + str(tensor_id)
                in self.weight_tensor_id_to_observer
            ):
                observer = self.weight_tensor_id_to_observer[
                    str(seen_q_op_info.idx) + "_" + str(tensor_id)
                ]
                set_tensor_info_dtype(tensor_info, observer)
                # if has bias, the dim is 1, we don't need run observer for it.
                if isinstance(op, torch.nn.LSTM):
                    if op._flat_weights[i].dim() > 1:
                        observer(op._flat_weights[i])
                    else:
                        pass
                elif isinstance(op, MergedEmbeddingBagWithCat):
                    observer(op.weights[i])
                else:
                    observer(op.weight)

        return args, kwargs

    def first_call_op_prepare_after_hook(
        self,
        op: Callable,
        output: Any,
        args: Tuple[Any, ...],
        qtensor_id: List[int],
        op_quantizeability_type: OpQuantizeabilityType,
    ) -> Any:
        """
        This function is called after an op call on a prepared model.
        * create an observer for the output, if needed, and record it in
          `tensor_id_to_observer`
        * amend the current seen op with the tensor ID of the output
        """
        self._first_call_op_prepare_after_hook_adjust_subgraphs(
            op, output, args, qtensor_id, op_quantizeability_type
        )
        return output

    def op_prepare_after_hook(
        self,
        op: Callable,
        outputs: Any,
        args: Tuple[Any, ...],
        global_op_idx: List[int],
    ) -> Any:
        """
        This function is called after an op call on a prepared model.
        * observe the output, if needed, which only works for OpQuantizeabilityType.QUANTIZEABLE.
        TODO: remove this after all ops support INT8->FP32.
        """
        seen_q_op_info = self._get_cur_seen_q_op_info()

        def _observer_output(output, tensor_info):
            tensor_id = tensor_info.id
            if str(tensor_id) in self.tensor_id_to_observer:
                observer = self.tensor_id_to_observer[str(tensor_id)]
                set_tensor_info_dtype(tensor_info, observer)
                observer(output.float())

        if isinstance(outputs, torch.Tensor):
            tensor_info = seen_q_op_info.output_tensor_infos[0]
            _observer_output(outputs, tensor_info)
        elif isinstance(outputs, tuple):
            idx = 0
            for element in outputs:
                # only do observer for tensor type.
                if isinstance(element, torch.Tensor):
                    tensor_info = seen_q_op_info.output_tensor_infos[idx]
                    _observer_output(element, tensor_info)
                    idx += 1
        return outputs

    def op_convert_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        root_module: torch.nn.Module,
    ) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is called before an op call in a converted model.
        For each arg in `args`, quantizes it if necessary.
        Returns potentially modified `op`, potentially modified `args`,
        potentially modified `kwargs`.
        """
        # TODO generalize this for more things
        # currently:
        # * can quantize args (via arg_quant_infos)
        # * can add scale and zp (via additional kwargs)
        arg_quant_infos, any_arg_quant_or_dequant_needed = self.get_op_convert_info(op)
        # Insert mul before nn.Linear for SmoothQuant
        act_key = str(self.idx)
        if act_key in self.idx_to_smooth_quant_scaling_factor:
            act_scaling_factors = self.idx_to_smooth_quant_scaling_factor[act_key]
            # if users modifies qconf.json and cancals quantization of the linear,
            # then any_arg_quant_or_dequant_needed[0] is False. Don't insert mul in this case.
            if act_scaling_factors is not None and any_arg_quant_or_dequant_needed[0]:
                w_key = str(self.idx) + "_0"
                act_scaling_factors = (
                    act_scaling_factors[w_key]
                    if len(act_scaling_factors) > 1
                    else next(iter(act_scaling_factors.values()))
                )
                # update arg_quant_infos
                if isinstance(arg_quant_infos[0][0], dict):
                    scale = (
                        arg_quant_infos[0][0][w_key]
                        if len(arg_quant_infos[0][0]) > 1
                        else next(iter(arg_quant_infos[0][0].values()))
                    )
                    zp = (
                        arg_quant_infos[0][1][w_key]
                        if len(arg_quant_infos[0][1]) > 1
                        else next(iter(arg_quant_infos[0][1].values()))
                    )
                else:
                    # For backward compatibility
                    assert isinstance(arg_quant_infos[0][0], torch.Tensor)
                    scale = arg_quant_infos[0][0]
                    zp = arg_quant_infos[0][1]
                arg_quant_infos = [(scale, zp, arg_quant_infos[0][2])]
                args = list(args)
                new_act = torch.mul(args[0], act_scaling_factors)
                args[0] = new_act
        args = iterate_and_apply_convert(
            args, arg_quant_infos, any_arg_quant_or_dequant_needed, op
        )
        return op, args, kwargs

    def op_weight_convert_before_hook(
        self,
        op: Callable,
    ) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is called before an op call in a converted model.
        For each arg in `args`, quantizes it if necessary.
        Returns potentially modified `op`, potentially modified `args`,
        potentially modified `kwargs`.
        """
        (
            arg_quant_infos,
            any_arg_quant_or_dequant_needed,
        ) = self.get_op_weight_convert_info(op)
        new_args = []
        if type(op) in [
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
            torch.nn.Linear,
        ]:
            tensor_arg_idx = 0
            quant_info = arg_quant_infos[tensor_arg_idx]
            if (
                quant_info is not None
                and any_arg_quant_or_dequant_needed[tensor_arg_idx]
            ):
                scale, zp, dtype = quant_info
                weight = op.weight
                ch_axis = 0
                if type(op) in [torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d]:
                    ch_axis = 1
                # Update weight of nn.Linear for SmoothQuant
                wei_key = str(self.idx) + "_0"
                if wei_key in self.idx_to_smooth_quant_scaling_factor:
                    wei_scaling_factors = self.idx_to_smooth_quant_scaling_factor[
                        wei_key
                    ]
                    if wei_scaling_factors is not None:
                        w_dtype = weight.dtype
                        if w_dtype != torch.float32:
                            weight = weight.to(torch.float32)
                        weight = torch.mul(weight, wei_scaling_factors)
                        if w_dtype != torch.float32:
                            weight = weight.to(w_dtype)
                if (
                    torch.is_autocast_enabled("cpu")
                    and torch.get_autocast_dtype("cpu") == torch.bfloat16
                ):
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    if scale.numel() > 1:
                        arg = torch.quantize_per_channel(
                            weight, scale, zp, ch_axis, dtype
                        )
                    else:
                        arg = torch.quantize_per_tensor(
                            weight, scale.item(), zp.item(), dtype
                        )
                    arg = arg.dequantize()
                    arg = arg.to(dtype=torch.bfloat16)
                else:
                    if scale.numel() > 1:
                        arg = torch.quantize_per_channel(
                            weight, scale, zp, ch_axis, dtype
                        )
                    else:
                        arg = torch.quantize_per_tensor(
                            weight, scale.item(), zp.item(), dtype
                        )
                    arg = arg.dequantize()
                new_args.append(arg)
            else:
                new_args.append(op.weight)
        elif isinstance(op, torch.nn.EmbeddingBag):
            tensor_arg_idx = 0
            quant_info = arg_quant_infos[tensor_arg_idx]
            if (
                quant_info is not None
                and any_arg_quant_or_dequant_needed[tensor_arg_idx]
            ):
                scale, zp, dtype = quant_info
                weight = op.weight
                if (
                    torch.torch.is_autocast_enabled("cpu")
                    and torch.get_autocast_dtype("cpu") == torch.bfloat16
                ):
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    arg = torch.quantize_per_tensor(
                        weight, scale.item(), zp.item(), dtype
                    )
                    arg = arg.dequantize()
                    arg = arg.to(dtype=torch.bfloat16)
                else:
                    arg = torch.quantize_per_tensor(
                        op.weight, scale.item(), zp.item(), dtype
                    )
                    arg = arg.dequantize()
                new_args.append(arg)
            else:
                new_args.append(op.weight)
        elif isinstance(op, MergedEmbeddingBagWithCat):
            weights = op.weights
            for tensor_arg_idx in range(0, len(arg_quant_infos)):
                quant_info = arg_quant_infos[tensor_arg_idx]
                if (
                    quant_info is not None
                    and any_arg_quant_or_dequant_needed[tensor_arg_idx]
                ):
                    scale, zp, dtype = quant_info
                    if (
                        torch.torch.is_autocast_enabled("cpu")
                        and torch.get_autocast_dtype("cpu") == torch.bfloat16
                    ):
                        if weights[tensor_arg_idx].dtype == torch.bfloat16:
                            weights[tensor_arg_idx] = weights[tensor_arg_idx].to(
                                dtype=torch.float32
                            )
                        arg = torch.quantize_per_tensor(
                            weights[tensor_arg_idx], scale.item(), zp.item(), dtype
                        )
                        arg = arg.dequantize()
                        arg = arg.to(dtype=torch.bfloat16)
                    else:
                        arg = torch.quantize_per_tensor(
                            op.weights[tensor_arg_idx], scale.item(), zp.item(), dtype
                        )
                        arg = arg.dequantize()
                    new_args.append(arg)
                else:
                    new_args.append(op.weights[tensor_arg_idx])
        elif isinstance(op, torch.nn.LSTM):
            step = 4 if op.bias else 2
            weights = op._flat_weights
            for tensor_arg_idx in range(0, len(arg_quant_infos), step):
                quant_info = arg_quant_infos[tensor_arg_idx]
                if (
                    quant_info is not None
                    and any_arg_quant_or_dequant_needed[tensor_arg_idx]
                ):
                    w_ih = weights[tensor_arg_idx]
                    w_hh = weights[tensor_arg_idx + 1]
                    w_ih_scale, w_ih_zp, w_ih_dtype = quant_info
                    w_hh_scale, w_hh_zp, w_hh_dtype = arg_quant_infos[
                        tensor_arg_idx + 1
                    ]
                    if (
                        torch.torch.is_autocast_enabled("cpu")
                        and torch.get_autocast_dtype("cpu") == torch.bfloat16
                    ):
                        weight_if_bf16 = w_ih.dtype == torch.bfloat16
                        if weight_if_bf16:
                            w_ih = w_ih.to(dtype=torch.float32)
                            w_hh = w_hh.to(dtype=torch.float32)
                        if w_ih_scale.numel() > 1:
                            w_ih = torch.quantize_per_channel(
                                w_ih, w_ih_scale, w_ih_zp, 0, w_ih_dtype
                            )
                            w_hh = torch.quantize_per_channel(
                                w_hh, w_hh_scale, w_hh_zp, 0, w_hh_dtype
                            )
                        else:
                            w_ih = torch.quantize_per_tensor(
                                w_ih, w_ih_scale.item(), w_ih_zp.item(), w_ih_dtype
                            )
                            w_hh = torch.quantize_per_tensor(
                                w_hh, w_hh_scale.item(), w_hh_zp.item(), w_hh_dtype
                            )
                        w_ih = w_ih.dequantize()
                        w_hh = w_hh.dequantize()
                        if weight_if_bf16:
                            w_ih = w_ih.to(dtype=torch.bfloat16)
                            w_hh = w_hh.to(dtype=torch.bfloat16)
                    else:
                        if w_ih_scale.numel() > 1:
                            w_ih = torch.quantize_per_channel(
                                w_ih, w_ih_scale, w_ih_zp, 0, w_ih_dtype
                            )
                            w_hh = torch.quantize_per_channel(
                                w_hh, w_hh_scale, w_hh_zp, 0, w_hh_dtype
                            )
                        else:
                            w_ih = torch.quantize_per_tensor(
                                w_ih, w_ih_scale, w_ih_zp, w_ih_dtype
                            )
                            w_hh = torch.quantize_per_tensor(
                                w_hh, w_hh_scale, w_hh_zp, w_hh_dtype
                            )
                        w_ih = w_ih.dequantize()
                        w_hh = w_hh.dequantize()
                    new_args.append(w_ih)
                    new_args.append(w_hh)
                    if op.bias:
                        new_args.append(weights[tensor_arg_idx + 2])
                        new_args.append(weights[tensor_arg_idx + 3])
                else:
                    for s in range(step):
                        new_args.append(weights[tensor_arg_idx + s])

        return new_args

    def op_convert_after_hook(
        self,
        op: Callable,
        outputs,
    ) -> Any:
        """
        This function is called after an op call in a converted model.
        """
        # we always add fakeQuant before the quantized op, but if one op doesn't support INT8->FP32,
        # we need add fakeQuant here to make the quantized op call in
        # INT8 path. It can be removed after all op support INT8->fp32
        seen_q_op_info = self._get_cur_seen_q_op_info()

        def _convert_output(
            output, tensor_info, insert_fake_quant, tensor_id_to_scale_zp
        ):
            tensor_id, inf_dtype = tensor_info.id, tensor_info.inf_dtype
            # so if inf_dtype is torch.qint8, we need add fake quant here.
            if (
                tensor_id in tensor_id_to_scale_zp
                and inf_dtype in [torch.qint8, torch.quint8]
                and insert_fake_quant
            ):
                scale, zp = tensor_id_to_scale_zp[tensor_id]
                output_is_bfloat16 = False
                if output.dtype == torch.bfloat16:
                    output_is_bfloat16 = True
                    output = output.to(torch.float32)
                output = torch.quantize_per_tensor(
                    output, scale.item(), zp.item(), inf_dtype
                )
                output = output.dequantize()
                if output_is_bfloat16:
                    output = output.to(torch.bfloat16)
            return output

        if isinstance(outputs, torch.Tensor):
            tensor_info = seen_q_op_info.output_tensor_infos[0]
            insert_fake_quant = seen_q_op_info.insert_fake_quant_after_outputs[0]
            outputs = _convert_output(
                outputs, tensor_info, insert_fake_quant, self.tensor_id_to_scale_zp
            )
        elif isinstance(outputs, tuple):
            # TODO: handle other tuple subclasses more generically
            new_outputs = []
            idx = 0
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    tensor_info = seen_q_op_info.output_tensor_infos[idx]
                    insert_fake_quant = seen_q_op_info.insert_fake_quant_after_outputs[
                        idx
                    ]
                    output = _convert_output(
                        output,
                        tensor_info,
                        insert_fake_quant,
                        self.tensor_id_to_scale_zp,
                    )
                    new_outputs.append(output)
                    idx += 1
                else:
                    new_outputs.append(output)
            # hacky check for collections.namedtuple, TODO improve this
            # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
            if hasattr(outputs, "_fields"):
                outputs = outputs.__class__(*new_outputs)
            else:
                outputs = tuple(new_outputs)
        else:
            pass
        return outputs

    def get_op_convert_info(
        self,
        op: Callable,
    ) -> OpConvertInfo:
        """
        Returns the information needed for convert time modifications to `op`.
        """
        return self.idx_to_op_convert_info[self.idx]

    def get_op_weight_convert_info(
        self,
        op: Callable,
    ) -> OpConvertInfo:
        """
        Returns the information needed for convert time modifications to `op`.
        """
        return self.idx_to_op_weight_convert_info[self.idx]

    def calculate_op_convert_info(
        self,
        seen_q_op_info: SeenQOpInfo,
    ) -> OpConvertInfo:
        """
        This precalculates the information which will be returned by
        `get_op_convert_info`.
        """
        # calculate quant infos
        (
            arg_quant_infos,
            any_arg_quant_or_dequant_needed,
        ) = get_input_args_quant_dequant_info(
            seen_q_op_info, self.tensor_id_to_scale_zp
        )

        return (
            arg_quant_infos,
            any_arg_quant_or_dequant_needed,
        )

    def calculate_op_weight_convert_info(
        self,
        seen_q_op_info: SeenQOpInfo,
    ) -> OpConvertInfo:
        """
        This precalculates the information which will be returned by
        `get_op_convert_info`.
        """
        # calculate quant infos
        (
            arg_quant_infos,
            any_arg_quant_or_dequant_needed,
        ) = get_weight_args_quant_dequant_info(
            seen_q_op_info, self.weight_tensor_id_to_scale_zp
        )

        return (
            arg_quant_infos,
            any_arg_quant_or_dequant_needed,
        )

    def _get_packed_param_name(self, seen_q_op_info: SeenQOpInfo) -> Optional[str]:
        """
        If the op in seen_q_op_info has a quantized packed param, returns it.
        Otherwise, returns None.
        """
        return self.idx_to_packed_weight_name.get(seen_q_op_info.idx, None)

    def _first_call_assign_qtensor_infos_to_mod_outputs_tensor(
        self,
        output: torch.Tensor,
        qtensor_id: List[int],
    ) -> torch.Tensor:
        """
        This is a helper function for _first_call_assign_qtensor_infos_to_mod_outputs
        to handle iterables of tensors without code duplication.
        """
        if not hasattr(output, "_qtensor_info"):
            output._qtensor_info = QTensorInfo(  # type: ignore[attr-defined]
                qtensor_id[0], output.dtype, output.dtype
            )
            qtensor_id[0] += 1
        self.output_qtensor_infos.append(output._qtensor_info)  # type: ignore[attr-defined]
        return output

    def _first_call_assign_qtensor_infos_to_mod_outputs(
        self,
        outputs: Any,
        qtensor_id: List[int],
    ) -> Any:
        """
        Takes `outputs`, which are a set of values about to be returned from
        the current module. If `_qtensor_info` attributes do not already exist
        on any tensors in `outputs`, this function adds them, initializing the
        dtype to `torch.float`. This allows us to reason about module output
        dtypes even if the last op in the module is not quantizeable.
        """
        # TODO: handle objects with deeper nested tensors
        if isinstance(outputs, torch.Tensor):
            self._first_call_assign_qtensor_infos_to_mod_outputs_tensor(
                outputs, qtensor_id
            )
        elif isinstance(outputs, tuple):
            # TODO: handle other tuple subclasses more generically
            new_outputs = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    new_outputs.append(
                        self._first_call_assign_qtensor_infos_to_mod_outputs_tensor(
                            output, qtensor_id
                        )
                    )
                else:
                    new_outputs.append(output)
            # hacky check for collections.namedtuple, TODO improve this
            # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
            if hasattr(outputs, "_fields"):
                outputs = outputs.__class__(*new_outputs)
            else:
                outputs = tuple(new_outputs)
        else:
            pass
        return outputs

    def _first_call_op_prepare_before_hook_create_subgraphs_tensor(
        self,
        op: Callable,
        arg: Any,
        arg_tensor_infos: List[Optional[QTensorInfo]],
        arg_tensor_force_inf_dtype: List[Optional[torch.dtype]],
        qtensor_id: List[int],
    ) -> None:
        """
        Runs the prepare hook during first_call for individual
        tensors. If the input argument is a tensor, this function is
        called directly. If the input argument is an iterable such
        as a list or a tuple, this function is called on each element of
        the iteratble.
        """
        # TODO(next): fix this for torch.cat
        if not isinstance(arg, torch.Tensor):
            arg_tensor_infos.append(None)
            arg_tensor_force_inf_dtype.append(None)
            return

        # If a tensor does not have an ID, add it. This allows
        # us to track inputs shared by multiple quantizeable modules.
        if not hasattr(arg, "_qtensor_info"):
            arg._qtensor_info = QTensorInfo(  # type: ignore[attr-defined]
                qtensor_id[0], arg.dtype, arg.dtype
            )

            qtensor_id[0] += 1
        arg_tensor_infos.append(arg._qtensor_info)  # type: ignore[attr-defined]
        arg_tensor_force_inf_dtype.append(arg.dtype)

    def _first_call_op_prepare_before_hook_create_subgraphs(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        qtensor_id: List[int],
        fqn: str,
        root_module: torch.nn.Module,
        op_quantizeability_type: OpQuantizeabilityType,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Given an op, args, kwargs about to be executed, records the subgraph
        of this op in `self`.
        """
        arg_tensor_infos: List[Optional[QTensorInfo]] = []
        arg_tensor_force_inf_dtype: List[Optional[torch.dtype]] = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for inner_arg in arg:
                    self._first_call_op_prepare_before_hook_create_subgraphs_tensor(
                        op,
                        inner_arg,
                        arg_tensor_infos,
                        arg_tensor_force_inf_dtype,
                        qtensor_id,
                    )
            else:
                self._first_call_op_prepare_before_hook_create_subgraphs_tensor(
                    op, arg, arg_tensor_infos, arg_tensor_force_inf_dtype, qtensor_id
                )

        if op_quantizeability_type is OpQuantizeabilityType.NOT_QUANTIZEABLE:
            op_type_is_module = isinstance(op, torch.nn.Module)
            op_type: Callable = type(op) if op_type_is_module else op  # type: ignore[assignment]
            self.seen_nonq_op_infos.append(
                SeenNonQOpInfo(str(op_type), fqn, arg_tensor_infos, [])
            )
            return args, kwargs

        if self.idx not in self.idx_to_seen_q_op_infos:
            op_type_is_module = isinstance(op, torch.nn.Module)
            op_type = type(op) if op_type_is_module else op  # type: ignore[assignment]
            weight_tensor_infos = []
            weight_idx = 0
            if type(op) in quantized_modules_has_weights:
                if isinstance(op, (torch.nn.LSTM, MergedEmbeddingBagWithCat)):
                    if isinstance(op, torch.nn.LSTM):
                        weights = op._flat_weights
                    else:
                        weights = op.weights
                    for i in range(len(weights)):
                        weight_tensor_infos.append(
                            QTensorInfo(
                                weight_idx,
                                weights[weight_idx].dtype,
                                weights[weight_idx].dtype,
                            )
                        )
                        weight_idx += 1
                else:
                    weight_tensor_infos.append(
                        QTensorInfo(weight_idx, op.weight.dtype, op.weight.dtype)
                    )
            self.idx_to_seen_q_op_infos[self.idx] = SeenQOpInfo(
                self.idx,
                str(op_type),
                op_type_is_module,
                fqn,
                arg_tensor_infos,
                arg_tensor_force_inf_dtype,
                [],
                [],
                weight_tensor_infos,
                self.qconfig,
            )
        return args, kwargs

    def _first_call_op_prepare_after_hook_adjust_subgraphs(
        self,
        op: Callable,
        outputs: Any,
        args: Tuple[Any, ...],
        qtensor_id: List[int],
        op_quantizeability_type: OpQuantizeabilityType,
    ) -> None:
        """
        After `op` was just executed, modifies the subgraph recorded
        for this op with the information about the output. Note, this
        has to be done in the "after" hook because the output of the op
        does not exist in the "before" hook.
        """

        # TODO(future PR): handle non-tensor outputs
        def _add_output_qtensor_info(output):
            output._qtensor_info = QTensorInfo(
                qtensor_id[0], output.dtype, output.dtype
            )  # type: ignore[arg-type]
            if op_quantizeability_type is OpQuantizeabilityType.QUANTIZEABLE:
                target = self.idx_to_seen_q_op_infos[self.idx].output_tensor_infos
                self.idx_to_seen_q_op_infos[
                    self.idx
                ].insert_fake_quant_after_outputs.append(False)
            else:
                target = self.seen_nonq_op_infos[-1].output_tensor_infos
            target.append(output._qtensor_info)
            qtensor_id[0] += 1

        if isinstance(outputs, torch.Tensor):
            _add_output_qtensor_info(outputs)
        elif isinstance(outputs, tuple):
            for element in outputs:
                if isinstance(element, torch.Tensor):
                    _add_output_qtensor_info(element)

    def _maybe_insert_input_observers(self, seen_q_op_info: SeenQOpInfo):
        input_observed_arg_idxs = get_input_observed_arg_idxs(
            seen_q_op_info.type, seen_q_op_info.type_is_module
        )

        qconfig = seen_q_op_info.qconfig
        found_duplicate_input = False
        for idx, tensor_info in enumerate(seen_q_op_info.input_tensor_infos):
            if tensor_info is None:
                continue
            if (
                input_observed_arg_idxs is not None
                and idx not in input_observed_arg_idxs
            ):
                continue
            if qconfig is None:
                # If qconfig is None, we do not need any input observers
                continue
            else:
                # always add observer if the op can be quantized.
                tensor_id = tensor_info.id  # type: ignore[attr-defined]
                weight_arg_idx = get_weight_arg_idx(seen_q_op_info.type)
                # avoid add weight observer for dynamic quantization.
                if idx == weight_arg_idx and not isinstance(
                    qconfig.activation(), torch.ao.quantization.PlaceholderObserver
                ):
                    # conv_transpose weight is iohw or iodhw, so we change the observer axis to 1.
                    if seen_q_op_info.type in [
                        str(F.conv_transpose2d),
                        str(F.conv_transpose3d),
                    ] and isinstance(
                        qconfig.weight(), torch.ao.quantization.PerChannelMinMaxObserver
                    ):
                        obs = qconfig.weight.with_args(ch_axis=1)()
                    else:
                        obs = qconfig.weight()
                else:
                    obs = qconfig.activation()
                if str(tensor_id) not in self.tensor_id_to_observer:
                    self.tensor_id_to_observer[str(tensor_id)] = obs
                else:
                    found_duplicate_input = True

        # add weight observer if the op is nn.module and has a weight.
        for tensor_info in seen_q_op_info.weight_tensor_infos:
            if tensor_info is None:
                continue
            if qconfig is None:
                # If qconfig is None, we do not need any input observers
                continue
            else:
                # always add observer if the op can be quantized.
                tensor_id = tensor_info.id  # type: ignore[attr-defined]
                if seen_q_op_info.type in (
                    str(torch.nn.EmbeddingBag),
                    str(MergedEmbeddingBagWithCat),
                ):
                    obs = qconfig.activation()
                    self.weight_tensor_id_to_observer[
                        str(seen_q_op_info.idx) + "_" + str(tensor_id)
                    ] = obs
                elif not isinstance(
                    qconfig.activation(), torch.ao.quantization.PlaceholderObserver
                ):
                    if seen_q_op_info.type in [
                        str(torch.nn.ConvTranspose2d),
                        str(torch.nn.ConvTranspose3d),
                    ] and isinstance(
                        qconfig.weight(), torch.ao.quantization.PerChannelMinMaxObserver
                    ):
                        obs = qconfig.weight.with_args(ch_axis=1)()
                    else:
                        obs = qconfig.weight()
                    self.weight_tensor_id_to_observer[
                        str(seen_q_op_info.idx) + "_" + str(tensor_id)
                    ] = obs
        # LSTM, we don't know whether has bais or not, so we add observer for all them, but will not use them at convert step.
        # w_ih, w_hh share same observe, and b_ih, b_hh also share same observer
        if seen_q_op_info.type == str(torch.nn.LSTM):
            if qconfig is not None and not isinstance(
                qconfig.activation(), torch.ao.quantization.PlaceholderObserver
            ):
                for i in range(0, len(seen_q_op_info.weight_tensor_infos), 2):
                    tensor_id = seen_q_op_info.weight_tensor_infos[i].id
                    obs = qconfig.weight()
                    self.weight_tensor_id_to_observer[
                        str(seen_q_op_info.idx) + "_" + str(tensor_id)
                    ] = obs
                    self.weight_tensor_id_to_observer[
                        str(seen_q_op_info.idx) + "_" + str(tensor_id + 1)
                    ] = obs

        # SmoothQuant: Linear activation observer and weight observer should know each other
        if (
            seen_q_op_info.type == str(torch.nn.Linear)
            and qconfig is not None
            and isinstance(qconfig.activation(), SmoothQuantActivationObserver)
            and isinstance(qconfig.weight(), SmoothQuantWeightObserver)
        ):
            x_tensor_id = seen_q_op_info.input_tensor_infos[0].id
            w_tensor_id = seen_q_op_info.weight_tensor_infos[0].id
            x_obs = self.tensor_id_to_observer[str(x_tensor_id)]
            w_obs = self.weight_tensor_id_to_observer[
                str(seen_q_op_info.idx) + "_" + str(w_tensor_id)
            ]
            # Duplicate input:
            # (1) In some cases, multiple linear layers share the same activation (like QKV).
            #   - If qconfig specifies share_weight_observers=True (default), we regard these
            #     weights as a single big tensor (i.e., concat along OC axis) during
            #     calibration. So, these weights share the same per-IC observer.
            #     But weights are not actually concated for computation.
            #   - If qconfig specifies share_weight_observers=False, they use different observers.
            # (2) It is also possible that linear shares activation with some non-weighted op.
            #   In that case, x_obs.weight_obs is not set. Also check it here.
            w_id_str = str(seen_q_op_info.idx) + "_" + str(w_tensor_id)
            if not found_duplicate_input or x_obs.weight_obs is None:
                x_obs.weight_obs = {w_id_str: w_obs.ic_obs}
            else:
                # The input (activation) is shared by more than one linear layers
                if getattr(qconfig, "share_weight_observers", True):
                    # Weights of these layers share the same per-IC observer
                    assert (
                        isinstance(x_obs.weight_obs, dict)
                        and len(x_obs.weight_obs) == 1
                    )
                    w_obs.ic_obs = next(iter(x_obs.weight_obs.values()))
                else:
                    # Weights of these layers use different observers
                    x_obs.weight_obs.update({w_id_str: w_obs.ic_obs})
            # In all cases, weight observer holds a reference to activation's per-IC observer
            w_obs.act_obs = x_obs.ic_obs
            # For all linear ops, set smooth_quant_enabled to true
            # Otherwise the observers just act as normal observers
            x_obs.smooth_quant_enabled = True
            w_obs.smooth_quant_enabled = True

    def _maybe_insert_output_observers(
        self,
        seen_q_op_info: SeenQOpInfo,
        root_module: torch.nn.Module,
    ):
        # always add output observer for int8_int8_ops
        op_type = seen_q_op_info.type
        if op_type in int8_int8_ops:
            qconfig = seen_q_op_info.qconfig
            for _, tensor_info in enumerate(seen_q_op_info.output_tensor_infos):
                if tensor_info is None:
                    continue
                if qconfig is None:
                    # If qconfig is None, we do not need any input observers
                    continue
                else:
                    output_tensor_id = tensor_info.id
                    self.tensor_id_to_observer[str(output_tensor_id)] = (
                        qconfig.activation()
                    )

    def insert_observers(self, root_module: torch.nn.Module):
        for _, seen_q_op_info in self.idx_to_seen_q_op_infos.items():
            self._maybe_insert_input_observers(seen_q_op_info)
            self._maybe_insert_output_observers(seen_q_op_info, root_module)

    def get_output_observer_from_fqn(self, fqn: str) -> Optional[torch.nn.Module]:
        for _, seen_q_op_info in self.idx_to_seen_q_op_infos.items():
            if seen_q_op_info.fqn != fqn:
                continue
            output_tensor_id = seen_q_op_info.output_tensor_infos[0].id
            if str(output_tensor_id) in self.tensor_id_to_observer:
                return self.tensor_id_to_observer[str(output_tensor_id)]
        return None

    # This is a hack to enable nn.Sequential to properly work with
    # this class.
    def forward(self, x):
        raise NotImplementedError(
            "Calling AutoQuantizationState.forward is not supported"
        )
        # return x


class AutoQuantizationStateModuleDict(torch.nn.ModuleDict):
    pass


def init_model_quant_state(model, module_id_to_fqn, configure):
    # Create a list before iterating because we are adding new
    # named modules inside the loop.
    named_modules = list(model.named_modules())

    # Record module instances which are leaves or children of leaves
    leaves = set()
    for fqn, child in named_modules:
        if is_leaf(child):
            for _, child_child in child.named_modules():
                leaves.add(child_child)
    model._fqn_to_auto_quant_state_map = AutoQuantizationStateModuleDict()
    for fqn, v in named_modules:
        # fqn is the global FQN, i.e. 'foo.bar.baz'
        # v is the module instance
        #
        # we need to associate the global FQN with SeenOp
        # for modules, this is the module FQN
        # for functions, this is the parent module FQN
        module_id_to_fqn[id(v)] = fqn
        if v in leaves:
            continue
        auto_quant_state = AutoQuantizationState(fqn, configure)
        # The code below registers the auto_quant_state object
        # of the child in the module hierarchy of the parent,
        # and adds the auto_quant_state object to the child
        # with a raw __setattr__, without registering it in
        # the module hierarchy of the child.
        # This is solving the problem of both storing extra state
        # (observers) as well as not modifying the meaning of user
        # code in child modules which iterates over all module
        # children.
        #
        # This narrows down the issue of dynamically adding
        # children to only affect the top level module and not
        # the children.

        # On the parent, register this module in the FQN map
        fqn_to_use_for_key = get_fqn_valid_for_module_dict_key(fqn)
        model._fqn_to_auto_quant_state_map[fqn_to_use_for_key] = auto_quant_state
        # On the child, manually set the attribute without
        # going through the `torch.nn.Module.__setattr__`
        # function, to prevent this object from appearing in
        # the child's module hierarchy.
        object.__setattr__(v, "_auto_quant_state", auto_quant_state)
