import os
import copy
from typing import List, Dict, Tuple, Any, Optional
import torch
from torch.fx.node import map_aggregate
from torch.ao.quantization import PlaceholderObserver
from torch.quantization.qconfig import QConfig
from torch.nn.utils.rnn import PackedSequence

from ._utils import (
    get_torch_function_hook_type,
    HookType,
    get_module_hook_type,
    OpQuantizeabilityType,
    attach_op_convert_info_to_model,
    save_quant_state,
    attach_scale_zp_values_to_model,
    convert_quant_state_map_to_nodes,
    sync_pool_and_lstm_input_output_scale_zp,
    module_call_to_function_call,
    quantized_modules_has_weights,
    load_qconf_summary_to_model,
    get_fqn_valid_for_module_dict_key,
    check_model_obsever_has_run,
)
from ._quantization_state import (
    AutoQuantizationState,
    AutoQuantizationStateModuleDict,
    init_model_quant_state,
)
from ._recipe import get_default_recipe
from ._module_swap_utils import swap_child_modules
from ._qconfig import WoqWeightDtype


# AutoQuantizationState lives in parent module's _modules.
# Currently, `torch.nn.Sequential`'s forward iterates over all
# items in _modules. To avoid changing the meaning of the program, for
# now we patch the forward to ignore our quantization state.
# Note: this is a hackedy hack, before launching we should consider
# checking the fix into `torch.nn.Sequential` to avoid the patch.
def _nn_sequential_patched_forward(cls, input):
    for module in cls:
        if not isinstance(module, AutoQuantizationStateModuleDict):
            input = module(input)
    return input


def _convert_PackedSequence_to_tuple_lstm(args):
    if isinstance(args, tuple) and len(args) == 2:  # (PackedSequence, hx)
        input, batch_sizes, sorted_indices, unsorted_indices = args[0]
        args = (input, batch_sizes, sorted_indices, unsorted_indices, args[-1])
    elif isinstance(args, tuple) and len(args) == 1:  # (PackedSequence, )
        input, batch_sizes, sorted_indices, unsorted_indices = args[0]
        args = (input, batch_sizes, sorted_indices, unsorted_indices)
    else:
        AssertionError(
            False
        ), "_convert_PackedSequence_to_tuple args should be a tuple with size 2 or PackedSequence"
    return args


def _convert_tuple_to_PackedSequence_lstm(args):
    assert (
        isinstance(args, tuple) and len(args) >= 4 and len(args) <= 5
    ), "_convert_tuple_to_PackedSequence input should be a tuple(5=<size >=4)"
    if len(args) == 4:
        return (PackedSequence(*args),)
    else:
        return (PackedSequence(*args[:-1]), args[-1])


def auto_prepare(
    model: torch.nn.Module,
    configure: QConfig,
    example_inputs: Optional[Tuple[Any]],
    example_kwarg_inputs: Optional[Dict[Any, Any]],
) -> torch.nn.Module:
    def convert_to_interception_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationPrepareTensorProxy)  # type: ignore[arg-type]
        else:
            return x

    cur_module = None
    first_call = True
    module_stack: List[torch.nn.Module] = []
    # Counter for tensor IDs, will be modified inplace by quant state.
    # This is used to track tensors from output ops to input ops. For example,
    # if op_n had a tensor output with id=1, and op_n+2 had a tensor input with
    # id=1, we know that the output of op_n is the input to op_n+2. Note,
    # this is a list because it needs to incremented inplace.
    qtensor_id = [0]
    module_id_to_fqn: Dict[int, str] = {}

    # Counter for global quantizeable ops, useful for intermediate activation
    # logging.
    global_op_idx = [0]

    global_disable_torch_function_override = False

    def check_add_has_scalar_tensor_input(args):
        r"""
        This function is about check add whether has scalar(tensor) input.
        """
        nonlocal global_disable_torch_function_override
        old_global_disable_torch_function_override = (
            global_disable_torch_function_override
        )
        global_disable_torch_function_override = True
        for arg in args:
            if not isinstance(arg, torch.Tensor) or arg.dim() == 0:
                global_disable_torch_function_override = (
                    old_global_disable_torch_function_override
                )
                return True
        global_disable_torch_function_override = (
            old_global_disable_torch_function_override
        )
        return False

    class QuantizationPrepareTensorProxy(torch.Tensor):
        """
        An override of `torch.Tensor` to enable dynamic tracing for
        quantization.
        For each function with a `__torch_function__` override, this proxy does
        the following for functions which need quantization:
        1. calls `_auto_quant_state.validate_cur_op` to validate that
           the currently seen op is the same as what was recorded during tracing
        2. calls `_auto_quant_state.op_prepare_before_hook`
        3. executes the original function
        4. calls `_auto_quant_state.op_prepare_after_hook`
        5. calls `_auto_quant_state.mark_cur_op_complete` to increment
           the current op index in preparation for the next op
        Otherwise, calls the original function.
        """

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            nonlocal global_disable_torch_function_override
            if (
                # global override means disable the override here
                global_disable_torch_function_override
                or
                # to prevent printing things from going into an infinite loop
                func == torch.Tensor.__repr__
                or
                # we don't need to override getters in this framework
                func.__name__ == "__get__"
            ):
                return super().__torch_function__(func, types, args, kwargs)

            # if we are in a function, the current module is always a parent
            nonlocal cur_module
            parent_module = cur_module
            nonlocal qtensor_id
            kwargs = kwargs if kwargs else {}
            hook_type = get_torch_function_hook_type(parent_module, func)
            # Don't support torch.add(tensor, scaler)
            # case, scalar+scalar, pytorch trace will convert the first input as a tensor at convert step,
            # but we didn't collect the quant info at calibration step, which can't get
            # quant info here(asster KeyError), so we disable torch.add(tensor, scaler) quantizaiton.
            if (
                hook_type is HookType.OP_HOOKS
                and func in [torch.add, torch.Tensor.add]
                and check_add_has_scalar_tensor_input(args)
            ):
                hook_type = None

            if hook_type is HookType.OP_HOOKS:
                fqn = module_id_to_fqn[id(parent_module)] if parent_module else None
                qstate = parent_module._auto_quant_state  # type: ignore[attr-defined]
                if not first_call:
                    qstate.validate_cur_op(func)
                # run "before" hook
                if first_call:
                    args, kwargs = qstate.first_call_op_prepare_before_hook(
                        func,
                        args,
                        kwargs,
                        qtensor_id,
                        fqn,
                        parent_module,
                        OpQuantizeabilityType.QUANTIZEABLE,
                    )
                else:
                    args, kwargs = qstate.op_prepare_before_hook(func, args, kwargs)
                # forward
                output = super().__torch_function__(func, types, args, kwargs)
                # run "after" hook
                if first_call:
                    output = qstate.first_call_op_prepare_after_hook(
                        func,
                        output,
                        args,
                        qtensor_id,
                        OpQuantizeabilityType.QUANTIZEABLE,
                    )
                else:
                    output = qstate.op_prepare_after_hook(
                        func, output, args, global_op_idx
                    )
                qstate.mark_cur_op_complete(func)
            else:
                # Hook type is not HookType.OP_HOOKS, if first_call is True we
                # record the DAG of non-quantizeable ops.
                if first_call:
                    qstate = getattr(parent_module, "_auto_quant_state", None)
                    if qstate:
                        fqn = (
                            module_id_to_fqn.get(id(parent_module), None)
                            if parent_module
                            else None
                        )
                        args, kwargs = qstate.first_call_op_prepare_before_hook(
                            func,
                            args,
                            kwargs,
                            qtensor_id,
                            fqn,
                            parent_module,
                            OpQuantizeabilityType.NOT_QUANTIZEABLE,
                        )

                output = super().__torch_function__(func, types, args, kwargs)

                if first_call:
                    qstate = getattr(parent_module, "_auto_quant_state", None)
                    if qstate:
                        output = qstate.first_call_op_prepare_after_hook(
                            func,
                            output,
                            args,
                            qtensor_id,
                            OpQuantizeabilityType.NOT_QUANTIZEABLE,
                        )

            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        QuantizationPrepareTensorProxy
                    )
                assert output is not NotImplemented

            return output

        def __repr__(self):
            return f"QuantizationPrepareTensorProxy({super().__repr__()})"

        # TODO(future PR): add other math overrides

    class QuantizationInterceptionModule(type(model)):  # type: ignore[misc]
        """
        An override of user defined subclass of `nn.Module` to enable
        dynamic tracing for quantization.
        `cur_module` keeps track of the current module in the stack.
        During the fist call, an `AutoQuantizationState` object is created and
        attached to each non-leaf modules which we need to check for
        quantizeable operations.
        We override the `__call__` function to do the following for each
        module:
        If the module is an op which needs quantization:
        1. calls `_auto_quant_state.validate_cur_op` to validate that
           the currently seen op is the same as what was recorded during tracing
        2. calls parent module's `._auto_quant_state.op_prepare_before_hook`
        3. executes the original module forward
        4. calls parent module's `_auto_quant_state.op_prepare_after_hook`
        5. calls `_auto_quant_state.mark_cur_op_complete` to increment
           the current op index in preparation for the next op
        Otherwise, calls the original module forward.
        """

        def __call__(self, *args, **kwargs):
            new_args = map_aggregate(args, convert_to_interception_proxy)
            new_kwargs = map_aggregate(kwargs, convert_to_interception_proxy)
            orig_module_call = torch.nn.Module.__call__
            orig_nn_sequential_forward = torch.nn.Sequential.forward

            def _patched_module_call(self, *args, **kwargs):
                nonlocal cur_module
                old_module = cur_module
                cur_module = self
                try:
                    parent_module = module_stack[-1] if len(module_stack) else None
                    module_stack.append(self)
                    fqn = module_id_to_fqn.get(id(self), None)

                    hook_type = get_module_hook_type(parent_module, cur_module)
                    if hook_type is HookType.OP_HOOKS:
                        parent_qstate: AutoQuantizationState = (
                            parent_module._auto_quant_state
                        )  # type: ignore[union-attr, assignment]
                        # before hooks
                        if not first_call:
                            parent_qstate.validate_cur_op(cur_module)

                        # If we are in this hook, `cur_module` is a leaf module.
                        # Therefore, we do not need to override any of its
                        # children. Disabling the overrides for performance.
                        nonlocal global_disable_torch_function_override
                        old_global_disable_torch_function_override = (
                            global_disable_torch_function_override
                        )
                        global_disable_torch_function_override = True
                        is_lstm_packed_input = isinstance(
                            cur_module, torch.nn.LSTM
                        ) and isinstance(args[0], PackedSequence)
                        if is_lstm_packed_input:
                            args = _convert_PackedSequence_to_tuple_lstm(args)
                        if first_call:
                            # mypy ignore is used instead of assert because this
                            # runs on every forward and assert has a performance cost
                            (
                                args,
                                kwargs,
                            ) = parent_qstate.first_call_op_prepare_before_hook(
                                cur_module,
                                args,
                                kwargs,
                                qtensor_id,
                                fqn,
                                cur_module,  # type: ignore[arg-type]
                                OpQuantizeabilityType.QUANTIZEABLE,
                            )
                        else:
                            # mypy ignore is used instead of assert because this
                            # runs on every forward and assert has a performance cost
                            args, kwargs = parent_qstate.op_prepare_before_hook(
                                cur_module, args, kwargs
                            )  # type: ignore[arg-type]

                        if is_lstm_packed_input:
                            args = _convert_tuple_to_PackedSequence_lstm(args)

                        # original forward
                        output = orig_module_call(self, *args, **kwargs)
                        # Re-enable the overrides.
                        global_disable_torch_function_override = (
                            old_global_disable_torch_function_override
                        )

                        # after hooks
                        if is_lstm_packed_input:
                            output = _convert_PackedSequence_to_tuple_lstm(output)
                        if first_call:
                            output = parent_qstate.first_call_op_prepare_after_hook(
                                cur_module,
                                output,
                                args,
                                qtensor_id,
                                OpQuantizeabilityType.QUANTIZEABLE,
                            )
                        else:
                            output = parent_qstate.op_prepare_after_hook(
                                cur_module, output, args, global_op_idx
                            )

                        if is_lstm_packed_input:
                            output = _convert_tuple_to_PackedSequence_lstm(output)

                        parent_qstate.mark_cur_op_complete(cur_module)
                    elif hook_type is HookType.MODULE_IO_HOOKS:
                        cur_qstate = cur_module._auto_quant_state
                        cur_qstate.reset_to_new_call()
                        # original forward
                        output = orig_module_call(self, *args, **kwargs)

                        # after hooks
                        if first_call:
                            output = cur_qstate.first_call_outputs_prepare_hook(
                                output, qtensor_id
                            )
                        else:
                            output = cur_qstate.outputs_prepare_hook(output)

                        cur_qstate.validate_is_at_last_seen_idx()
                    elif hook_type is HookType.ARG_DEQUANTS:
                        if first_call and parent_module is not None:
                            parent_qstate_fc = getattr(
                                parent_module, "_auto_quant_state", None
                            )
                            if parent_qstate_fc:
                                (
                                    args,
                                    kwargs,
                                ) = parent_qstate_fc.first_call_op_prepare_before_hook(
                                    cur_module,
                                    args,
                                    kwargs,
                                    qtensor_id,
                                    fqn,
                                    cur_module,
                                    OpQuantizeabilityType.NOT_QUANTIZEABLE,
                                )

                        output = orig_module_call(self, *args, **kwargs)
                        # if this fp32 was inplace, make sure to set the output dtype
                        # back to torch.float
                        if hasattr(output, "_qtensor_info"):
                            del output._qtensor_info

                        if first_call and parent_module is not None:
                            parent_qstate_fc = getattr(
                                parent_module, "_auto_quant_state", None
                            )
                            if parent_qstate_fc:
                                output = (
                                    parent_qstate_fc.first_call_op_prepare_after_hook(
                                        cur_module,
                                        output,
                                        args,
                                        qtensor_id,
                                        OpQuantizeabilityType.NOT_QUANTIZEABLE,
                                    )
                                )
                    else:
                        output = orig_module_call(self, *args, **kwargs)

                    return output
                finally:
                    module_stack.pop()
                    cur_module = old_module

            torch.nn.Module.__call__ = _patched_module_call
            torch.nn.Sequential.forward = _nn_sequential_patched_forward  # type: ignore[assignment]
            nonlocal first_call
            try:
                if first_call:
                    init_model_quant_state(self, module_id_to_fqn, configure)

                global_op_idx[0] = 0
                output = super().__call__(*new_args, **new_kwargs)

                if first_call:
                    for _, v in self.named_modules():
                        if hasattr(v, "_auto_quant_state"):
                            v._auto_quant_state.insert_observers(v)
                return output
            finally:
                torch.nn.Module.__call__ = orig_module_call
                torch.nn.Sequential.forward = orig_nn_sequential_forward  # type: ignore[assignment]
                first_call = False

        def save_qconf_summary(self, qconf_summary):
            r"""
            This function is about save model's quant_state_map to a json file.
            """
            assert (
                qconf_summary is not None
            ), "A configure file name should be given to save the qconf_summary"
            quant_state_map = self._fqn_to_auto_quant_state_map
            # If user have given a json file, we will save the qconf_summary according to the user's setting,
            # otherwise,  we will first get a default_recipe, and then save the default_recipe's setting.
            if not hasattr(self, "_qconf_summary"):
                # compute scales and zero_point.
                attach_scale_zp_values_to_model(model)
                nodes = convert_quant_state_map_to_nodes(quant_state_map)
                # pooling and lstm's input and output should have same scale_zp.
                sync_pool_and_lstm_input_output_scale_zp(quant_state_map, nodes)
                get_default_recipe(nodes)
            else:
                if check_model_obsever_has_run(model):
                    # re-compute the scales and zp if user load a json file and re-do the calibration step.
                    attach_scale_zp_values_to_model(model)
                else:
                    # do nothing if user just loaded a json file and not re-do the calibration step
                    pass
            # Setting model qconf_summary attr which can be easily to check the whether the scale/zp has been computed.
            self._qconf_summary = qconf_summary
            save_quant_state(quant_state_map, qconf_summary)

        def load_qconf_summary(self, qconf_summary):
            r"""
            This function is about load the user qconf_summary, which will overwrite the model's quant_state_map.
            """
            if os.path.exists(qconf_summary) and os.stat(qconf_summary).st_size != 0:
                self._qconf_summary = qconf_summary
                load_qconf_summary_to_model(self, qconf_summary)
            else:
                AssertionError(
                    False,
                    ("Can not load a empty file or none existed file" + qconf_summary),
                )

    model.q_config = configure
    # For Dynamic quantization, most user model has a dynamic control flow, the DBR
    # doesn't support it now, so there skip DRB when user want to run dynamic quantization.
    if not isinstance(configure.activation(), PlaceholderObserver):
        model.__class__ = QuantizationInterceptionModule
        # init model quantization state using example_inputs
        assert example_inputs is not None or example_kwarg_inputs is not None, (
            "IPEX: example_inputs and example_kwarg_inputs cannot be None at same time "
            "for static quantization."
        )
        if example_kwarg_inputs is None:
            model(*example_inputs)
        elif example_inputs is None:
            model(**example_kwarg_inputs)
        else:
            AssertionError(
                False,
                "IPEX quantization.prepare: example_inputs and example_kwarg_inputs cannot be set at same time "
                "for static quantization.",
            )
    return model


def copy_prepared_model(model):
    copied_model = copy.deepcopy(model)
    copied_model.q_config = model.q_config
    if isinstance(copied_model.q_config.activation(), PlaceholderObserver):
        return copied_model
    copied_model._fqn_to_auto_quant_state_map = copy.deepcopy(
        model._fqn_to_auto_quant_state_map
    )
    named_modules = list(copied_model.named_modules())
    for fqn, v in named_modules:
        fqn_to_use_for_key = get_fqn_valid_for_module_dict_key(fqn)
        if fqn_to_use_for_key in copied_model._fqn_to_auto_quant_state_map:
            auto_quant_state = copied_model._fqn_to_auto_quant_state_map[
                fqn_to_use_for_key
            ]
            object.__setattr__(v, "_auto_quant_state", auto_quant_state)
    if hasattr(model, "_qconf_summary"):
        copied_model._qconf_summary = copy.deepcopy(model._qconf_summary)
    copied_model.__class__ = model.__class__
    return copied_model


def auto_convert(
    module: torch.nn.Module,
) -> torch.nn.Module:
    def convert_to_dispatch_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationConvertTensorProxy)  # type: ignore[arg-type]
        else:
            return x

    global_disable_torch_function_override = False

    def check_add_has_scalar_tensor_input(args):
        r"""
        This function is about check add whether has scalar(tensor) input.
        """
        nonlocal global_disable_torch_function_override
        old_global_disable_torch_function_override = (
            global_disable_torch_function_override
        )
        global_disable_torch_function_override = True
        for arg in args:
            if not isinstance(arg, torch.Tensor) or arg.dim() == 0:
                global_disable_torch_function_override = (
                    old_global_disable_torch_function_override
                )
                return True
        global_disable_torch_function_override = (
            old_global_disable_torch_function_override
        )
        return False

    class QuantizationConvertTensorProxy(torch.Tensor):
        """
        An override of `torch.Tensor` to enable dynamic dispatch for
        quantization inference.
        For each function with a `__torch_fuction__` override, this proxy does
        the following for functions which need quantization:
        1. calls `_auto_quant_state.validate_cur_op` to validate that
           the currently seen op is the same as what was recorded during tracing
        2. calls `_auto_quant_state.op_convert_before_hook`.
        3. executes the function, with target, args and kwargs possibly modified
           by (2)
        4. calls `_auto_quant_state.inference_function_after_hook`.
        5. calls `_auto_quant_state.mark_cur_op_complete` to increment
           the current op index in preparation for the next op
        Otherwise, calls the original function.
        """

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            nonlocal global_disable_torch_function_override
            if (
                # global override means disable the override here
                global_disable_torch_function_override
                or
                # to prevent printing things from going into an infinite loop
                func == torch.Tensor.__repr__
                or
                # we don't need to override getters in this framework
                func.__name__ == "__get__"
            ):
                return super().__torch_function__(func, types, args, kwargs)

            kwargs = kwargs if kwargs else {}
            # if we are in a function, the current module is always a parent
            parent_module = cur_module
            hook_type = get_torch_function_hook_type(parent_module, func)
            # Don't support torch.add(tensor, scaler)
            # case, scalar+scalar, pytorch trace will convert the first input as a tensor,
            # but we didn't collect the quant info at calibration step, which can't get
            # quant info here(asster KeyError), so we disable torch.add(tensor, scaler) quantizaiton.
            if (
                hook_type is HookType.OP_HOOKS
                and func in [torch.add, torch.Tensor.add]
                and check_add_has_scalar_tensor_input(args)
            ):
                hook_type = None

            if hook_type is HookType.OP_HOOKS:
                qstate: AutoQuantizationState = parent_module._auto_quant_state  # type: ignore[union-attr]
                # before hooks
                qstate.validate_cur_op(func)
                func, args, kwargs = qstate.op_convert_before_hook(
                    func, args, kwargs, parent_module
                )  # type: ignore[arg-type]

                # forward
                output = super().__torch_function__(func, types, args, kwargs)

                # after hooks
                output = qstate.op_convert_after_hook(func, output)
                qstate.mark_cur_op_complete(func)
            else:  # HookType.NONE
                output = super().__torch_function__(func, types, args, kwargs)

            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        QuantizationConvertTensorProxy
                    )
                assert output is not NotImplemented
            return output

        def __repr__(self):
            return f"QuantizationConvertTensorProxy({super().__repr__()})"

    cur_module = None
    module_stack: List[torch.nn.Module] = []

    assert len(module.__class__.__bases__) == 1

    class QuantizationDispatchModule(module.__class__.__bases__[0]):  # type: ignore[name-defined]
        """
        An override of user defined subclass of `nn.Module` to enable
        dynamic tracing for quantization, after model conversion
        to quantized domain.
        `cur_module` keeps track of the current module in the stack.
        Tensor arguments are converted to `QuantizationConvertTensorProxy`.
        We override the `__call__` function to do the following for each
        module:
        If the module is an op which needs quantization:
        1. calls `_auto_quant_state.validate_cur_op` to validate that
           the currently seen op is the same as what was recorded during tracing
        2. calls parent module's `._auto_quant_state.op_convert_before_hook`
        3. executes the original module forward
        4. calls parent module's `_auto_quant_state.op_convert_after_hook`
        5. calls `_auto_quant_state.mark_cur_op_complete` to increment
           the current op index in preparation for the next op
        Otherwise, calls the original module forward.
        """

        def __call__(self, *args, **kwargs):
            new_args = map_aggregate(args, convert_to_dispatch_proxy)
            new_kwargs = map_aggregate(kwargs, convert_to_dispatch_proxy)
            orig_module_call = torch.nn.Module.__call__
            orig_nn_sequential_forward = torch.nn.Sequential.forward

            def _patched_module_call(self, *args, **kwargs):
                nonlocal cur_module
                old_module = cur_module
                cur_module = self
                nonlocal global_disable_torch_function_override
                try:
                    parent_module = module_stack[-1] if len(module_stack) else None
                    module_stack.append(self)
                    hook_type = get_module_hook_type(parent_module, cur_module)
                    if hook_type is HookType.OP_HOOKS:
                        # before hooks
                        qstate: AutoQuantizationState = (
                            parent_module._auto_quant_state
                        )  # type: ignore[union-attr, assignment]
                        qstate.validate_cur_op(cur_module)

                        # If we are in this hook, `cur_module` is a leaf module.
                        # Therefore, we do not need to override any of its
                        # children. Disabling the overrides for performance.
                        old_global_disable_torch_function_override = (
                            global_disable_torch_function_override
                        )
                        global_disable_torch_function_override = True
                        is_lstm_packed_input = isinstance(
                            cur_module, torch.nn.LSTM
                        ) and isinstance(args[0], PackedSequence)
                        if is_lstm_packed_input:
                            args = _convert_PackedSequence_to_tuple_lstm(args)
                        _, args, kwargs = qstate.op_convert_before_hook(
                            cur_module, args, kwargs, cur_module
                        )
                        if is_lstm_packed_input:
                            args = _convert_tuple_to_PackedSequence_lstm(args)
                        if type(cur_module) in quantized_modules_has_weights:
                            weights = qstate.op_weight_convert_before_hook(cur_module)
                            output = module_call_to_function_call(self, args, weights)
                        else:
                            output = orig_module_call(self, *args, **kwargs)
                        # after hooks
                        if is_lstm_packed_input:
                            output = _convert_PackedSequence_to_tuple_lstm(output)
                        output = qstate.op_convert_after_hook(cur_module, output)
                        if is_lstm_packed_input:
                            output = _convert_tuple_to_PackedSequence_lstm(output)
                        # Re-enable the override.
                        global_disable_torch_function_override = (
                            old_global_disable_torch_function_override
                        )

                        qstate.mark_cur_op_complete(cur_module)
                    elif hook_type is HookType.MODULE_IO_HOOKS:
                        cur_qstate: AutoQuantizationState = cur_module._auto_quant_state
                        cur_qstate.reset_to_new_call()
                        # before hooks (TODO)
                        # forward
                        output = orig_module_call(self, *args, **kwargs)
                        # after hooks
                        # For the sake of performance, we assume no overrides
                        # are needed for quantizing/dequantizing things
                        old_global_disable_torch_function_override = (
                            global_disable_torch_function_override
                        )
                        global_disable_torch_function_override = True

                        output = cur_qstate.outputs_convert_hook(output)
                        global_disable_torch_function_override = (
                            old_global_disable_torch_function_override
                        )
                        cur_qstate.validate_is_at_last_seen_idx()
                    else:
                        output = orig_module_call(self, *args, **kwargs)
                    return output
                finally:
                    module_stack.pop()
                    cur_module = old_module

            torch.nn.Module.__call__ = _patched_module_call
            torch.nn.Sequential.forward = _nn_sequential_patched_forward  # type: ignore[assignment]

            try:
                output = super().__call__(*new_args, **new_kwargs)

                def unwrap_proxy(a):
                    if isinstance(a, QuantizationConvertTensorProxy):
                        a.__class__ = torch.Tensor  # type: ignore[assignment]
                    return a

                output = map_aggregate(output, unwrap_proxy)
                return output
            finally:
                torch.nn.Module.__call__ = orig_module_call
                torch.nn.Sequential.forward = orig_nn_sequential_forward  # type: ignore[assignment]

    # If module doesn't have a configure_file attr, we can say that user didn't run save_qconf_summary method which have
    # computed the scales and zp, or didn't use the user's setting from a given json file(load_qconf_summary), we need to compute
    # the scale and zp here.
    if not hasattr(module, "_qconf_summary"):
        quant_state_map = module._fqn_to_auto_quant_state_map
        # compute scales and zero_point.
        attach_scale_zp_values_to_model(module)
        nodes = convert_quant_state_map_to_nodes(quant_state_map)
        # pooling and lstm's input and output should have same scale_zp.
        sync_pool_and_lstm_input_output_scale_zp(quant_state_map, nodes)
        get_default_recipe(nodes)
    else:
        if check_model_obsever_has_run(module):
            # re-compute the scales and zp if user load a json file and re-do the calibration step.
            attach_scale_zp_values_to_model(module)
        else:
            # clear observer if module have, this will works when the user's json setting is loaded
            # and not re-do the calibration step.
            for _, v in module._fqn_to_auto_quant_state_map.items():
                v.tensor_id_to_observer.clear()
                v.weight_tensor_id_to_observer.clear()

    # Attach quant_info to parent each module
    attach_op_convert_info_to_model(module)
    swap_child_modules(module)
    module.__class__ = QuantizationDispatchModule
    return module


NF4_QUANT_TABLE = [
    -1.0 - 1e-2,  # 0b0000
    -0.8480964004993439,  # 0b0001
    -0.6106329262256622,  # 0b0010
    -0.4599952697753906,  # 0b0011
    -0.33967943489551544,  # 0b0100
    -0.23460740596055984,  # 0b0101
    -0.13791173323988914,  # 0b0110
    -0.045525018125772476,  # 0b0111
    0.03979014977812767,  # 0b1000
    0.1202552504837513,  # 0b1001
    0.2035212516784668,  # 0b1010
    0.2920137718319893,  # 0b1011
    0.3893125355243683,  # 0b1100
    0.5016634166240692,  # 0b1101
    0.6427869200706482,  # 0b1110
    0.8614784181118011,  # 0b1111
]


NF4_DEQUANT_TABLE = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def map_float_tensor_to_nf4(t, dtype=torch.uint8):
    # Map [-1, 1] to nf4
    # Assume t in [-1, 1]
    out_uint8 = torch.empty(t.shape, dtype=dtype)
    for i in range(len(NF4_QUANT_TABLE)):
        out_uint8[t > NF4_QUANT_TABLE[i]] = i
    return out_uint8


def map_nf4_tensor_to_float(t, dtype=torch.float32):
    # Map nf4 to [-1, 1]
    out_dq = torch.empty(t.shape).to(dtype)
    for i in range(len(NF4_DEQUANT_TABLE)):
        out_dq[t == i] = NF4_DEQUANT_TABLE[i]
    return out_dq


def is_4bit(dtype):
    return dtype in (WoqWeightDtype.INT4, WoqWeightDtype.NF4)


def is_sym_quant(dtype):
    return dtype in (WoqWeightDtype.NF4,)


def quantize_per_channel(
    t: torch.Tensor, dtype, scales=None, zero_points=None, sym_quant=False
):
    r"""
    Quantize a weight tensor of Linear modules per channel.
    Assume the tensor shape is [output channel, input channel],
    each output channel has its own quantization parameters.

    Args:
        input: The tensor to be quantized
        dtype: data type of the quantized tensor, int8, int4 or nf4
        scales: Scales for quantization. If None, find by min/max.
        zero_points: zero points for quantization. If None, find by min/max.
        sym_quant: symmetric or asymmetric quantization

    Returns:
        A tuple of
        - The quantized tensor
        - Scales
        - Zero points
    """
    assert t.ndim == 2
    assert dtype in (WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4)
    assert not (
        dtype == WoqWeightDtype.NF4 and not sym_quant
    ), "NF4 must be symmetric quant"

    def get_qparams(scales, zps):
        if scales is not None and (sym_quant or zps is not None):
            return scales, zps
        eps = torch.tensor([torch.finfo(torch.float32).eps])
        zeros = torch.zeros(t.shape[0], dtype=t.dtype, device=t.device)
        mins = torch.minimum(t.min(dim=1)[0], zeros)
        maxs = torch.maximum(t.max(dim=1)[0], zeros)
        zps = None
        if dtype == WoqWeightDtype.INT8:
            if sym_quant:
                scales = torch.maximum(torch.abs(maxs), torch.abs(mins)) / 127
                scales = torch.max(scales, eps)
            else:
                scales = (maxs - mins) / 255
                scales = torch.max(scales, eps)
                zps = -torch.round(mins / scales)
                zps -= 128
        elif dtype == WoqWeightDtype.INT4:
            if sym_quant:
                scales = torch.maximum(torch.abs(maxs), torch.abs(mins)) / 7
                scales = torch.max(scales, eps)
            else:
                scales = (maxs - mins) / 15
                scales = torch.max(scales, eps)
                zps = -torch.round(mins / scales)
        else:  # NF4
            scales = torch.maximum(torch.abs(maxs), torch.abs(mins))
            scales = torch.max(scales, eps)
        return scales, zps

    scales, zps = get_qparams(scales, zero_points)
    inv_scales = 1 / scales.unsqueeze(1)
    if dtype == WoqWeightDtype.INT8:
        qmin = -128
        qmax = 127
        qt = torch.clamp(
            torch.round(t * inv_scales) + (zps.unsqueeze(1) if zps is not None else 0),
            min=qmin,
            max=qmax,
        ).to(torch.int8)
    elif dtype == WoqWeightDtype.INT4:
        qmin = 0
        qmax = 15
        # for sym_quant, shift to 0-15 for storage
        qt = torch.clamp(
            torch.round(t * inv_scales) + (zps.unsqueeze(1) if not sym_quant else 8),
            min=qmin,
            max=qmax,
        ).to(torch.uint8)
    else:  # NF4
        qt = map_float_tensor_to_nf4(t * inv_scales)
    if is_4bit(dtype):
        if qt.size(-1) % 2:
            qt = torch.nn.functional.pad(qt, (0, 1), value=0)
        qt = qt[:, 1::2].bitwise_left_shift(4).bitwise_or_(qt[:, ::2].bitwise_and(0xF))
    return qt.contiguous(), scales, zps


def dequantize_per_channel(
    qt: torch.Tensor,
    scales: torch.Tensor,
    zps: Optional[torch.Tensor],
    dtype,
    weight_shape=None,
):
    r"""
    Dequantize a weight tensor of Linear modules per channel.
    Assume the tensor shape is [output channel, input channel],
    each output channel has its own quantization parameters.

    Args:
        qt: The tensor to be dequantized
        scales: Scales for dequantization
        zps: Zero points for dequantization
        dtype: data type of the quantized tensor, int8, int4 or nf4
        weight_shape: True weight shape. INT4 tensor's input channel may
            be padded to even, so we need this to return the correct weight.

    Returns:
        The dequantized tensor
    """
    assert qt.ndim == 2
    assert dtype in (WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4)
    scales = scales.squeeze()
    sym_quant = zps is None
    if sym_quant:
        zps = torch.zeros_like(scales)
        if dtype == WoqWeightDtype.INT4:
            # shift from [0, 15] to [-8, 7]
            zps += 8
    else:
        zps = zps.squeeze()
    if dtype == WoqWeightDtype.INT8:
        return (qt.to(torch.float) - zps.unsqueeze(-1)) * scales.unsqueeze(-1)
    elif dtype == WoqWeightDtype.INT4:
        t = torch.empty(
            qt.shape[0], qt.shape[1] * 2, dtype=torch.uint8, device=qt.device
        )
        t[:, ::2] = qt.bitwise_and(0xF)
        t[:, 1::2] = qt.bitwise_right_shift(4)
        t = (t.to(torch.float) - zps.unsqueeze(-1)) * scales.unsqueeze(-1)
        if weight_shape is not None:
            t = t[: weight_shape[0], : weight_shape[1]].contiguous()
        return t
    else:  # NF4
        t = torch.empty(
            qt.shape[0], qt.shape[1] * 2, dtype=torch.uint8, device=qt.device
        )
        t[:, ::2] = qt.bitwise_and(0xF)
        t[:, 1::2] = qt.bitwise_right_shift(4)
        t = map_nf4_tensor_to_float(t)
        if weight_shape is not None:
            t = t[: weight_shape[0], : weight_shape[1]].contiguous()
        t = t * scales.unsqueeze(-1)
        return t


def quantize_per_block(
    input: torch.Tensor,
    dtype,
    group_size,
    scales=None,
    zero_points=None,
    sym_quant=False,
):
    r"""
    Quantize a weight tensor of Linear modules per block.
    Assume the tensor shape is [output channel, input channel],
    block shape is [1, group_size].

    Args:
        input: The tensor to be quantized
        dtype: data type of the quantized tensor, int8, int4 or nf4
        group_size: Size of group along input channel
        scales: Scales for quantization. If None, find by min/max.
        zero_points: zero points for quantization. If None, find by min/max.
        sym_quant: symmetric or asymmetric quantization

    Returns:
        A tuple of
        - The quantized tensor
        - Scales in shape [N, #block_k]
        - Zero points in shape [N, #block_k]
    """
    assert (
        input.dim() == 2
    ), f"{__name__}: Expect input has 2 dimensions but got {input.dim()}"
    assert group_size > 0, f"{__name__}: Expect group_size > 0 but got {group_size}"
    assert dtype in (WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4)
    assert not (
        dtype == WoqWeightDtype.NF4 and not sym_quant
    ), "NF4 must be symmetric quant"
    N = input.size(0)
    K = input.size(1)
    k_rem = K % group_size
    has_rem = k_rem != 0

    def get_qparams(scales, zps):
        if scales is not None and (sym_quant or zps is not None):
            return scales, zps
        eps = torch.tensor([torch.finfo(torch.float32).eps])
        t_com = input[:, : K - k_rem].view(N, K // group_size, group_size)
        mins = torch.minimum(t_com.min(dim=-1)[0], torch.tensor([0]))
        maxs = torch.maximum(t_com.max(dim=-1)[0], torch.tensor([0]))
        zps = None
        if dtype == WoqWeightDtype.INT8:
            if sym_quant:
                scales = torch.maximum(torch.abs(maxs), torch.abs(mins)) / 127
                scales = torch.max(scales, eps)
            else:
                scales = (maxs - mins) / 255
                scales = torch.max(scales, eps)
                zps = -torch.round(mins / scales)
                zps -= 128
        elif dtype == WoqWeightDtype.INT4:
            if sym_quant:
                scales = torch.maximum(torch.abs(maxs), torch.abs(mins)) / 7
                scales = torch.max(scales, eps)
            else:
                scales = (maxs - mins) / 15
                scales = torch.max(scales, eps)
                zps = -torch.round(mins / scales)
        else:  # NF4
            scales = torch.maximum(torch.abs(maxs), torch.abs(mins))
            scales = torch.max(scales, eps)
        if k_rem != 0:
            t_rem = input[:, K - k_rem :].view(N, 1, k_rem)
            mins_rem = torch.minimum(t_rem.min(dim=-1)[0], torch.tensor([0]))
            maxs_rem = torch.maximum(t_rem.max(dim=-1)[0], torch.tensor([0]))
            zps_rem = None
            if dtype == WoqWeightDtype.INT8:
                if sym_quant:
                    scales_rem = (
                        torch.maximum(torch.abs(maxs_rem), torch.abs(mins_rem)) / 127
                    )
                    scales_rem = torch.max(scales_rem, eps)
                else:
                    scales_rem = (maxs_rem - mins_rem) / 255
                    scales_rem = torch.max(scales_rem, eps)
                    zps_rem = -torch.round(mins_rem / scales_rem)
                    zps_rem -= 128
            elif dtype == WoqWeightDtype.INT4:
                if sym_quant:
                    scales_rem = (
                        torch.maximum(torch.abs(maxs_rem), torch.abs(mins_rem)) / 7
                    )
                    scales_rem = torch.max(scales_rem, eps)
                else:
                    scales_rem = (maxs_rem - mins_rem) / 15
                    scales_rem = torch.max(scales_rem, eps)
                    zps_rem = -torch.round(mins_rem / scales_rem)
            else:  # NF4
                scales_rem = torch.maximum(torch.abs(maxs_rem), torch.abs(mins_rem))
                scales_rem = torch.max(scales_rem, eps)
            scales = torch.cat([scales, scales_rem], dim=-1)
            if not sym_quant:
                assert zps is not None and zps_rem is not None
                zps = torch.cat([zps, zps_rem], dim=-1)
        return scales, zps

    scales, zps = get_qparams(scales, zero_points)
    Kc = (K + group_size - 1) // group_size
    t_com = input[:, : K - k_rem].view(N, K // group_size, group_size)
    scales_com = scales[:, : Kc - has_rem]
    inv_scales_com = 1 / scales_com.unsqueeze(-1)
    if not sym_quant:
        assert zps is not None
        zps_com = zps[:, : Kc - has_rem]
    else:
        zps_com = None
    if dtype == WoqWeightDtype.INT8:
        qmin = -128
        qmax = 127
        qt = torch.clamp(
            torch.round(t_com * inv_scales_com)
            + (zps_com.unsqueeze(-1) if zps_com is not None else 0),
            min=qmin,
            max=qmax,
        )
    elif dtype == WoqWeightDtype.INT4:
        qmin = 0
        qmax = 15
        # for sym_quant, shift to 0-15 for storage
        qt = torch.clamp(
            torch.round(t_com * inv_scales_com)
            + (zps_com.unsqueeze(-1) if zps_com is not None else 8),
            min=qmin,
            max=qmax,
        )
    else:  # NF4
        qt = map_float_tensor_to_nf4(t_com * inv_scales_com)
    qt = qt.view(N, K // group_size * group_size)
    if k_rem != 0:
        t_rem = input[:, K - k_rem :].view(N, 1, k_rem)
        scales_rem = scales[:, Kc - has_rem :]
        inv_scales_rem = 1 / scales_rem.unsqueeze(-1)
        if not sym_quant:
            assert zps is not None
            zps_rem = zps[:, Kc - has_rem :]
        else:
            zps_rem = None
        if dtype == WoqWeightDtype.INT8:
            qt_rem = torch.clamp(
                torch.round(t_rem * inv_scales_rem)
                + (zps_rem.unsqueeze(-1) if zps_rem is not None else 0),
                min=qmin,
                max=qmax,
            )
        elif dtype == WoqWeightDtype.INT4:
            # for sym_quant, shift to 0-15 for storage
            qt_rem = torch.clamp(
                torch.round(t_rem * inv_scales_rem)
                + (zps_rem.unsqueeze(-1) if zps_rem is not None else 8),
                min=qmin,
                max=qmax,
            )
        else:  # NF4
            qt_rem = map_float_tensor_to_nf4(t_rem * inv_scales_rem)
        qt_rem = qt_rem.view(N, k_rem)
        qt = torch.cat([qt, qt_rem], dim=1).contiguous()
    # INT8 weight: always store in int8
    # INT4 weight: store in int8 if sym_quant, otherwise in uint8
    # NF4 weight: always store in uint8
    qt = qt.to(torch.uint8 if is_4bit(dtype) else torch.int8)
    qt = qt.view(N, K)
    if is_4bit(dtype):
        if qt.size(-1) % 2:
            qt = torch.nn.functional.pad(qt, (0, 1), value=0)
        qt = qt[:, 1::2].bitwise_left_shift(4).bitwise_or_(qt[:, ::2])
    return qt.contiguous(), scales, zps


def dequantize_per_block(
    qt: torch.Tensor,
    scales: torch.Tensor,
    zps: Optional[torch.Tensor],
    dtype,
    group_size,
    weight_shape=None,
):
    r"""
    Dequantize a weight tensor of Linear modules per block.
    Assume the tensor shape is [output channel, input channel],
    block shape is [1, group_size].

    Args:
        qt: The tensor to be dequantized
        scales: Scales in shape [N, #block_k]
        zps: Zero points in shape [N, #block_k]
        dtype: data type of the quantized tensor, int8, int4 or nf4
        group_size: Size of group along input channel
        block_oc: Block size of output channel, should be the same for weight packing

    Returns:
        The dequantized tensor
    """
    N = qt.size(0)
    K = qt.size(1) * 2 if is_4bit(dtype) else qt.size(1)
    if scales.dim() > 2:
        scales = scales.squeeze()
    if zps is None:
        zps = torch.zeros_like(scales)
        if dtype == WoqWeightDtype.INT4:
            # shift from [0, 15] to [-8, 7]
            zps += 8
    if zps.dim() > 2:
        zps = zps.squeeze()
    if is_4bit(dtype):
        t = torch.empty(
            qt.shape[0], qt.shape[1] * 2, dtype=torch.uint8, device=qt.device
        )
        t[:, ::2] = qt.bitwise_and(0xF)
        t[:, 1::2] = qt.bitwise_right_shift(4)
        qt = t
    k_rem = K % group_size
    has_rem = k_rem != 0
    Kc = (K + group_size - 1) // group_size
    qt_com = qt[:, : K - k_rem].view(N, K // group_size, group_size)
    scales_com = scales[:, : Kc - has_rem]
    if dtype == WoqWeightDtype.NF4:
        t = (
            (map_nf4_tensor_to_float(qt_com) * scales_com.unsqueeze(-1))
            .view(N, K - k_rem)
            .contiguous()
        )
    else:
        zps_com = zps[:, : Kc - has_rem]
        t = (
            (
                (qt_com.to(torch.float) - zps_com.unsqueeze(-1))
                * scales_com.unsqueeze(-1)
            )
            .view(N, K - k_rem)
            .contiguous()
        )
    if k_rem:
        qt_rem = qt[:, K - k_rem :].view(N, 1, k_rem)
        scales_rem = scales[:, Kc - has_rem :]
        if dtype == WoqWeightDtype.NF4:
            t_rem = (
                (map_nf4_tensor_to_float(qt_rem) * scales_rem.unsqueeze(-1))
                .view(N, k_rem)
                .contiguous()
            )
        else:
            zps_rem = zps[:, Kc - has_rem :]
            t_rem = (
                (
                    (qt_rem.to(torch.float) - zps_rem.unsqueeze(-1))
                    * scales_rem.unsqueeze(-1)
                )
                .view(N, k_rem)
                .contiguous()
            )
        t = torch.cat([t, t_rem], dim=1).contiguous()
    if weight_shape is not None:
        t = t[: weight_shape[0], : weight_shape[1]].contiguous()
    return t
