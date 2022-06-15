import os
import copy
from typing import List, Dict, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from torch.fx.node import map_aggregate
from torch.ao.quantization import PlaceholderObserver
from torch.quantization.qconfig import QConfig

from ._utils import get_torch_function_hook_type, HookType, get_module_hook_type, OpQuantizeabilityType, \
    attach_op_convert_info_to_model, save_quant_state, attach_scale_zp_values_to_model, convert_quant_state_map_to_nodes, \
        sync_pool_and_lstm_input_output_scale_zp, module_call_to_function_call, quantized_modules_has_weights, \
        load_qconf_summary_to_model, get_fqn_valid_for_module_dict_key
from ._quantization_state import AutoQuantizationState, AutoQuantizationStateModuleDict, init_model_quant_state
from ._recipe import get_default_recipe
from ._module_swap_utils import swap_child_modules

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

def _check_add_has_scalar_input(args):
    r"""
    This function is about check add whether has scalar input.
    """
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            return True
    return False

def auto_prepare(
    model : torch.nn.Module,
    configure: QConfig,
    example_inputs: Tuple[Any],
) -> torch.nn.Module:

    def convert_to_interception_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationPrepareTensorProxy)  # type: ignore[arg-type]
        else:
            return x

    cur_module = None
    first_call = True
    module_stack : List[torch.nn.Module] = []
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
                global_disable_torch_function_override or
                # to prevent printing things from going into an infinite loop
                func == torch.Tensor.__repr__ or
                # we don't need to override getters in this framework
                func.__name__ == '__get__'
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
            if hook_type is HookType.OP_HOOKS and func in [torch.add, torch.Tensor.add] and _check_add_has_scalar_input(args):
                hook_type = None

            if hook_type is HookType.OP_HOOKS:
                fqn = module_id_to_fqn[id(parent_module)] if parent_module else None
                qstate = parent_module._auto_quant_state  # type: ignore[attr-defined]
                if not first_call:
                    qstate.validate_cur_op(func)
                # run "before" hook
                if first_call:
                    args, kwargs = qstate.first_call_op_prepare_before_hook(
                        func, args, kwargs, qtensor_id, fqn, parent_module, OpQuantizeabilityType.QUANTIZEABLE)
                else:
                    args, kwargs = qstate.op_prepare_before_hook(
                        func, args, kwargs)
                # forward
                output = super().__torch_function__(func, types, args, kwargs)
                # run "after" hook
                if first_call:
                    output = qstate.first_call_op_prepare_after_hook(
                        func, output, args, qtensor_id, OpQuantizeabilityType.QUANTIZEABLE)
                else:
                    output = qstate.op_prepare_after_hook(
                        func, output, args, global_op_idx)
                qstate.mark_cur_op_complete(func)
            else:
                # Hook type is not HookType.OP_HOOKS, if first_call is True we
                # record the DAG of non-quantizeable ops.
                if first_call:
                    qstate = getattr(parent_module, '_auto_quant_state', None)
                    if qstate:
                        fqn = module_id_to_fqn.get(id(parent_module), None) \
                            if parent_module else None
                        args, kwargs = qstate.first_call_op_prepare_before_hook(
                            func, args, kwargs, qtensor_id, fqn, parent_module, OpQuantizeabilityType.NOT_QUANTIZEABLE)

                output = super().__torch_function__(func, types, args, kwargs)

                if first_call:
                    qstate = getattr(parent_module, '_auto_quant_state', None)
                    if qstate:
                        output = qstate.first_call_op_prepare_after_hook(
                            func, output, args, qtensor_id, OpQuantizeabilityType.NOT_QUANTIZEABLE)

            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        QuantizationPrepareTensorProxy)
                assert output is not NotImplemented

            return output

        def __repr__(self):
            return f'QuantizationPrepareTensorProxy({super().__repr__()})'

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
                        parent_qstate: AutoQuantizationState = \
                            parent_module._auto_quant_state  # type: ignore[union-attr, assignment]
                        # before hooks
                        if not first_call:
                            parent_qstate.validate_cur_op(cur_module)

                        # If we are in this hook, `cur_module` is a leaf module.
                        # Therefore, we do not need to override any of its
                        # children. Disabling the overrides for performance.
                        nonlocal global_disable_torch_function_override
                        old_global_disable_torch_function_override = \
                            global_disable_torch_function_override
                        global_disable_torch_function_override = True

                        if first_call:
                            # mypy ignore is used instead of assert because this
                            # runs on every forward and assert has a performance cost
                            args, kwargs = parent_qstate.first_call_op_prepare_before_hook(
                                cur_module, args, kwargs, qtensor_id,
                                fqn, cur_module, # type: ignore[arg-type]
                                OpQuantizeabilityType.QUANTIZEABLE)
                        else:
                            # mypy ignore is used instead of assert because this
                            # runs on every forward and assert has a performance cost
                            args, kwargs = parent_qstate.op_prepare_before_hook(
                                cur_module, args, kwargs)  # type: ignore[arg-type]

                        # original forward
                        output = orig_module_call(self, *args, **kwargs)
                        # Re-enable the overrides.
                        global_disable_torch_function_override = \
                            old_global_disable_torch_function_override

                        # after hooks
                        if first_call:
                            output = parent_qstate.first_call_op_prepare_after_hook(
                                cur_module, output, args, qtensor_id, OpQuantizeabilityType.QUANTIZEABLE)
                        else:
                            output = parent_qstate.op_prepare_after_hook(
                                cur_module, output, args, global_op_idx)
                        parent_qstate.mark_cur_op_complete(cur_module)
                    elif hook_type is HookType.MODULE_IO_HOOKS:
                        cur_qstate = cur_module._auto_quant_state
                        cur_qstate.reset_to_new_call()
                        # original forward
                        output = orig_module_call(self, *args, **kwargs)

                        # after hooks
                        if first_call:
                            output = cur_qstate.first_call_outputs_prepare_hook(
                                output, qtensor_id)
                        else:
                            output = cur_qstate.outputs_prepare_hook(output)

                        cur_qstate.validate_is_at_last_seen_idx()
                    elif hook_type is HookType.ARG_DEQUANTS:
                        if first_call and parent_module is not None:
                            parent_qstate_fc = getattr(
                                parent_module, '_auto_quant_state', None)
                            if parent_qstate_fc:
                                args, kwargs = \
                                    parent_qstate_fc.first_call_op_prepare_before_hook(
                                        cur_module, args, kwargs, qtensor_id, fqn,
                                        cur_module,
                                        OpQuantizeabilityType.NOT_QUANTIZEABLE)

                        output = orig_module_call(self, *args, **kwargs)
                        # if this fp32 was inplace, make sure to set the output dtype
                        # back to torch.float
                        if hasattr(output, '_qtensor_info'):
                            del output._qtensor_info

                        if first_call and parent_module is not None:
                            parent_qstate_fc = getattr(
                                parent_module, '_auto_quant_state', None)
                            if parent_qstate_fc:
                                output = \
                                    parent_qstate_fc.first_call_op_prepare_after_hook(
                                        cur_module, output, args, qtensor_id,
                                        OpQuantizeabilityType.NOT_QUANTIZEABLE)
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
                        if hasattr(v, '_auto_quant_state'):
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
            assert qconf_summary is not None, "A configure file name should be given to save the qconf_summary"
            quant_state_map = self._fqn_to_auto_quant_state_map
            # If user have given a json file, we will save the qconf_summary according to the user's setting,
            # otherwise,  we will first get a default_recipe, and then save the default_recipe's setting.
            if not hasattr(self, '_qconf_summary'):
                # compute scales and zero_point.
                attach_scale_zp_values_to_model(model)
                nodes = convert_quant_state_map_to_nodes(quant_state_map)
                # pooling and lstm's input and output should have same scale_zp.
                sync_pool_and_lstm_input_output_scale_zp(quant_state_map, nodes)
                get_default_recipe(nodes)
            # Setting model qconf_summary attr which can be easily to check the whether the scale/zp has been computed.
            self._qconf_summary = qconf_summary
            save_quant_state(quant_state_map, qconf_summary)
        
        def load_qconf_summary(self, qconf_summary):
            r"""
            This function is about load the user qconf_summary, which will overwrite the model's quant_state_map.
            """
            if (os.path.exists(qconf_summary) and os.stat(qconf_summary).st_size != 0):
                self._qconf_summary = qconf_summary
                load_qconf_summary_to_model(self, qconf_summary)
            else:
                assert False, "Can not load a empty file or none existed file" + qconf_summary

    model.q_config = configure
    # For Dynamic quantization, most user model has a dynamic control flow, the DBR
    # doesn't support it now, so there skip DRB when user want to run dynamic quantization. 
    if not isinstance(configure.activation(), PlaceholderObserver):
        model.__class__ = QuantizationInterceptionModule
        # init model quantization state using example_inputs
        model(*example_inputs)
    return model

def copy_prepared_model(model):
    copied_model = copy.deepcopy(model)
    copied_model.q_config = model.q_config
    if isinstance(copied_model.q_config.activation(), PlaceholderObserver):
        return copied_model
    copied_model._fqn_to_auto_quant_state_map = copy.deepcopy(model._fqn_to_auto_quant_state_map)
    named_modules = list(copied_model.named_modules())
    for fqn, v in named_modules:
        fqn_to_use_for_key = get_fqn_valid_for_module_dict_key(fqn)
        if fqn_to_use_for_key in copied_model._fqn_to_auto_quant_state_map:
            auto_quant_state = copied_model._fqn_to_auto_quant_state_map[fqn_to_use_for_key]
            object.__setattr__(v, '_auto_quant_state', auto_quant_state)
    if hasattr(model, '_qconf_summary'):
        copied_model._qconf_summary = copy.deepcopy(model._qconf_summary)
    copied_model.__class__ = model.__class__
    return copied_model

def auto_convert(
    module : torch.nn.Module,
    ) -> torch.nn.Module:
    def convert_to_dispatch_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationConvertTensorProxy)  # type: ignore[arg-type]
        else:
            return x

    global_disable_torch_function_override = False

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
                global_disable_torch_function_override or
                # to prevent printing things from going into an infinite loop
                func == torch.Tensor.__repr__ or
                # we don't need to override getters in this framework
                func.__name__ == '__get__'
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
            if hook_type is HookType.OP_HOOKS and func in [torch.add, torch.Tensor.add] and _check_add_has_scalar_input(args):
                hook_type = None

            if hook_type is HookType.OP_HOOKS:
                qstate: AutoQuantizationState = parent_module._auto_quant_state  # type: ignore[union-attr]
                # before hooks
                qstate.validate_cur_op(func)
                func, args, kwargs = qstate.op_convert_before_hook(
                    func, args, kwargs, parent_module)  # type: ignore[arg-type]

                # forward
                output = super().__torch_function__(func, types, args, kwargs)

                # after hooks
                output = qstate.op_convert_after_hook(
                    func, output)
                qstate.mark_cur_op_complete(func)
            else:  # HookType.NONE
                output = super().__torch_function__(func, types, args, kwargs)

            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        QuantizationConvertTensorProxy)
                assert output is not NotImplemented
            return output

        def __repr__(self):
            return f'QuantizationConvertTensorProxy({super().__repr__()})'

    cur_module = None
    module_stack : List[torch.nn.Module] = []

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
                        qstate: AutoQuantizationState = \
                            parent_module._auto_quant_state  # type: ignore[union-attr, assignment]
                        qstate.validate_cur_op(cur_module)

                        # If we are in this hook, `cur_module` is a leaf module.
                        # Therefore, we do not need to override any of its
                        # children. Disabling the overrides for performance.
                        old_global_disable_torch_function_override = \
                            global_disable_torch_function_override
                        global_disable_torch_function_override = True
                        _, args, kwargs = qstate.op_convert_before_hook(
                            cur_module, args, kwargs, cur_module)
                        if type(cur_module) in quantized_modules_has_weights:
                            weights = qstate.op_weight_convert_before_hook(cur_module)
                            output = module_call_to_function_call(self, args, weights)
                        else:
                             output = orig_module_call(self, *args, **kwargs)
                        # after hooks
                        output = qstate.op_convert_after_hook(
                            cur_module, output)

                        # Re-enable the override.
                        global_disable_torch_function_override = \
                            old_global_disable_torch_function_override

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
                        old_global_disable_torch_function_override = \
                            global_disable_torch_function_override
                        global_disable_torch_function_override = True
                        
                        output = cur_qstate.outputs_convert_hook(output)
                        global_disable_torch_function_override = \
                            old_global_disable_torch_function_override
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
 
    # If module doesn't have a configure_file attr, we can say that user has run save_qconf_summary method which have
    # computed the scales and zp, or use the user's setting from a given json file(load_qconf_summary), we need to compute
    # the scale and zp here.
    if not hasattr(module, '_qconf_summary'):
        quant_state_map = module._fqn_to_auto_quant_state_map
        # compute scales and zero_point.
        attach_scale_zp_values_to_model(module)
        nodes = convert_quant_state_map_to_nodes(quant_state_map)
        # pooling and lstm's input and output should have same scale_zp.
        sync_pool_and_lstm_input_output_scale_zp(quant_state_map, nodes)
        get_default_recipe(nodes)
    else:
        # Clear observer if module have, this will works when the user's json setting is loaded.
        for _, v in module._fqn_to_auto_quant_state_map.items():
            v.tensor_id_to_observer.clear()
            v.weight_tensor_id_to_observer.clear()
    # Attach quant_info to parent each module
    attach_op_convert_info_to_model(module)
    swap_child_modules(module)
    module.__class__ = QuantizationDispatchModule
    return module
