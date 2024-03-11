import torch
import copy
import types
from ..utils._logger import warn_if_user_explicitly_set
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from ._functional import (
    sgd_step,
    adagrad_step,
    lamb_step,
    adam_step,
    adamw_step,
    lars_step,
)
from ._lamb import Lamb
from ._lars import Lars
from ..nn import utils

IPEX_FUSED_OPTIMIZER_LIST_CPU = [
    torch.optim.SGD,
    torch.optim.Adagrad,
    torch.optim.Adam,
    Lamb,
    Lars,
]

IPEX_FUSED_OPTIMIZER_LIST_XPU = [
    torch.optim.SGD,
    torch.optim.AdamW,
    torch.optim.Adam,
    Lamb,
]

OPTIMIZER_FUSED_STEP_MAPPING_CPU = {
    torch.optim.SGD: sgd_step,
    torch.optim.Adagrad: adagrad_step,
    torch.optim.Adam: adam_step,
    Lamb: lamb_step,
    Lars: lars_step,
}

OPTIMIZER_FUSED_STEP_MAPPING_XPU = {
    torch.optim.SGD: sgd_step,
    torch.optim.AdamW: adamw_step,
    torch.optim.Adam: adam_step,
    Lamb: lamb_step,
    Lars: lars_step,
}


def patch_zero_grad_for_master_weight_training(optimizer):
    r"""
    Patch "zero_grad" method of optimizer to support BFloat16 master weight training
    Under master weight training case, the grad is actually on 'bf16_params' or 'fp16_params'.
    So the 'zero_grad' should work on the 'bf16_params' or 'fp16_params' too.
    """

    def zero_grad(self, set_to_none: bool = True):
        for _, v in self.params_attr.items():
            _param = v.parameter
            if _param is None:
                continue
            if _param.grad is not None:
                if set_to_none:
                    _param.grad = None
                else:
                    if _param.grad.grad_fn is not None:
                        _param.grad.detach_()
                    else:
                        _param.grad.requires_grad_(False)
                    _param.grad.zero_()
        self._original_zero_grad(set_to_none)

    if not hasattr(optimizer, "_original_zero_grad"):
        setattr(optimizer, "_original_zero_grad", optimizer.zero_grad)  # noqa: B010
        optimizer.zero_grad = types.MethodType(zero_grad, optimizer)


def patch_step_for_master_weight_training(optimizer):
    r"""
    Patch "step" method of optimizer to support master weight training
    1.Convert BF16 or FP16 grad to FP32
    2.Call original "step" to update parameters
    3.Sync FP32 master weight back to BF16 or FP16 weight
    """

    def master_param_non_fused_step(self, closure=None):
        # convert bf16 or fp16 weight'grad to float.
        for k, v in self.params_attr.items():
            _param = v.parameter
            if _param is None or _param is k:
                continue
            if _param.requires_grad and _param.grad is not None:
                k.grad = _param.grad.detach().float()

        loss = self._original_step(closure)
        # sync mater weight to model's paramerter
        for k, v in self.params_attr.items():
            _param = v.parameter
            if _param is None or _param is k:
                continue
            if k.device.type == "cpu":
                if _param.dtype == torch.bfloat16:
                    torch.ops.torch_ipex.sync_master_weight_to_bf16(k, _param)
                else:
                    assert _param.dtype == torch.float16
                    torch.ops.torch_ipex.sync_master_weight_to_fp16(k, _param)
            elif k.device.type == "xpu":
                _param.data = k.data.to(dtype=torch.bfloat16)
            else:
                pass
        return loss

    # Split master_param_non_fused_step into 2 steps:
    # 1.Sync_grad: Convert grad to FP32
    # 2.step_sync_weight: Call original "step" to update parameters and
    #   Sync FP32 master weight back to weight
    # This is because gradscaler will unscale grad and
    # it needs to sync grad to the FP32's grad first. After that gradscaler
    # will update weight and it also needs to sync FP32 master weight back to weight.
    def sync_grad(self):
        for k, v in self.params_attr.items():
            _param = v.parameter
            if _param is None or _param is k:
                continue
            assert (
                _param.dtype != torch.bfloat16
            ), "GradScaler is not recommended for bf16 training"
            if _param.requires_grad and _param.grad is not None:
                k.grad = _param.grad.detach().float()

    def step_sync_weight(self, closure=None):
        loss = self._original_step(closure)
        # sync mater weight to model's paramerter
        for k, v in self.params_attr.items():
            _param = v.parameter
            if _param is None or _param is k:
                continue
            assert (
                _param.dtype != torch.bfloat16
            ), "GradScaler is not recommended for bf16 training"
            torch.ops.torch_ipex.sync_master_weight_to_fp16(k, _param)
        return loss

    if not hasattr(optimizer, "_original_step"):
        setattr(optimizer, "_original_step", optimizer.step)  # noqa: B010
        optimizer.step = types.MethodType(master_param_non_fused_step, optimizer)
        optimizer.sync_grad = types.MethodType(sync_grad, optimizer)
        optimizer.step_sync_weight = types.MethodType(step_sync_weight, optimizer)


def pack_state(state, state_key, state_value, attr):
    if attr.num_modules != 1:
        return
    m_cls = list(attr.modules_cls)[0]
    if m_cls in utils._parameter_wrapper.IPEX_WEIGHT_PREPACK_MODULE_CPU():
        if (
            m_cls is torch.nn.Conv1d
            or m_cls is torch.nn.Conv2d
            or m_cls is torch.nn.Conv3d
        ):
            if m_cls is torch.nn.Conv2d:
                memory_format = torch.channels_last
            elif m_cls is torch.nn.Conv3d:
                memory_format = torch.channels_last_3d
            else:
                memory_format = torch.contiguous_format
            value_temp = (
                state_value.to(memory_format=memory_format)
                if attr.weight_channels_last
                else state_value
            )
            state[state_key] = attr.op_ctx.pack(value_temp)
        else:
            state[state_key] = attr.op_ctx.pack(state_value)


def patch_load_state_dict(optimizer):
    r"""
    Re-pack parameter state after load state_dict
    """

    def repack(self):
        for group in self.param_groups:
            for _, p in enumerate(group["params"]):
                if p in self.params_attr and p.device.type == "cpu":
                    attr = self.params_attr[p]
                    if attr.op_ctx is not None:
                        # weight attr need "op" info to pack state while bias attr not
                        state = self.state[p]
                        plain_format_shape = attr.plain_format_shape
                        for state_key, state_value in state.items():
                            if (
                                isinstance(state_value, torch.Tensor)
                                and state_value.size() == plain_format_shape
                            ):
                                # We have an assumption here that any tensor's in parameter state, if they
                                # have same shapes with the parameter, they should share same layout with
                                # the parameter. Thus we need pack the state as we did to parameters.
                                pack_state(state, state_key, state_value, attr)

    def original_load_state_dict_without_state_cast(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        Copied from torch/optim/optimizer.py.
        We need copy it here and change the behavior of state cast.
        For example, in out bf16 training. The mumentum buffer should always
        be float, but the original load_state_dict for optimizer will cast it to
        bfloat16 which will loss accuracy

        The original code:

            def cast(param, value, key=None):
                if isinstance(value, torch.Tensor):
                    # Floating-point types are a bit special here. They are the only ones
                    # that are assumed to always match the type of params.
                    # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                    if (key != "step"):
                        if param.is_floating_point():
                            value = value.to(param.dtype)
                        value = value.to(param.device)
                    return value
                elif isinstance(value, dict):
                    return {k: cast(param, v, key=k) for k, v in value.items()}
                elif isinstance(value, container_abcs.Iterable):
                    return type(value)(cast(param, v) for v in value)
                else:
                    return value
            state = defaultdict(dict)
            for k, v in state_dict['state'].items():
                if k in id_map:
                    param = id_map[k]
                    state[param] = cast(param, v)
                else:
                    state[k] = v

        We change it to:

            state = defaultdict(dict)
            for k, v in state_dict['state'].items():
                if k in id_map:
                    param = id_map[k]
                    state[param] = v
                else:
                    state[k] = v
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of " "parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g["params"] for g in groups)),
            )
        }

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = v
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    def load_state_dict(self, state_dict):
        original_load_state_dict_without_state_cast(self, state_dict)
        repack(self)

    if not hasattr(optimizer, "_original_load_state_dict"):
        setattr(  # noqa: B010
            optimizer, "_original_load_state_dict", optimizer.load_state_dict
        )
        optimizer.load_state_dict = types.MethodType(load_state_dict, optimizer)


def pack_optimizer_states(optimizer, param, attr):
    """
    1. convert user's optimizer weights and related states to packed format
    """
    if optimizer is None:
        return
    if param in optimizer.state:
        state = optimizer.state[param]
        plain_format_shape = attr.plain_format_shape
        for state_key, state_value in state.items():
            if (
                isinstance(state_value, torch.Tensor)
                and state_value.size() == plain_format_shape
            ):
                # We have an assumption here that any tensor's in parameter state, if they
                # have same shapes with the parameter, they should share same layout with
                # the parameter. Thus we need pack the state as we did to parameters.
                pack_state(state, state_key, state_value, attr)


def patch_state_dict(optimizer):
    r"""
    To support resume training.
    Patch "state_dict" method to return unpacked/FP32 parameters/states
    """

    def get_optimizer_unpacked_state_dict(self):
        opt = self
        opt_temp = copy.deepcopy(opt)
        for (k1, _), (_, v2) in zip(opt.state.items(), opt_temp.state.items()):
            if k1 in opt.params_attr:
                attr = opt.params_attr[k1]
                for state_key, state_value in v2.items():
                    if (
                        isinstance(state_value, torch.Tensor)
                        and state_value.shape == k1.shape
                    ):
                        # We have an assumption  here that any tensor's in parameter state, if they
                        # have same shapes with the parameter, they should share same layout with
                        # the parameter. Thus we need unpack the state as we did to parameters.
                        if attr.op_ctx is not None:
                            assert attr.num_modules == 1
                            state_value = attr.op_ctx.to_public(state_value)
                        v2[state_key] = state_value
        return opt_temp.state_dict()

    if not hasattr(optimizer, "_original_state_dict"):
        setattr(optimizer, "_original_state_dict", optimizer.state_dict)  # noqa: B010
        optimizer.state_dict = types.MethodType(
            get_optimizer_unpacked_state_dict, optimizer
        )


def optimizer_fusion(optimizer, device_type, user_explict_fuse):
    r"""
    Patch "step" method to choose IPEX optimized fused update kernel.
    """

    if not hasattr(optimizer, "params_attr"):
        setattr(optimizer, "params_attr", {})  # noqa: B010
    try:
        if device_type == "cpu":
            step = OPTIMIZER_FUSED_STEP_MAPPING_CPU[type(optimizer)]
        elif device_type == "xpu":
            step = OPTIMIZER_FUSED_STEP_MAPPING_XPU[type(optimizer)]
        else:
            msg = (
                "IPEX does not support device type "
                + str(device_type)
                + ". For now, only support CPU, XPU."
            )
            warn_if_user_explicitly_set(user_explict_fuse, msg)
            return optimizer
        if not hasattr(optimizer, "_original_step"):
            setattr(optimizer, "_original_step", optimizer.step)  # noqa: B010
        optimizer.step = types.MethodType(step, optimizer)
        setattr(optimizer, "fused", True)  # noqa: B010
    except KeyError:
        msg = (
            "Does not suport fused step for "
            + str(type(optimizer))
            + ", will use non-fused step"
        )
        warn_if_user_explicitly_set(user_explict_fuse, msg)
    return optimizer
