import torch
import copy
import types
import warnings
from ._functional import sgd_step, adagrad_step, lamb_step, adam_step, adamw_step
from ._lamb import Lamb
from ..nn import utils

IPEX_FUSED_OPTIMIZER_LIST_CPU = [
    torch.optim.SGD,
    torch.optim.Adagrad,
    torch.optim.Adam,
    Lamb,
]

IPEX_FUSED_OPTIMIZER_LIST_XPU = [
    torch.optim.SGD,
    torch.optim.AdamW,
]

OPTIMIZER_FUSED_STEP_MAPPING_CPU = {
    torch.optim.SGD: sgd_step,
    torch.optim.Adagrad: adagrad_step,
    torch.optim.Adam: adam_step,
    Lamb: lamb_step,
}

# TODO: For align frontend and pass build, the xpu code is temp commented
OPTIMIZER_FUSED_STEP_MAPPING_XPU = {
    torch.optim.SGD: sgd_step,
    torch.optim.AdamW: adamw_step,
}

def patch_zero_grad_for_master_weight_training(optimizer):
    r"""
    Patch "zero_grad" method of optimizer to support BFloat16 master weight training
    Under master weight training case, the grad is actually on 'bf16_params' or 'fp16_params'.
    So the 'zero_grad' should work on the 'bf16_params' or 'fp16_params' too.
    """
    def zero_grad(self, set_to_none: bool = False):
        for p in self.params_attr:
            if 'bf16_param' in self.params_attr[p]:
                _param = self.params_attr[p]['bf16_param']
            elif 'fp16_param' in self.params_attr[p]:
                _param = self.params_attr[p]['fp16_param']
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
    setattr(optimizer, '_original_zero_grad', optimizer.zero_grad)
    setattr(optimizer, 'zero_grad', types.MethodType(zero_grad, optimizer))

def patch_step_for_master_weight_training(optimizer):
    r"""
    Patch "step" method of optimizer to support master weight training
    1.Convert BF16 or FP16 grad to FP32
    2.Call original "step" to update parameters
    3.Sync FP32 master weight back to BF16 or FP16 weight
    """
    def master_param_non_fused_step(self, closure=None):
        # convert bf16 or fp16 weight'grad to float.
        for k, value in self.params_attr.items():
            for low_precision in ['bf16_param', 'fp16_param']:
                if low_precision in value.keys():
                    # check have grad
                    if value[low_precision].requires_grad and value[low_precision].grad != None:
                        k.grad = value[low_precision].grad.detach().float()

        loss = self._original_step(closure)
        # sync mater weight to model's paramerter
        for k, value in self.params_attr.items():
            if k.device.type == 'cpu':
                if 'bf16_param' in value.keys():
                    torch.ops.torch_ipex.sync_master_weight_to_bf16(k, value['bf16_param'])
                if 'fp16_param' in value.keys():
                    torch.ops.torch_ipex.sync_master_weight_to_fp16(k, value['fp16_param'])
            elif k.device.type == 'xpu':
                if 'bf16_param' in value.keys():
                    value['bf16_param'].data = k.data.to(dtype=torch.bfloat16)
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
        for k, value in self.params_attr.items():
            assert 'bf16_param' not in value.keys(), "GradScaler is not recommended for bf16 training"
            if 'fp16_param' in value.keys():
                if value['fp16_param'].requires_grad:
                    k.grad = value['fp16_param'].grad.detach().float()
    def step_sync_weight(self, closure=None):
        loss = self._original_step(closure)
        # sync mater weight to model's paramerter
        for k, value in self.params_attr.items():
            assert 'bf16_param' not in value.keys(), "GradScaler is not recommended for bf16 training"
            if 'fp16_param' in value.keys():
                torch.ops.torch_ipex.sync_master_weight_to_fp16(k, value['fp16_param'])
        return loss
    setattr(optimizer, '_original_step', optimizer.step)
    setattr(optimizer, 'step', types.MethodType(master_param_non_fused_step, optimizer))
    setattr(optimizer, 'sync_grad', types.MethodType(sync_grad, optimizer))
    setattr(optimizer, 'step_sync_weight', types.MethodType(step_sync_weight, optimizer))

def patch_load_state_dict(optimizer):
    r"""
    Forbid optimizer load state dict after weight-prepack or weight-cast
    """
    def load_state_dict(self, state_dict):
        assert False, "_ipex_optimizer does not support load_state_dict"
    setattr(optimizer, '_original_load_state_dict', optimizer.load_state_dict)
    setattr(optimizer, 'load_state_dict', types.MethodType(load_state_dict, optimizer))

def refresh_optimizer_params_after_cast(m, attr, float_param, master_weight_split, optimizer):
    r"""
    After casting nn.Modules parameters, need refresh corresponding parameters in optimizers
    For master weight solution, the parameters in optimizers should be master weight (not BF16 weight)
    """
    if optimizer is None:
        return
    # update params
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            if p is float_param:
                if master_weight_split:
                    group['params'][i] = getattr(m, attr)
                else:
                    group['params'][i] = getattr(m, 'master_' + attr)
                # update optimizer's state.
                new_param = group['params'][i]
                if p in optimizer.state:
                    optimizer.state[new_param] = optimizer.state.pop(p)

def pack_optimizer_params_and_states(optimizer, param_pair, attrs, pack_dtype):
    """
    1. convert user's optimizer weights and related states to packed format
    While optimizer is maintain "master weight", the key in attrs is "weight",
    Need pass "weight" here as attr_key visit attr.
    2. convert user's optimizer bias to new model's bias since there is a "clone"
    """
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            if p in param_pair:
                new_param = param_pair[p]
                group['params'][i] = new_param
                # copy optimizer's state.
                if p in optimizer.state:
                    optimizer.state[new_param] = optimizer.state.pop(p)
                    # Prepack the state according to the prepacked weight.
                    # it covers both conv and linear now. TODO: LSTM or other ops.
                    if new_param in attrs:
                        attr = attrs[new_param]
                        if 'op' in attr:
                            # weight attr need "op" info to pack state while bias attr not
                            state = optimizer.state[new_param]
                            for state_key, state_value in state.items():
                                if isinstance(state_value, torch.Tensor) and state_value.size() == p.size():
                                    # We have an assumtion here that any tensor's in parameter state, if they
                                    # have same shapes with the parameter, they should share same layout with 
                                    # the parameter. Thus we need pack the state as we did to parameters.
                                    if attr['op'] in utils._weight_prepack.IPEX_WEIGHT_PREPACK_MODULE_CPU:
                                        if attr['op'] is torch.nn.Conv1d or attr['op'] is torch.nn.Conv2d or attr['op'] is torch.nn.Conv3d:
                                            if attr['op'] is torch.nn.Conv2d:
                                                memory_format = torch.channels_last
                                            elif attr['op'] is torch.nn.Conv3d:
                                                memory_format = torch.channels_last_3d
                                            else:
                                                memory_format = torch.contiguous_format
                                            value_temp = state_value.to(memory_format=memory_format) \
                                                if attr['weight_channels_last'] else state_value
                                            state[state_key] = attr['ctx'].pack(value_temp)
                                        else:
                                            state[state_key] = attr['ctx'].pack(state_value)                                   

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
                params_attr = opt.params_attr[k1]
                for state_key, state_value in v2.items():
                    if isinstance(state_value, torch.Tensor) and state_value.shape == k1.shape:
                        # We have an assumtion here that any tensor's in parameter state, if they
                        # have same shapes with the parameter, they should share same layout with 
                        # the parameter. Thus we need unpack the state as we did to parameters.
                        if 'op' in params_attr:
                            # Secondly, unpack releated states
                            if params_attr['op'] in utils._weight_prepack.IPEX_WEIGHT_PREPACK_MODULE_CPU:
                                state_value = params_attr['ctx'].to_public(state_value)
                            else:
                                assert False, "unsupported op to unpack"
                        v2[state_key] = state_value
        return opt_temp.state_dict()
    setattr(optimizer, '_original_state_dict', optimizer.state_dict)
    setattr(optimizer, 'state_dict', types.MethodType(get_optimizer_unpacked_state_dict, optimizer))

def optimizer_fusion(optimizer, master_weight_split, is_xpu=False):
    r"""
    Patch "step" method to choose IPEX optimized fused update kernel.
    """
    setattr(optimizer, 'fused', True)
    if not hasattr(optimizer, 'params_attr'):
        setattr(optimizer, 'params_attr', {})
    try:
        if not is_xpu:
            step = OPTIMIZER_FUSED_STEP_MAPPING_CPU[type(optimizer)]
        else:
            step = OPTIMIZER_FUSED_STEP_MAPPING_XPU[type(optimizer)]
        if not hasattr(optimizer, '_original_step'):
            setattr(optimizer, '_original_step', optimizer.step)
        setattr(optimizer, 'step', types.MethodType(step, optimizer))
    except KeyError:
        warnings.warn("Does not suport fused step for " + str(type(optimizer)) + ", will use non-fused step")
    return optimizer
