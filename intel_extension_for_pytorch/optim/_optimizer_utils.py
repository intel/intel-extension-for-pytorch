import torch
import types
import warnings
from ._functional import sgd_step, adamw_step
from ..nn import utils # noqa

# TODO: later enable torch.optim.Adagrad, Adam and Lamb
IPEX_FUSED_OPTIMIZER_LIST = [
    torch.optim.SGD,
    torch.optim.AdamW,
]

# TODO: later enable torch.optim.Adagrad: adagrad_step and Lamb: lamb_step
# TODO: later enable adam, now we use adamw
OPTIMIZER_FUSED_STEP_MAPPING = {
    torch.optim.SGD: sgd_step,
    torch.optim.AdamW: adamw_step,
}


def patch_zero_grad_for_master_weight_training(optimizer):
    r"""
    Patch "zero_grad" method of optimizer to support BFloat16 master weight training
    Under master weight training case, the grad is actually on 'bf16_params'. So the 'zero_grad'
    should work on the 'bf16_params' too.
    """

    def zero_grad(self, set_to_none: bool = False):
        for p in self.params_attr:
            if 'bf16_param' in self.params_attr[p]:
                bf16_param = self.params_attr[p]['bf16_param']
            if bf16_param.grad is not None:
                if set_to_none:
                    bf16_param.grad = None
                else:
                    if bf16_param.grad.grad_fn is not None:
                        bf16_param.grad.detach_()
                    else:
                        bf16_param.grad.requires_grad_(False)
                    bf16_param.grad.zero_()
        self._original_zero_grad(set_to_none)
    setattr(optimizer, '_original_zero_grad', optimizer.zero_grad) # noqa B010
    setattr(optimizer, 'zero_grad', types.MethodType(zero_grad, optimizer)) # noqa B010


def patch_step_for_master_weight_training(optimizer):
    r"""
    Patch "step" method of optimizer to support BFloat16 master weight training
    1.Convert BF16 grad to FP32
    2.Call original "step" to update parameters
    3.Sync FP32 master weight back to BF16 weight
    """

    def master_param_non_fused_step(self, closure=None):
        # convert bf16 weight'grad to float.
        for k, value in self.params_attr.items():
            if 'bf16_param' in value.keys():
                if value['bf16_param'].requires_grad:
                    k.grad = value['bf16_param'].grad.detach().float()

        loss = self._original_step(closure)
        # sync mater weight to model's paramerter
        for k, value in self.params_attr.items():
            if 'bf16_param' in value.keys():
                # k is FP32 master weight, value is associated BF16 weight
                value['bf16_param'].data = k.data.to(dtype=torch.bfloat16)
        return loss
    setattr(optimizer, '_original_step', optimizer.step) # noqa B010
    setattr(optimizer, 'step', types.MethodType(master_param_non_fused_step, optimizer)) # noqa B010


def patch_load_state_dict(optimizer):
    r"""
    Forbid optimizer load state dict after weight-prepack or weight-cast
    """

    def load_state_dict(self, state_dict):
        assert False, "_ipex_optimizer does not support load_state_dict" # noqa B011
    setattr(optimizer, '_original_load_state_dict', optimizer.load_state_dict) # noqa B010
    setattr(optimizer, 'load_state_dict', types.MethodType(load_state_dict, optimizer)) # noqa B010


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
                    # optimizer.state['master weight']
                    optimizer.state[new_param] = optimizer.state.pop(p)


def optimizer_fusion(optimizer, master_weight_split):
    r"""
    Patch "step" method to choose IPEX optimized fused update kernel.
    """
    setattr(optimizer, 'fused', True) # noqa B010
    if not hasattr(optimizer, 'params_attr'):
        setattr(optimizer, 'params_attr', {}) # noqa B010
    try:
        step = OPTIMIZER_FUSED_STEP_MAPPING[type(optimizer)]
        if not hasattr(optimizer, '_original_step'):
            setattr(optimizer, '_original_step', optimizer.step) # noqa B010
        setattr(optimizer, 'step', types.MethodType(step, optimizer)) # noqa B010
    except KeyError:
        warnings.warn("Does not suport fused step for " + str(type(optimizer)) + ", will use non-fused step")
    return optimizer
