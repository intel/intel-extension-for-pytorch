r"""Port from torch/optim/adagrad.py"""
import torch
from . import _functional as F

class FusedSplitAdagrad(torch.optim.Adagrad):
    """Implements Adagrad algorithm.

    It it based on torch.optim.Adagrad
    For now, we only support create this optimizer py "ipex.optimize" inteface and expected optimize dtype==torch.bfloat
    Intel pytorch extension have 2 optimize based on torch.optim.Adagrad
        1. Fuse the process of addcmul_, sqrt, add, addcdiv as one op
        2. Support master weight split update on bfloat16 training to get better accuracy

    Args:
        opt: an instance of torch.optim.Adagrad, will set FusedSplitAdagrad's state and params to given opt's state and params
        weight_attr: this weight_attr keep the trail part of some params, need to be used at while run "step" method


    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, opt, weight_attr):
        assert type(opt) == torch.optim.Adagrad
        self.defaults = opt.defaults
        self.param_groups = opt.param_groups
        self.state = opt.state
        self.attr = weight_attr

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    fused = not p.grad.is_sparse
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state['sum'])
                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            F.adagrad(params_with_grad,
                      grads,
                      state_sums,
                      state_steps,
                      self.attr,
                      group['lr'],
                      group['weight_decay'],
                      group['lr_decay'],
                      group['eps'],
                      fused)

        return loss
