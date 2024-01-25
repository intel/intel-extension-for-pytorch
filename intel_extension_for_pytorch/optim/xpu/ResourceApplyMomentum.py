import torch
from torch import Tensor
from typing import List
from torch.optim.optimizer import Optimizer


def resource_apply_momentum(
    params_momentum_buffer_list: List[Tensor],
    d_p_list: List[Tensor],
    *,
    momentum: float,
    lr: float,
    nesterov: bool,
    numel: int,
):
    # grads may not be present always and hence the list may be empty.
    # eg. during warmup steps.
    if len(params_momentum_buffer_list) == 0:
        return
    # Check if it is first iteration
    if params_momentum_buffer_list[1] is None:
        for i, d_p in enumerate(d_p_list):
            param = params_momentum_buffer_list[2 * i]
            if momentum != 0:
                buf = torch.clone(d_p).detach()
                params_momentum_buffer_list[2 * i + 1] = buf
                d_p = buf
            param.add_(d_p)
    else:
        for idx in range(numel):
            torch.ops.torch_ipex.fused_resource_apply_momentum(
                params_momentum_buffer_list[idx * 2],
                params_momentum_buffer_list[idx * 2 + 1],
                d_p_list[idx],
                momentum,
                lr,
                nesterov,
            )


class FusedResourceApplyMomentum(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    # ============================================================
    # mom_t = mom * self.momentum - grad * scaled_lr
    # mom_t = state_ops.assign(mom, mom_t, use_locking=False)
    # if self.use_nesterov:
    #   var_t = var + mom_t * self.momentum - grad * scaled_lr
    # else:
    #   var_t = var + mom_t
    # return state_ops.assign(var, var_t, use_locking=False).op
    # ============================================================
    """

    def __init__(self, params, lr, momentum=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum")
        super(FusedResourceApplyMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FusedResourceApplyMomentum, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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
            params_with_grad_momentum = []
            d_p_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad_momentum.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        params_with_grad_momentum.append(None)
                    else:
                        params_with_grad_momentum.append(state["momentum_buffer"])
            resource_apply_momentum(
                params_with_grad_momentum,
                d_p_list,
                momentum=momentum,
                lr=lr,
                nesterov=nesterov,
                numel=int(len(params_with_grad_momentum) / 2),
            )
            # update momentum_buffers in state
            # Parse the interleaved params_with_grad_momentum list and do the
            # state update as per the code below:
            # for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            #   state = self.state[p]
            #   state['momentum_buffer'] = momentum_buffer
            l = int(len(params_with_grad_momentum) / 2)
            for i in range(l):
                state = self.state[params_with_grad_momentum[2 * i]]
                state["momentum_buffer"] = params_with_grad_momentum[2 * i + 1]
        return loss
