import torch
import intel_extension_for_pytorch
from torch.optim.optimizer import Optimizer, required


class SplitSGD(Optimizer):
    r"""Implements low precision stochastic gradient descent with extra state."""

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SplitSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SplitSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dtype == torch.bfloat16:
                    param_state = self.state[p]
                    if 'bottom_half' not in param_state:
                        b_d = param_state['bottom_half'] = torch.zeros_like(
                            p.data, dtype=torch.bfloat16, device=p.device)
                    else:
                        b_d = param_state['bottom_half']

                if p.dtype == torch.bfloat16:
                    intel_extension_for_pytorch._C.packed_add(p.data, b_d, d_p, -group['lr'])
                    param_state['bottom_half'] = b_d
                else:
                    p.data.add_(d_p, alpha=-group['lr'])
        return loss
