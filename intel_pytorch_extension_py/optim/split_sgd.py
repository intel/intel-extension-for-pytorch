import torch
from torch.optim.optimizer import Optimizer, required
import _torch_ipex

_available = False
try:
    from _torch_ipex import packed_add_ 
    _available = True
except ImportError as e:
    pass

def is_available():
    return _available

class SplitSGD(Optimizer):
    r"""Implements low precision stochastic gradient descent with extra state."""

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if not is_available():
            raise ValueError("Module function 'packed_add_' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum != 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay != 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov:
            raise ValueError("Invalid nesterov value")
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
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dtype == torch.bfloat16:
                    param_state = self.state[p]
                    if 'bottom_half' not in param_state:
                        b_d = param_state['bottom_half'] = torch.zeros_like(
                            p.data, dtype=torch.bfloat16, device=p.data.device)
                    else:
                        b_d = param_state['bottom_half']

                if p.dtype == torch.bfloat16:
                    packed_add_(p.data, b_d, d_p, -group['lr'])
                else:
                    p.data.add_(d_p, alpha=-group['lr'])

        return loss
