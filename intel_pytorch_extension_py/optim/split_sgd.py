r"""Port from torch/optim/sgd.py"""
import torch
from . import _functional as F

class SplitSGD(torch.optim.SGD):
    """Implements Adagrad algorithm.

    It it based on torch.optim.SGD
    For now, we only support create this optimizer py "ipex.optimize" inteface and expected optimize dtype==torch.bfloat
    We support master weight split update on bfloat16 training to get better accuracy

    Args:
        opt: an instance of torch.optim.SGD, will set SplitSGD's state and params to given opt's state and params
        weight_attr: this weight_attr keep the trail part of some params, need to be used at while run "step" method
    """

    def __init__(self, opt, weight_attr):
        assert type(opt) == torch.optim.SGD
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
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            F.sgd(params_with_grad,
                  d_p_list,
                  self.attr,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
