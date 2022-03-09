import torch
import intel_extension_for_pytorch
from torch.optim.optimizer import Optimizer, required


# Specific for RN50 training. That is just a try for peak performance.
class FusionSGD(Optimizer):
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
        super(FusionSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FusionSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        """ Fusion the SGD sperate ops(fusion_amdd = add + mul + add + add) """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if momentum != 0 and weight_decay != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        d_p = d_p.add(p.data, alpha=weight_decay)
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        d_p = buf
                        p.data.add_(d_p, alpha=-group['lr'])
                    else:
                        # d_p = d_p.add(p, alpha=weight_decay)
                        # buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        # p.add_(buf, alpha=-group['lr'])

                        # update p, buf.
                        intel_extension_for_pytorch._C.fusion_amdd(p.data, d_p, param_state['momentum_buffer'], weight_decay, momentum,
                                                                   1 - dampening, -group['lr'])
                        # print('p.data = ', p.data.cpu())
                        # print('buf = ', param_state['momentum_buffer'].cpu())

        return None
