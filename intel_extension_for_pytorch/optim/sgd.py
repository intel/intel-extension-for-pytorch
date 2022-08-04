import torch
import intel_extension_for_pytorch
from torch.optim.optimizer import Optimizer, required


class SGDMasterWeight(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

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

        params_list = list(params)
        for p in params_list:
            if isinstance(p, dict):
                for pi in p["params"]:
                    p_master_weight = pi.detach().clone()
                    pi.master_weight = p_master_weight
            else:
                p_master_weight = p.detach().clone()
                p.master_weight = p_master_weight

        super(SGDMasterWeight, self).__init__(params_list, defaults)

    def __setstate__(self, state):
        super(SGDMasterWeight, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.master_weight.device is not p.data.device:
                    p.master_weight = p.master_weight.to(p.data.device)

                param_state = self.state[p]

                # first time, no momentum buffer has been created
                momentum_buffer_not_existed = 'momentum_buffer' not in param_state
                if momentum_buffer_not_existed:
                    buf = param_state['momentum_buffer'] = torch.clone(p.grad.to(p.master_weight.dtype)).detach()
                else:
                    buf = param_state['momentum_buffer']

                # original logic
                # d_p = p.grad.to(p.master_weight.dtype)

                # if weight_decay != 0:
                #     d_p = d_p.add(p.master_weight, alpha=weight_decay)

                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                #     if nesterov:
                #         d_p = d_p.add(buf, alpha=momentum)
                #     else:
                #         d_p = buf

                # p.master_weight.add_(d_p, alpha=-group['lr'])

                # p.data.copy_(p.master_weight.data)

                # fuse SGDMasterWeight update into one kernel
                # TODO: intel_extension_for_pytorch in Python code will be removed
                intel_extension_for_pytorch._C.fused_SGDMasterWeight(p.master_weight.data, p.data, p.grad, weight_decay,
                                                                     momentum_buffer_not_existed, buf,
                                                                     momentum, (1 - dampening), nesterov, group['lr'])

        return loss
