from typing import Callable, Iterable, Tuple

import torch
import ipex
from torch.optim import Optimizer
import math

class AdamWMasterWeight(Optimizer):
    r"""Implements AdamWMasterWeight algorithm, migrated with Torch official AdamW and Customer AdamW used in Bert.

    Official AdamW:
    https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    Customer AdamW:
    https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamWMasterWeight variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        transformer (boolean, optional) switch the official AdamW and customer 
            AdamW (default: False)
        correct_bias (boolean, optional) control the behaviour of the bias calculation
            (default: True)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, transformer=False, correct_bias=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if transformer and amsgrad:
            raise ValueError("Invalid combination for attribute transformer and amsgrad.")
        if not transformer and not correct_bias:
            raise ValueError("Invalid combination for attribute transformer and correct bias.")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        amsgrad=amsgrad, transformer=transformer,
                        correct_bias=correct_bias)
        super(AdamWMasterWeight, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                p_master_weight = p.detach().clone()
                # p.master_weight keeps the original datatype of the parameter, for example, fp32.
                setattr(p, "master_weight", p_master_weight)

    def __setstate__(self, state):
        super(AdamWMasterWeight, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            for p in group['params']:
                if p.grad is None:
                    continue

                if p.master_weight.device is not p.data.device:
                    p.master_weight = p.master_weight.to(p.data.device)

                if p.grad.is_sparse:
                    raise RuntimeError('AdamWMasterWeight does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.master_weight, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.master_weight, memory_format=torch.preserve_format)

                    # state['max_exp_avg_sq'] is passed to curtomer kernel. If amsgrad is configured False, pass null tensor
                    state['max_exp_avg_sq'] = torch.Tensor().xpu()
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.master_weight, memory_format=torch.preserve_format)

                # # get value
                # exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # if group['amsgrad']:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                # leave it outside, because it is done in host side and it is a scalar
                state['step'] += 1

                # Fusion AdamW here using customer kernels depends on the attribute transformer
                if not group['transformer']:
                    ipex._C.fused_adamWMasterWeight(p.master_weight.data,
                                                    p.data,
                                                    p.grad.data,
                                                    group['amsgrad'],
                                                    state['exp_avg'],
                                                    state['exp_avg_sq'],
                                                    state['max_exp_avg_sq'],
                                                    state['step'],
                                                    group['lr'],
                                                    group['eps'],
                                                    beta1,
                                                    beta2,
                                                    group['weight_decay'])
                else:
                    ipex._C.transformer_adamWMasterWeight(p.master_weight.data,
                                                          p.data,
                                                          p.grad.data,
                                                          state['exp_avg'],
                                                          state['exp_avg_sq'],
                                                          state['max_exp_avg_sq'],
                                                          state['step'],
                                                          group['lr'],
                                                          group['eps'],
                                                          beta1,
                                                          beta2,
                                                          group['weight_decay'],
                                                          group['correct_bias'])

        return loss
