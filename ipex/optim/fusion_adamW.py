from typing import Callable, Iterable, Tuple

import torch
import ipex
from torch.optim import Optimizer



class FusedAdamWMasterWeight(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in
    `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                p_master_weight = p.detach().clone()
                # p.master_weight keeps the original datatype of the parameter, for example, fp32.
                setattr(p, "master_weight", p_master_weight)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.master_weight.device is not p.data.device:
                    p.master_weight = p.master_weight.to(p.data.device)

                # convert grad to fp32 in updating
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.master_weight.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.master_weight.data)

                beta1, beta2 = group["betas"]

                state["step"] += 1

                # fp32 master weight is involved into adamW to keep the acc and return fp32 master weight and bf16 weight at the same time.
                ipex._C.fused_adamW(p.master_weight.data, p.data, grad, state["exp_avg"], state["exp_avg_sq"], state["step"],
                                    group["lr"], group["eps"], beta1, beta2, group["weight_decay"],
                                    group["correct_bias"])

                # after updated, master weight replace the previous parameter.
                # p.data.copy_(p.master_weight.data)

        return loss
