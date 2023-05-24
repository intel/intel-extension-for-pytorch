import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable


class Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr=1e-3,
        momentum=0,
        eeta=1e-3,
        weight_decay=0,
        epsilon=0.0,
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0 or eeta > 1:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eeta=eeta,
            epsilon=epsilon,
        )

        super().__init__(params, defaults)

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
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]
            eps = group["epsilon"]

            for index_p, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                decayed_grad = p.grad
                scaled_lr = lr

                w_norm = torch.norm(p)
                g_norm = torch.norm(p.grad)
                trust_ratio = torch.where(
                    w_norm > 0 and g_norm > 0,
                    eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                    torch.ones_like(w_norm),
                )

                scaled_lr *= trust_ratio.item()
                if weight_decay != 0:
                    decayed_grad = decayed_grad.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            decayed_grad
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)
        return loss
