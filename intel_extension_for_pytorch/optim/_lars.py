import torch
from typing import Iterable
from torch import nn

"""
    We recommend using create_optimizer_lars and setting bn_bias_separately=True
    instead of using class Lars directly, which helps LARS skip parameters
    in BatchNormalization and bias, and has better performance in general.
    Polynomial Warmup learning rate decay is also helpful for better performance in general.
"""


def create_optimizer_lars(
    model, lr, momentum, weight_decay, bn_bias_separately, epsilon
):
    if bn_bias_separately:
        optimizer = Lars(
            [
                dict(
                    params=get_common_parameters(
                        model, exclude_func=get_norm_bias_parameters
                    )
                ),
                dict(params=get_norm_parameters(model), weight_decay=0, lars=False),
                dict(
                    params=get_bias_parameters(model, exclude_func=get_norm_parameters),
                    lars=False,
                ),
            ],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
        )
    else:
        optimizer = Lars(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
        )
    return optimizer


class Lars(torch.optim.Optimizer):
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
            lars=True,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # print("Using lars step?")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # print(len(group), group['lars'])
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]
            lars = group["lars"]
            eps = group["epsilon"]

            for index_p, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
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
        # print("Finished a normal step")
        return loss


"""
    Functions which help to skip bias and BatchNorm
"""
BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and "bias" in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param
