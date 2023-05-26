import torch
from ._functional import _lamb_impl


class Lamb(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        fused (boolean, optional): whether to use fused kernel to accelerate
            (default: False)
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, fused=False
    ):
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
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused
        )
        super(Lamb, self).__init__(params, defaults)
        self.params_attr = {}
        self.fused = fused

    def __setstate__(self, state):
        super(Lamb, self).__setstate__(state)

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
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            trails = []
            state_steps = []

            for p in group["params"]:
                grad = p.grad
                if grad is not None:
                    params_with_grad.append(p)
                    if grad.is_sparse:
                        raise RuntimeError("Lamb does not support sparse gradients")
                    grads.append(grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        buffer_dtype = (
                            p.dtype if p.dtype is torch.float64 else torch.float
                        )
                        state["exp_avg"] = torch.zeros(
                            p.shape, dtype=buffer_dtype, device=p.device
                        )
                        state["exp_avg_sq"] = torch.zeros(
                            p.shape, dtype=buffer_dtype, device=p.device
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            beta1, beta2 = group["betas"]
            _lamb_impl(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )
        return loss
