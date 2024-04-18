r"""Functional interface, port from torch/optim/_function.py"""

import torch
from torch import Tensor
from typing import List, Optional


def is_master_weight(param, params_attr):
    if len(params_attr) == 0 or param not in params_attr:
        return False
    _param = params_attr[param].parameter
    return (
        param.dtype == torch.float
        and _param is not None
        and _param.dtype == torch.bfloat16
    )


def get_bf16_grad(param, params_attr):
    assert is_master_weight(param, params_attr)
    return params_attr[param].parameter.grad


def get_param2(param, params_attr):
    # For pure fp32 case, param2 is not needed.
    # For master weight case, param2 is the bf16 copy of fp32 weight
    # For master weight split case, param2 is the trail part of fp32 weight
    param2 = torch.Tensor()
    if param in params_attr:
        if params_attr[param].parameter_trail is not None:
            assert param.dtype is torch.bfloat16
            param2 = params_attr[param].parameter_trail
        elif is_master_weight(param, params_attr):
            param2 = params_attr[param].parameter
    return param2


def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)


def _single_tensor_adagrad(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    fused: bool
):
    for param, param2, grad, state_sum, step_t in zip(
        params, params2, grads, state_sums, state_steps
    ):
        # update step
        step_t += 1
        step = step_t.item()
        grad = grad if not maximize else -grad
        if not (grad.is_sparse or torch.is_complex(param)):
            torch.ops.torch_ipex.adagrad_fused_step(
                param, grad, state_sum, param2, step, lr, weight_decay, lr_decay, eps
            )
            continue

        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            grad = grad.add(param, alpha=weight_decay)

        grad = grad.to(param.dtype)
        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(
                _make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
            )
        else:
            is_complex = torch.is_complex(param)
            if is_complex:
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)


# keep this function here if enable fused_foreach_adagrad_later
def _multi_tensor_adagrad(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    fused: bool
):
    # Foreach functions will throw errors if given empty lists
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(grads)

    _single_tensor_adagrad(
        params,
        params2,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        has_sparse_grad=has_sparse_grad,
        maximize=False,
        fused=fused,
    )
    return


def adagrad(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: bool = None,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
    fused: bool
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad

    func(
        params,
        params2,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        fused=fused,
    )


@torch.no_grad()
def adagrad_step(self, closure=None):
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
        params2 = []
        grads = []
        state_sums = []
        state_steps = []

        has_sparse_grad = False
        for p in group["params"]:
            grad = (
                get_bf16_grad(p, self.params_attr)
                if is_master_weight(p, self.params_attr)
                else p.grad
            )
            if grad is not None:
                if grad.is_sparse:
                    has_sparse_grad = True
                params_with_grad.append(p)
                grads.append(grad)
                state = self.state[p]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])
                param2 = get_param2(p, self.params_attr)
                params2.append(param2)

        adagrad(
            params_with_grad,
            params2,
            grads,
            state_sums,
            state_steps,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            lr_decay=group["lr_decay"],
            eps=group["eps"],
            has_sparse_grad=has_sparse_grad,
            foreach=group["foreach"],
            maximize=group["maximize"],
            fused=self.fused,
        )

    return loss


def _sgd_non_fused_micro_step(
    param: Tensor,
    grad: Tensor,
    momentum_buffer: Optional[Tensor],
    momentum: float,
    lr: float,
    weight_decay: float,
    dampening: float,
    nesterov: bool,
):
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)

    if momentum != 0:
        buf = momentum_buffer

        if buf is None:
            buf = torch.clone(grad).detach()
            momentum_buffer = buf
        else:
            buf.mul_(momentum).add_(grad, alpha=1 - dampening)

        if nesterov:
            grad = grad.add(buf, alpha=momentum)
        else:
            grad = buf

    param.add_(grad, alpha=-lr)
    return momentum_buffer


def _single_tensor_sgd(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
    fused: bool
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        if not grad.is_sparse:
            momentum_buffer_list[i] = torch.ops.torch_ipex.sgd_fused_step(
                param,
                grad,
                momentum_buffer_list[i],
                params2[i],
                momentum,
                lr,
                weight_decay,
                dampening,
                nesterov,
            )
            continue

        if (
            param.dtype == torch.bfloat16
            and grad.is_sparse
            and grad.dtype == torch.bfloat16
            and weight_decay == 0
            and momentum == 0
        ):
            # packed_add can support sparse tensor
            torch.ops.torch_ipex.packed_add(param, params2[i], grad, -lr)
        else:
            # no special optimize for other non fused case, fall back to naive implementation
            grad = grad.to(param.dtype)
            momentum_buffer_list[i] = _sgd_non_fused_micro_step(
                param,
                grad,
                momentum_buffer_list[i],
                momentum,
                lr,
                weight_decay,
                dampening,
                nesterov,
            )


def _single_tensor_lars(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    eeta: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
    fused: bool
):
    if maximize:
        lr = -lr

    for i, param in enumerate(params):
        # if not grads[i].is_sparse:
        momentum_buffer_list[i] = torch.ops.torch_ipex.lars_fused_step(
            param,
            grads[i],
            momentum_buffer_list[i],
            params2[i],
            momentum,
            lr,
            eeta,
            eps,
            weight_decay,
            dampening,
            nesterov,
        )
        # continue


# keep this function here if enable fused_foreach_sgd_later
def _multi_tensor_sgd(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
    fused: bool
):
    if len(params) == 0:
        return

    _single_tensor_sgd(
        params,
        params2,
        grads,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        has_sparse_grad=has_sparse_grad,
        fused=fused,
    )


def sgd(
    params: List[Tensor],
    params2: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: bool = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    fused: bool
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(
        params,
        params2,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        fused=fused,
    )


@torch.no_grad()
def sgd_step(self, closure=None):
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
        params2 = []
        d_p_list = []
        momentum_buffer_list = []
        has_sparse_grad = False

        for p in group["params"]:
            grad = (
                get_bf16_grad(p, self.params_attr)
                if is_master_weight(p, self.params_attr)
                else p.grad
            )
            if grad is not None:
                params_with_grad.append(p)
                d_p_list.append(grad)
                if grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

                param2 = get_param2(p, self.params_attr)
                params2.append(param2)

        sgd(
            params_with_grad,
            params2,
            d_p_list,
            momentum_buffer_list,
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            lr=group["lr"],
            dampening=group["dampening"],
            nesterov=group["nesterov"],
            maximize=group["maximize"],
            has_sparse_grad=has_sparse_grad,
            foreach=group["foreach"],
            fused=self.fused,
        )

        if group["momentum"] != 0:
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

    return loss


def lars(
    params: List[Tensor],
    params2: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: bool = None,
    *,
    eeta: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    fused: bool
):
    r"""Functional API that performs LARS algorithm computation.
    dampening = 0
    nesterov = False
    maximize = False
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    func = _single_tensor_lars

    func(
        params,
        params2,
        d_p_list,
        momentum_buffer_list,
        eeta=eeta,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        fused=fused,
    )


@torch.no_grad()
def lars_step(self, closure=None):
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
        params2 = []
        d_p_list = []
        momentum_buffer_list = []
        has_sparse_grad = False

        for p in group["params"]:
            grad = (
                get_bf16_grad(p, self.params_attr)
                if is_master_weight(p, self.params_attr)
                else p.grad
            )
            if grad is not None:
                params_with_grad.append(p)
                d_p_list.append(grad)
                if grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

                param2 = get_param2(p, self.params_attr)
                params2.append(param2)
        if group["lars"]:
            lars(
                params_with_grad,
                params2,
                d_p_list,
                momentum_buffer_list,
                eeta=group["eeta"],
                eps=group["epsilon"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=0,
                nesterov=0,
                maximize=0,
                has_sparse_grad=has_sparse_grad,
                foreach=None,
                fused=self.fused,
            )
        else:
            sgd(
                params_with_grad,
                params2,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=0,
                nesterov=0,
                maximize=0,
                has_sparse_grad=has_sparse_grad,
                foreach=None,
                fused=self.fused,
            )

        # update momentum_buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state["momentum_buffer"] = momentum_buffer

    return loss


def _lamb_fused_impl(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    attr: dict,
    state_steps: List[int],
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    r"""Functional API that performs Lamb algorithm computation.
    See :class:`~torch.optim.Lamb` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        param2 = get_param2(param, attr)
        torch.ops.torch_ipex.lamb_fused_step(
            param,
            exp_avg,
            exp_avg_sq,
            grad,
            param2,
            step,
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
        )


def _lamb_impl(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    r"""Functional API that performs Lamb algorithm computation."""
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        grad = grad.to(exp_avg.dtype)
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        adam_step = (exp_avg / bias_correction1) / (
            (exp_avg_sq / bias_correction2).sqrt() + eps
        )

        if weight_decay != 0:
            adam_step.add_(param, alpha=weight_decay)

        weight_norm = param.norm(p=2)
        rtw_norm = adam_step.norm(p=2)
        if weight_norm == 0 or rtw_norm == 0:
            true_ratio = 1
        else:
            true_ratio = weight_norm / rtw_norm

        param.add_(adam_step, alpha=-lr * true_ratio)


@torch.no_grad()
def lamb_step(self, closure=None):
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
            grad = (
                get_bf16_grad(p, self.params_attr)
                if is_master_weight(p, self.params_attr)
                else p.grad
            )
            if grad is not None:
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")
                grads.append(grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    buffer_dtype = p.dtype if p.dtype is torch.float64 else torch.float
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
        _lamb_fused_impl(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            self.params_attr,
            state_steps,
            beta1,
            beta2,
            group["lr"],
            group["weight_decay"],
            group["eps"],
        )
    return loss


@torch.no_grad()
def adam_step(self, closure=None):
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
        params2 = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group["betas"]

        for p in group["params"]:
            grad = (
                get_bf16_grad(p, self.params_attr)
                if is_master_weight(p, self.params_attr)
                else p.grad
            )
            if grad is not None:
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    buffer_dtype = p.dtype if p.dtype is torch.float64 else torch.float
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=buffer_dtype
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=buffer_dtype
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=buffer_dtype
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

                param2 = get_param2(p, self.params_attr)
                params2.append(param2)

        adam(
            params_with_grad,
            params2,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=group["amsgrad"],
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
        )

    return loss


def adam(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(
        params,
        params2,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


def _single_tensor_adam(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        else:
            max_exp_avg_sq = torch.Tensor()
        step_t = state_steps[i]
        param2 = params2[i]
        # update step
        step_t += 1
        step = step_t.item()

        torch.ops.torch_ipex.adam_fused_step(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            param2,
            amsgrad,
            step,
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
        )


def _multi_tensor_adam(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    _single_tensor_adam(
        params,
        params2,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=False,
    )


def adamw(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # TODO: no foreach for now, so default false when passed
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    func(
        params,
        params2,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


def _single_tensor_adamw(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        else:
            max_exp_avg_sq = torch.Tensor()
        step_t = state_steps[i]
        param2 = params2[i]
        # update step
        step_t += 1
        step = step_t.item()

        torch.ops.torch_ipex.adamw_fused_step(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            param2,
            amsgrad,
            step,
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
        )


def _multi_tensor_adamw(
    params: List[Tensor],
    params2: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    _single_tensor_adamw(
        params,
        params2,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=False,
    )


@torch.no_grad()
def adamw_step(self, closure=None):
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
        # fp32 master weight and fp32 weight(some layer no need cast)
        params_with_grad = []
        # bf16 weight(mapped to fp32 master weight) and empty tensor(empty means no need casted layer's weight)
        params2 = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group["betas"]

        for p in group["params"]:
            # params_attr: {'layer.master_weight(fp32)': {'bf16_param': 'layer.weight(bf16)'}}
            grad = (
                get_bf16_grad(p, self.params_attr)
                if is_master_weight(p, self.params_attr)
                else p.grad
            )
            if grad is not None:
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    buffer_dtype = p.dtype
                    if p.dtype is not torch.float:
                        raise RuntimeError(
                            "parameter in optimizer(Adamw) is not FP32, need check"
                        )

                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=buffer_dtype
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=buffer_dtype
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=buffer_dtype
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

                param2 = get_param2(p, self.params_attr)
                params2.append(param2)

        adamw(
            params_with_grad,
            params2,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=group["amsgrad"],
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
        )

    return loss
