r"""Functional interface, port from torch/optim/_function.py"""
import torch
from torch import Tensor
from typing import List, Optional

def is_master_weight(param, params_attr):
    return  (
        param.dtype == torch.float and
        param in params_attr and
        'bf16_param' in params_attr[param]
    )

def get_bf16_grad(param, params_attr):
    assert is_master_weight(param, params_attr)
    return params_attr[param]['bf16_param'].grad

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

def _adagrad_impl(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[int],
    attr: dict,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    fused: bool,
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    for (param, grad, state_sum, step) in zip(params, grads, state_sums, state_steps):
        param2 = torch.Tensor()
        if param in attr:
            if 'trail' in attr[param]:
                assert param.dtype is torch.bfloat16
                param2 = attr[param]['trail']
            if 'bf16_param' in attr[param]:
                assert param.dtype is torch.float
                param2 = attr[param]['bf16_param']
        if fused and not grad.is_sparse:
            torch.ops.torch_ipex.adagrad_fused_step(
                param,
                grad,
                state_sum,
                param2,
                step,
                lr,
                weight_decay,
                lr_decay,
                eps)
            continue

        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
        else:
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)

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
        grads = []
        state_sums = []
        state_steps = []

        for p in group['params']:
            grad = get_bf16_grad(p, self.params_attr) if is_master_weight(p, self.params_attr) else p.grad
            if grad is not None:
                params_with_grad.append(p)
                grads.append(grad)
                state = self.state[p]
                state_sums.append(state['sum'])
                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

        _adagrad_impl(
            params_with_grad,
            grads,
            state_sums,
            state_steps,
            self.params_attr,
            group['lr'],
            group['weight_decay'],
            group['lr_decay'],
            group['eps'],
            self.fused)

    return loss

def _sgd_non_fused_micro_step(
    params: Tensor,
    d_p_list: Tensor,
    momentum_buffer_list: Optional[Tensor],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
):
    if weight_decay != 0:
        d_p = d_p.add(param, alpha=weight_decay)

    if momentum != 0:
        buf = momentum_buffer_list[i]

        if buf is None:
            buf = torch.clone(d_p).detach()
            momentum_buffer_list[i] = buf
        else:
            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

        if nesterov:
            d_p = d_p.add(buf, alpha=momentum)
        else:
            d_p = buf

    param.add_(d_p, alpha=alpha)

def _sgd_impl(
    params: List[Tensor],
    d_p_list: List[Tensor],
    attr: dict,
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    fused: bool
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        param2 = torch.Tensor()
        if param in attr:
            if 'trail' in attr[param]:
                assert param.dtype is torch.bfloat16
                param2 = attr[param]['trail']
            if 'bf16_param' in attr[param]:
                assert param.dtype is torch.float
                param2 = attr[param]['bf16_param']

        if fused and not d_p.is_sparse:
            momentum_buffer_list[i] = torch.ops.torch_ipex.sgd_fused_step(
                param,
                d_p,
                momentum_buffer_list[i],
                param2,
                momentum,
                lr,
                weight_decay,
                dampening,
                nesterov)
            continue

        if (
            d_p.is_sparse and
            d_p.dtype == torch.bfloat16 and
            weight_decay == 0 and
            momentum == 0
        ):
            # packed_add can support sparse tensor
            torch.ops.torch_ipex.packed_add(param, param2, d_p, alpha=-lr)
        else:
            # no special optimize for other non fused case, fall back to naive implementation
            d_p = d_p.to(param.dtype)
            _sgd_non_fused_micro_step(
                param,
                d_p,
                momentum_buffer_list[i],
                momentum,
                lr,
                weight_decay,
                dampening,
                nesterov
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
        d_p_list = []
        momentum_buffer_list = []
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for p in group['params']:
            grad = get_bf16_grad(p, self.params_attr) if is_master_weight(p, self.params_attr) else p.grad
            if grad is not None:
                params_with_grad.append(p)
                d_p_list.append(grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        _sgd_impl(
            params_with_grad,
            d_p_list,
            self.params_attr,
            momentum_buffer_list,
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            fused=self.fused)

        # update momentum_buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state['momentum_buffer'] = momentum_buffer

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
        param2 = torch.Tensor()
        if param in attr:
            if 'trail' in attr[param]:
                assert param.dtype is torch.bfloat16
                param2 = attr[param]['trail']
            if 'bf16_param' in attr[param]:
                assert param.dtype is torch.float
                param2 = attr[param]['bf16_param']
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
            eps)

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
    r"""Functional API that performs Lamb algorithm computation.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        grad = grad.to(exp_avg.dtype)
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        adam_step = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)

        if weight_decay != 0:
            adam_step.add_(param, alpha=weight_decay)

        weight_norm = param.norm(p=2)
        rtw_norm = adam_step.norm(p=2)
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

        for p in group['params']:
            grad = get_bf16_grad(p, self.params_attr) if is_master_weight(p, self.params_attr) else p.grad
            if grad is not None:
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients')
                if grad.device != torch.device('cpu'):
                    raise RuntimeError('Lamb supports only CPU device')
                grads.append(grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    buffer_dtype = p.dtype if p.dtype is torch.float64 else torch.float
                    state['exp_avg'] = torch.zeros(p.shape, dtype=buffer_dtype)
                    state['exp_avg_sq'] = torch.zeros(p.shape, dtype=buffer_dtype)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

        beta1, beta2 = group['betas']
        _lamb_fused_impl(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            self.params_attr,
            state_steps,
            beta1,
            beta2,
            group['lr'],
            group['weight_decay'],
            group['eps'])
    return loss