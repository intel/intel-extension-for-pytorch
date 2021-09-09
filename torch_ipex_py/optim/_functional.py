r"""Functional interface, port from torch/optim/_function.py"""
import torch
from torch import Tensor
from typing import List, Optional

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

def adagrad(params: List[Tensor],
            grads: List[Tensor],
            state_sums: List[Tensor],
            state_steps: List[int],
            attr: dict,
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float,
            fused: bool):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    for (param, grad, state_sum, step) in zip(params, grads, state_sums, state_steps):
        if param.dtype == torch.bfloat16 and param in attr:
            state_trail = attr[param]['trail']
        else:
            state_trail = torch.Tensor()
        if fused:
            torch.ops.torch_ipex.adagrad_fused_step(
                param,
                grad,
                state_sum,
                state_trail,
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

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        attr: dict,
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):


        d_p = d_p_list[i]
        float_d_p, float_param = None, None
        if d_p.dtype == torch.bfloat16:
            assert param in attr, "split sgd requires record 'trail' part of params in attr"
            trail = attr[param]['trail']

        if weight_decay != 0 or momentum != 0:
            float_d_p = d_p.float()
            if  d_p.dtype == torch.bfloat16:
                float_d_p = d_p.float()
                float_param =  torch.ops.torch_ipex.cat_bfloat16_float(param, trail)
            else:
                float_param =  param
                float_d_p = d_p

        if weight_decay != 0:
            float_d_p = float_d_p.add(float_param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(float_d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(float_d_p, alpha=1 - dampening)

            if nesterov:
                float_d_p = d_p.add(buf, alpha=momentum)
            else:
                float_d_p = buf

        if param.dtype is torch.bfloat16:
            if float_d_p is not None and float_param is not None:
                float_param.add_(float_d_p, alpha=-lr)
                top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(float_param)
                param.copy_(top_half)
                trail.copy_(bot_half)
            else:
                torch.ops.torch_ipex.packed_add(param, trail, d_p, alpha=-lr)
        else:
            if float_d_p is not None:
                param.add_(float_d_p, alpha=-lr)
            else:
                param.add_(d_p, alpha=-lr)
