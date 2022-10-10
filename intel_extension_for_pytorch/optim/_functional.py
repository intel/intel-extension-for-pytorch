r"""Functional interface, port from torch/optim/_function.py"""
import torch
from torch import Tensor
from typing import List, Optional
import intel_extension_for_pytorch


def is_master_weight(param, params_attr):
    return (
        param.dtype == torch.float
        and param in params_attr
        and 'bf16_param' in params_attr[param]
    )


def get_bf16_grad(param, params_attr):
    assert is_master_weight(param, params_attr)
    return params_attr[param]['bf16_param'].grad


def get_param2(param, params_attr):
    # For pure fp32 layer, param2 is not needed.
    # For master weight layer, param2 is the bf16 copy of fp32 weight
    # For master weight split layer, param2 is the trail part of fp32 weight
    param2 = torch.Tensor()
    if param in params_attr:
        if 'trail' in params_attr[param]:
            assert param.dtype is torch.bfloat16
            param2 = params_attr[param]['trail']
        if 'bf16_param' in params_attr[param]:
            assert param.dtype is torch.float
            param2 = params_attr[param]['bf16_param']
    return param2


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


def _single_tensor_sgd(params: List[Tensor],
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
                       fused: bool):
    if maximize:
        lr = -lr

    # TODO: multi tensor apply for updating some small weights
    # watch out empty tensor in params2 and mixed datatype in grad
    for i, param in enumerate(params):
        if not grads[i].is_sparse:
            # param: fp32 master weight and fp32 weight(some layer no need cast)
            # grad: bf16/fp32 grad, bf16 grad from casted layer, fp32 grad from no casted layer
            # momentum_buffer_list[i]: be none for first iter
            # param2: bf16 weight(mapped to fp32 master weight) and empty tensor(empty means no need casted layer's weight)
            momentum_buffer_list[i] = intel_extension_for_pytorch._C.fused_SGD(
                param,
                grads[i],
                momentum_buffer_list[i],
                params2[i],
                momentum,
                lr,
                weight_decay,
                dampening,
                nesterov)
            continue

        if (
            param.dtype == torch.bfloat16
            and grads[i].is_sparse
            and grads[i].dtype == torch.bfloat16
            and weight_decay == 0
            and momentum == 0
        ):
            # packed_add can support sparse tensor
            raise RuntimeError("not implemented step function for sparse grad update")
            # torch.ops.torch_ipex.packed_add(param, params2[i], grads[i], alpha=-lr)
        else:
            # no special optimize for other non fused case, fall back to naive implementation
            grads[i] = grads[i].to(param.dtype)
            momentum_buffer_list[i] = _sgd_non_fused_micro_step(
                param,
                grads[i],
                momentum_buffer_list[i],
                momentum,
                lr,
                weight_decay,
                dampening,
                nesterov
            )


def _multi_tensor_sgd(params: List[Tensor],
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
                      fused: bool):

    if len(params) == 0:
        return

    _single_tensor_sgd(params,
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
                       fused=fused)


def sgd(params: List[Tensor],
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
        fused: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
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
         fused=fused)


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
        # fp32 master weight and fp32 weight(some layer no need cast)
        params_with_grad = []
        # bf16 weight(mapped to fp32 master weight) and empty tensor(empty means no need casted layer's weight)
        params2 = []
        # bf16/fp32 grad, bf16 grad from casted layer, fp32 grad from no casted layer
        d_p_list = []
        # be None for first iter
        momentum_buffer_list = []
        has_sparse_grad = False

        # p is model's parameter
        # in refresh_optimizer_params_after_cast, p is master weight
        for p in group['params']:
            # params_attr: {'layer.master_weight(fp32)': {'bf16_param': 'layer.weight(bf16)'}}
            # if p is master weight, the grad should be fetched from layer.weight(bf16)
            grad = get_bf16_grad(p, self.params_attr) if is_master_weight(p, self.params_attr) else p.grad
            if grad is not None:
                params_with_grad.append(p)
                d_p_list.append(grad)
                if grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

                param2 = get_param2(p, self.params_attr)
                params2.append(param2)

        # TODO: the attribute 'foreach' can be used for us to choose multi_tensor_apply
        # TODO: torch1.10 has no attribute: group['maximize'] and group['foreach']. They are in torch1.13
        sgd(params_with_grad,
            params2,
            d_p_list,
            momentum_buffer_list,
            weight_decay=group['weight_decay'],
            momentum=group['momentum'],
            lr=group['lr'],
            dampening=group['dampening'],
            nesterov=group['nesterov'],
            maximize=False,
            has_sparse_grad=has_sparse_grad,
            foreach=False,
            fused=self.fused)

        # update momentum_buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state['momentum_buffer'] = momentum_buffer

    return loss


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
        beta1, beta2 = group['betas']

        for p in group['params']:
            # params_attr: {'layer.master_weight(fp32)': {'bf16_param': 'layer.weight(bf16)'}}
            grad = get_bf16_grad(p, self.params_attr) if is_master_weight(p, self.params_attr) else p.grad
            if grad is not None:
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    buffer_dtype = p.dtype
                    if p.dtype is not torch.float:
                        raise RuntimeError("parameter in optimizer(Adamw) is not FP32, need check")

                    state['step'] = torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=buffer_dtype)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=buffer_dtype)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=buffer_dtype)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

                param2 = get_param2(p, self.params_attr)
                params2.append(param2)

        # TODO: the attribute 'foreach' can be used for us to choose multi_tensor_apply
        # TODO: torch1.10 has no attribute: group['maximize'] and group['foreach']. They are in torch1.13
        adamw(params_with_grad,
              params2,
              grads,
              exp_avgs,
              exp_avg_sqs,
              max_exp_avg_sqs,
              state_steps,
              amsgrad=group['amsgrad'],
              beta1=beta1,
              beta2=beta2,
              lr=group['lr'],
              weight_decay=group['weight_decay'],
              eps=group['eps'],
              maximize=False,
              foreach=False)

    return loss


def adamw(params: List[Tensor],
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
          maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    # TODO: no foreach for now, so default false when passed
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    func(params,
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
         maximize=maximize)


def _single_tensor_adamw(params: List[Tensor],
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
                         maximize: bool):

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

        intel_extension_for_pytorch._C.fused_ADAMW(param,
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
                                                   eps)


def _multi_tensor_adamw(params: List[Tensor],
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
                        maximize: bool):

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
        maximize=False
    )
