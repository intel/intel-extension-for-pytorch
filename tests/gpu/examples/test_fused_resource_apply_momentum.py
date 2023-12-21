import torch
from torch import Tensor
from torch import nn as nn
from typing import List
from torch.optim.optimizer import Optimizer
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest  # noqa

# for testing fused optimizer
batch_size = 256
class_num = 1000
input_channel = 512
hidden_channel = 2048
num_iter = 50
momentum = 0.9
lr = 5.0


class TrainingModel(nn.Module):
    def __init__(self):
        super(TrainingModel, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(
                input_channel,
                hidden_channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )
        self.fc = nn.Linear(
            in_features=hidden_channel, out_features=class_num, bias=True
        )

    def forward(self, x):
        x = self.m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def reference_resource_apply_momentum(
    params_momentum_buffer_list: List[Tensor],
    d_p_list: List[Tensor],
    *,
    momentum: float,
    lr: float,
    nesterov: bool,
):
    # grads may not be present always and hence the list may be empty.
    # eg. during warmup steps.
    if len(params_momentum_buffer_list) == 0:
        return
    # Check if it is first iteration
    if params_momentum_buffer_list[1] is None:
        for i, d_p in enumerate(d_p_list):
            param = params_momentum_buffer_list[2 * i]
            if momentum != 0:
                buf = torch.clone(d_p).detach()
                params_momentum_buffer_list[2 * i + 1] = buf
                d_p = buf
            param.add_(d_p)
    else:
        for i, grad in enumerate(d_p_list):
            weight = params_momentum_buffer_list[2 * i]
            momentum_buffer = params_momentum_buffer_list[2 * i + 1]

            momentum_buffer = momentum_buffer * momentum - grad * lr
            if nesterov:
                weight += momentum_buffer * momentum - grad * lr
            else:
                weight += momentum_buffer
            params_momentum_buffer_list[2 * i] = weight
            params_momentum_buffer_list[2 * i + 1] = momentum_buffer


class ReferenceFusedResourceApplyMomentum(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    # ============================================================
    # mom_t = mom * self.momentum - grad * scaled_lr
    # mom_t = state_ops.assign(mom, mom_t, use_locking=False)
    # if self.use_nesterov:
    #   var_t = var + mom_t * self.momentum - grad * scaled_lr
    # else:
    #   var_t = var + mom_t
    # return state_ops.assign(var, var_t, use_locking=False).op
    # ============================================================
    """

    def __init__(self, params, lr, momentum=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum")
        super(ReferenceFusedResourceApplyMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ReferenceFusedResourceApplyMomentum, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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
            params_with_grad_momentum = []
            d_p_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad_momentum.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        params_with_grad_momentum.append(None)
                    else:
                        params_with_grad_momentum.append(state["momentum_buffer"])
            reference_resource_apply_momentum(
                params_with_grad_momentum,
                d_p_list,
                momentum=momentum,
                lr=lr,
                nesterov=nesterov,
            )
            # update momentum_buffers in state
            # Parse the interleaved params_with_grad_momentum list and do the
            # state update as per the code below:
            # for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            #   state = self.state[p]
            #   state['momentum_buffer'] = momentum_buffer
            l = int(len(params_with_grad_momentum) / 2)
            for i in range(l):
                state = self.state[params_with_grad_momentum[2 * i]]
                state["momentum_buffer"] = params_with_grad_momentum[2 * i + 1]
        return loss


class TestTorchMethod(TestCase):
    def test_fused_resource_apply_momentum(self):
        def align_state(modelA, modelB):
            for paramA, paramB in zip(
                list(modelA.parameters()), list(modelB.parameters())
            ):
                paramA.data = paramB.detach().clone().data
                if (
                    hasattr(paramA, "grad")
                    and hasattr(paramB, "grad")
                    and paramA.grad is not None
                ):
                    paramA.grad.data = paramB.grad.clone().data
            torch.xpu.synchronize()

        def assert_equal(modelA, modelB):
            for paramA, paramB in zip(
                list(modelA.parameters()), list(modelB.parameters())
            ):
                self.assertEqual(paramA.cpu(), paramB.cpu(), atol=1e-5, rtol=1.3e-6)

        model = TrainingModel().to(device="xpu")
        model_reference = TrainingModel().to(device="xpu")

        align_state(model, model_reference)

        optimizer = torch.xpu.optim.FusedResourceApplyMomentum(
            model.parameters(), lr=lr, momentum=momentum, nesterov=True
        )
        optimizer_reference = ReferenceFusedResourceApplyMomentum(
            model_reference.parameters(), lr=lr, momentum=momentum, nesterov=True
        )

        input = torch.randn(batch_size, input_channel, 7, 7).to(device="xpu")
        input_reference = input.detach().clone()
        target = (
            torch.empty(batch_size, dtype=torch.long)
            .random_(class_num)
            .to(device="xpu")
        )

        criterion = nn.CrossEntropyLoss()

        for idx in range(num_iter):
            output = model(input)
            output_reference = model_reference(input_reference)

            loss = criterion(output, target)
            loss_reference = criterion(output_reference, target)

            optimizer.zero_grad(set_to_none=True)
            optimizer_reference.zero_grad(set_to_none=True)

            loss.backward()
            loss_reference.backward()

            align_state(model, model_reference)

            optimizer.step()
            optimizer_reference.step()

            assert_equal(model, model_reference)
            print("[successful][", idx, "] check pass")
