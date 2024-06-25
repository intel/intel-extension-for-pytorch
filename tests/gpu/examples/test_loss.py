import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")
S = 5


class TestNNMethod(TestCase):
    def test_kl_div_loss(self, dtype=torch.float):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1)
        prob2 = F.softmax(torch.randn(input_shape), 1)

        loss = nn.KLDivLoss(reduction="batchmean")
        l = loss(log_prob1, prob2)

        loss_none_reduce = nn.KLDivLoss(reduction="sum")(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        print(l, expected)

        loss_dpcpp = loss.to("xpu")
        log_prob1_dpcpp = log_prob1.to("xpu")
        prob2_dpcpp = prob2.to("xpu")

        l_dpcpp = loss_dpcpp(log_prob1_dpcpp, prob2_dpcpp)
        loss_none_reduce_dpcpp = nn.KLDivLoss(reduction="sum").to("xpu")
        expected_dpcpp = (
            loss_none_reduce_dpcpp(log_prob1_dpcpp, prob2_dpcpp) / input_shape[0]
        )

        print(l_dpcpp.to("cpu"), expected_dpcpp.to("cpu"))
        self.assertEqual(l, l_dpcpp.cpu())
        self.assertEqual(expected, expected_dpcpp.cpu())

    def _do_test(self, loss, input, target, dtype, device):
        # For some loss, torch cpu didn't support torch.half, change it to float
        # And cast the result to torch.half
        real_dtype = (
            torch.float if dtype == torch.half and device.type == "cpu" else dtype
        )
        input = input.to(dtype=real_dtype, device=device)
        target = target.to(dtype=real_dtype, device=device)

        output = loss(input, target)
        grad_output = torch.ones_like(output, dtype=dtype, device=device)
        grad_inputs = torch.autograd.grad(output, input, grad_output)

        output = output.to(dtype)
        grad_input_list = []
        for grad_input in grad_inputs:
            grad_input_list.append(grad_input.to(dtype))
        print("output is: ", output.cpu())
        print("grad_inputs are: ", tuple(x.cpu() for x in grad_inputs))

        return output, grad_input_list

    def test_bce_loss(self, dtype=torch.float):
        m = nn.Sigmoid()
        input = torch.randn(3, requires_grad=True)
        target = torch.empty(3).random_(2)
        for reduction in ["none", "mean", "sum"]:
            print("reduction is ", reduction)
            loss = nn.BCELoss(reduction=reduction)
            print("ON CPU:")
            output_cpu, grad_input_cpu = self._do_test(
                loss, m(input), target, dtype, cpu_device
            )
            print("ON XPU:")
            output_xpu, grad_input_xpu = self._do_test(
                loss, m(input), target, dtype, xpu_device
            )

            self.assertEqual(output_cpu, output_xpu)
            self.assertEqual(grad_input_cpu, grad_input_xpu)

    def test_bce_loss_with_weight(self, dtype=torch.float):
        m = nn.Sigmoid()
        input = torch.randn(3, requires_grad=True)
        target = torch.empty(3).random_(2)
        weight = torch.empty(3).random_(1)
        for reduction in ["none", "mean", "sum"]:
            print("reduction is ", reduction)
            print("ON CPU:")
            loss = nn.BCELoss(weight=weight.to(cpu_device), reduction=reduction)
            output_cpu, grad_input_cpu = self._do_test(
                loss, m(input), target, dtype, cpu_device
            )
            print("ON XPU:")
            loss = nn.BCELoss(weight=weight.to(xpu_device), reduction=reduction)
            output_xpu, grad_input_xpu = self._do_test(
                loss, m(input), target, dtype, xpu_device
            )

            self.assertEqual(output_cpu, output_xpu)
            self.assertEqual(grad_input_cpu, grad_input_xpu)

    def test_soft_margin_loss(self, dtype=torch.float):
        input = torch.randn((S, S), requires_grad=True)
        target = torch.randn((S, S))
        for reduction in ["none", "mean", "sum"]:
            print("reduction is ", reduction)
            loss = nn.SoftMarginLoss(reduction=reduction)
            print("ON CPU:")
            output_cpu, grad_input_cpu = self._do_test(
                loss, input, target, dtype, cpu_device
            )
            print("ON XPU:")
            output_xpu, grad_input_xpu = self._do_test(
                loss, input, target, dtype, xpu_device
            )

            self.assertEqual(output_cpu, output_xpu)
            self.assertEqual(grad_input_cpu, grad_input_xpu)

    def test_smooth_l1_loss(self, dtype=torch.float):
        input = torch.randn((S, S), requires_grad=True)
        target = torch.randn((S, S))
        for reduction in ["none", "mean", "sum"]:
            print("reduction is ", reduction)
            loss = nn.SmoothL1Loss(reduction=reduction)
            print("ON CPU:")
            output_cpu, grad_input_cpu = self._do_test(
                loss, input, target, dtype, cpu_device
            )
            print("ON XPU:")
            output_xpu, grad_input_xpu = self._do_test(
                loss, input, target, dtype, xpu_device
            )

            self.assertEqual(output_cpu, output_xpu)
            self.assertEqual(grad_input_cpu, grad_input_xpu)

    def test_mse_loss(self):
        def _test_mse_loss(dtype=torch.float):
            input = torch.randn((S, S), requires_grad=True, dtype=torch.float)
            target = torch.randn((S, S), dtype=torch.float)
            for reduction in ["none", "mean", "sum"]:
                print("reduction is ", reduction)
                loss = nn.MSELoss(reduction=reduction)
                print("ON CPU:")
                output_cpu, grad_input_cpu = self._do_test(
                    loss, input, target, dtype, cpu_device
                )
                print("ON XPU:")
                output_xpu, grad_input_xpu = self._do_test(
                    loss, input, target, dtype, xpu_device
                )

                print("cpu dtype:", output_cpu.dtype, grad_input_cpu[0].dtype)
                print("xpu dtype:", output_xpu.dtype, grad_input_xpu[0].dtype)

                if dtype == torch.float:
                    rtol, atol = 1e-3, 1e-5
                elif dtype == torch.float16:
                    rtol, atol = 1e-2, 1e-3

                self.assertEqual(output_cpu, output_xpu, atol=atol, rtol=rtol)
                self.assertEqual(grad_input_cpu, grad_input_xpu, atol=atol, rtol=rtol)

        _test_mse_loss(torch.float)
        _test_mse_loss(torch.half)

    def test_l1_loss(self, dtype=torch.float):
        input = torch.randn((S, S), requires_grad=True)
        target = torch.randn((S, S))
        for reduction in ["none", "mean", "sum"]:
            print("reduction is ", reduction)
            loss = nn.L1Loss(reduction=reduction)
            print("ON CPU:")
            output_cpu, grad_input_cpu = self._do_test(
                loss, input, target, dtype, cpu_device
            )
            print("ON XPU:")
            output_xpu, grad_input_xpu = self._do_test(
                loss, input, target, dtype, xpu_device
            )

            self.assertEqual(output_cpu, output_xpu)
            self.assertEqual(grad_input_cpu, grad_input_xpu)
