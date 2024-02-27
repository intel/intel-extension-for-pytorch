import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest
import platform


class TestNNMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_linear(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(4, 2)
        tanh = nn.Tanh()

        linear_weight_cpu = linear.weight.clone()
        # print("linear weight", linear_weight_cpu)

        x_cpu = torch.tensor(
            [[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]],
            requires_grad=True,
            dtype=dtype,
        )
        # print("x_cpu", x_cpu)

        z_cpu = linear(x_cpu)
        # print("z_cpu", z_cpu)

        y_cpu = tanh(z_cpu)
        # print("y_cpu", y_cpu)

        y_cpu.backward(torch.tensor([[1.01, 8.32], [2.4, 3.22]]))
        x_grad_cpu = x_cpu.grad.clone()
        linear_weight_grad_cpu = linear.weight.grad.clone()
        # print("cpu input grad", x_grad_cpu)
        # print("cpu linear grad", linear_weight_grad_cpu)

        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        tanh_dpcpp = tanh.to("xpu")

        linear_weight_dpcpp = linear_dpcpp.weight.clone()
        # print("dpcpp linear weight", linear_weight_dpcpp.cpu())

        x_dpcpp = torch.tensor(
            [[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]],
            requires_grad=True,
            device="xpu",
            dtype=dtype,
        )
        # print("x_dpcpp", x_dpcpp.to("cpu"))

        z_dpcpp = linear_dpcpp(x_dpcpp)
        # print("z_dpcpp", z_dpcpp.to("cpu"))

        y_dpcpp = tanh(z_dpcpp)
        # print("y_dpcpp", y_dpcpp.to("cpu"))

        y_dpcpp.backward(torch.tensor([[1.01, 8.32], [2.4, 3.22]], device="xpu"))
        x_grad_dpcpp = x_dpcpp.grad.clone()
        linear_weight_grad_dpcpp = linear_dpcpp.weight.grad.clone()
        # print("dpcpp input grad", x_grad_dpcpp.cpu())
        # print("dpcpp linear grad", linear_weight_grad_dpcpp.cpu())

        self.assertEqual(x_cpu, x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)
        self.assertEqual(x_grad_cpu, x_grad_dpcpp)
        self.assertEqual(linear_weight_cpu, linear_weight_dpcpp)
        self.assertEqual(linear_weight_grad_cpu, linear_weight_grad_dpcpp)

        # new added case for the shared weights in one tensor
        # functionality
        x_cpu = torch.ones([3, 4], dtype=dtype)
        grad_cpu = torch.ones([3, 2], dtype=dtype)
        weight = torch.ones([3, 8], dtype=dtype)

        weight[:, 4:] = 2

        # print(x_cpu)
        # print(weight)

        # print(weight[:, :4])
        y1_cpu = F.linear(x_cpu, weight[:, :4])
        # # print(y_cpu)
        y2_cpu = F.linear(x_cpu, weight[:, 4:])
        # # print(y_cpu)

        # print("--------------------------------------------------------------------")

        x_sycl = x_cpu.to("xpu")
        weight_sycl = weight.to("xpu")
        # # print(x_sycl.cpu())
        # # print(weight_sycl.cpu())

        y1_sycl = F.linear(x_sycl, weight_sycl[:, :4])
        # # print(y_sycl.cpu())
        y2_sycl = F.linear(x_sycl, weight_sycl[:, 4:])
        # # print(y_sycl.cpu())
        self.assertEqual(x_cpu, x_sycl)
        self.assertEqual(weight, weight_sycl)
        self.assertEqual(y1_cpu, y1_sycl)
        self.assertEqual(y2_cpu, y2_sycl)

    # FIXME: https://jira.devtools.intel.com/browse/PYTORCHDGQ-4046
    # skip the case on Arc + windows because the device number of Arc is wrongly queried to be 2 on windows platform
    @pytest.mark.skipif(
        torch.xpu.device_count() < 2
        or (not torch.xpu.has_2d_block_array() and platform.system() == "Windows"),
        reason="doesn't support with one device",
    )
    def test_primitive_cache_for_different_devices(self):
        BS = 1
        linear0 = torch.nn.Linear(224, 224)
        input0 = torch.rand(BS, 3, 224, 224)
        linear1 = torch.nn.Linear(224, 224)
        input1 = torch.rand(BS, 3, 224, 224)

        linear1.weight.data = linear0.weight.data
        linear1.bias.data = linear0.bias.data
        input1.data = input0.data

        linear0 = linear0.to("xpu:0")
        input0 = input0.to("xpu:0")
        linear1 = linear1.to("xpu:1")
        input1 = input1.to("xpu:1")

        output0 = linear0(input0)
        print("Execution on device0")

        # oneDNN primitive used below should not be hit from the primitive cache
        # because here 2 linear op execute on different devices
        output1 = linear1(input1)
        print("Execution on device1")

        self.assertEqual(output0.cpu(), output1.cpu())
