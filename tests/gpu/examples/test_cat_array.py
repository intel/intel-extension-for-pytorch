import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cat_array(self, dtype=torch.float):
        user_cpu1 = torch.randn([2, 2, 3], device=cpu_device)
        user_cpu2 = torch.randn([2, 2, 3], device=cpu_device)
        user_cpu3 = torch.randn([2, 2, 3], device=cpu_device)

        res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=1)
        print("CPU Result:")
        print(res_cpu)

        res_dpcpp = torch.cat((user_cpu1.to(dpcpp_device), user_cpu2.to(
            dpcpp_device), user_cpu3.to(dpcpp_device)), dim=1)
        print("SYCL Result:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

    def test_cat_block_layout(self, dtype=torch.float):
        print("cat case1: block, plain, plain")
        x_cpu1 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu2 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu3 = torch.randn([1, 2, 28, 28], device=cpu_device)
        conv_cpu = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu1 = conv_cpu(x_cpu1)
        y_cpu2 = x_cpu2
        y_cpu3 = x_cpu3
        res_cpu = torch.cat((y_cpu1, y_cpu2, y_cpu3))

        x_xpu1 = x_cpu1.to(dpcpp_device)
        x_xpu2 = x_cpu2.to(dpcpp_device)
        x_xpu3 = x_cpu3.to(dpcpp_device)
        conv_xpu = conv_cpu.to(dpcpp_device)
        with torch.xpu.onednn_layout():
            y_xpu1 = conv_xpu(x_xpu1)
            y_xpu2 = x_xpu2
            y_xpu3 = x_xpu3
            res_xpu = torch.cat((y_xpu1, y_xpu2, y_xpu3))
            self.assertEqual(res_cpu, res_xpu.cpu())
            self.assertEqual(
                res_cpu.is_contiguous(memory_format=torch.channels_last),
                res_xpu.is_contiguous(memory_format=torch.channels_last))

        print("cat case2: plain, block, block")
        x_cpu1 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu2 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu3 = torch.randn([1, 2, 28, 28], device=cpu_device)
        conv_cpu = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu1 = x_cpu1
        y_cpu2 = conv_cpu(x_cpu2)
        y_cpu3 = conv_cpu(x_cpu3)
        res_cpu = torch.cat((y_cpu1, y_cpu2, y_cpu3))

        x_xpu1 = x_cpu1.to(dpcpp_device)
        x_xpu2 = x_cpu2.to(dpcpp_device)
        x_xpu3 = x_cpu3.to(dpcpp_device)
        conv_xpu = conv_cpu.to(dpcpp_device)
        with torch.xpu.onednn_layout():
            y_xpu1 = x_xpu1
            y_xpu2 = conv_xpu(x_xpu2)
            y_xpu3 = conv_xpu(x_xpu3)
            res_xpu = torch.cat((y_xpu1, y_xpu2, y_xpu3))
            self.assertEqual(res_cpu, res_xpu.cpu())
            self.assertEqual(
                res_cpu.is_contiguous(memory_format=torch.channels_last),
                res_xpu.is_contiguous(memory_format=torch.channels_last))

        print("cat case3: CL, block, plain")
        x_cpu1 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu2 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu3 = torch.randn([1, 2, 28, 28], device=cpu_device)
        conv_cpu = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu1 = x_cpu1.to(memory_format=torch.channels_last)
        y_cpu2 = conv_cpu(x_cpu2)
        y_cpu3 = x_cpu3
        res_cpu = torch.cat((y_cpu1, y_cpu2, y_cpu3))

        x_xpu1 = x_cpu1.to(dpcpp_device)
        x_xpu2 = x_cpu2.to(dpcpp_device)
        x_xpu3 = x_cpu3.to(dpcpp_device)
        conv_xpu = conv_cpu.to(dpcpp_device)
        with torch.xpu.onednn_layout():
            y_xpu1 = x_xpu1.to(memory_format=torch.channels_last)
            y_xpu2 = conv_xpu(x_xpu2)
            y_xpu3 = x_xpu3
            res_xpu = torch.cat((y_xpu1, y_xpu2, y_xpu3))
            self.assertEqual(res_cpu, res_xpu.cpu())
            self.assertEqual(
                res_cpu.is_contiguous(memory_format=torch.channels_last),
                res_xpu.is_contiguous(memory_format=torch.channels_last))

        print("cat case4: plain, block, CL")
        x_cpu1 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu2 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu3 = torch.randn([1, 2, 28, 28], device=cpu_device)
        conv_cpu = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu1 = x_cpu1
        y_cpu2 = conv_cpu(x_cpu2)
        y_cpu3 = x_cpu3.to(memory_format=torch.channels_last)
        res_cpu = torch.cat((y_cpu1, y_cpu2, y_cpu3))

        x_xpu1 = x_cpu1.to(dpcpp_device)
        x_xpu2 = x_cpu2.to(dpcpp_device)
        x_xpu3 = x_cpu3.to(dpcpp_device)
        conv_xpu = conv_cpu.to(dpcpp_device)
        with torch.xpu.onednn_layout():
            y_xpu1 = x_xpu1
            y_xpu2 = conv_xpu(x_xpu2)
            y_xpu3 = x_xpu3.to(memory_format=torch.channels_last)
            res_xpu = torch.cat((y_xpu1, y_xpu2, y_xpu3))
            self.assertEqual(res_cpu, res_xpu.cpu())
            self.assertEqual(
                res_cpu.is_contiguous(memory_format=torch.channels_last),
                res_xpu.is_contiguous(memory_format=torch.channels_last))
