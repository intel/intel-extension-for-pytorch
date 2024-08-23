import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

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

        res_dpcpp = torch.cat(
            (
                user_cpu1.to(dpcpp_device),
                user_cpu2.to(dpcpp_device),
                user_cpu3.to(dpcpp_device),
            ),
            dim=1,
        )
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
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )

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
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )

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
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )

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
                res_xpu.is_contiguous(memory_format=torch.channels_last),
            )

    @pytest.mark.skipif(
        torch.xpu.device_count() == 1, reason="doesn't support with one device"
    )
    @pytest.mark.skip(
        reason="PT2.5: Native API failed. Native API returns: -36 (PI_ERROR_INVALID_QUEUE) -36 (PI_ERROR_INVALID_QUEUE)",
    )
    def test_cat_multi_device(self, dtype=torch.float):
        x_cpu1 = torch.randn([1, 2, 28, 28], device=cpu_device)
        x_cpu2 = torch.randn([1, 2, 28, 28], device=cpu_device)
        res_cpu = torch.cat((x_cpu1, x_cpu2))
        x_xpu1 = x_cpu1.clone().to("xpu:1")
        x_xpu2 = x_cpu2.clone().to("xpu:1")
        res_xpu = torch.cat((x_xpu1, x_xpu2))
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_cat_size0_tensor(self):
        output1_cpu = torch.cat(
            (torch.tensor([], device="cpu"), torch.tensor([1], device="cpu")), dim=0
        )
        output2_cpu = torch.cat(
            (torch.tensor([], device="cpu"), torch.tensor([1, 2], device="cpu")), dim=0
        )
        output3_cpu = torch.cat(
            (
                torch.tensor([], device="cpu"),
                torch.tensor([[1, 2], [3, 4]], device="cpu"),
            ),
            dim=0,
        )
        output4_cpu = torch.cat(
            (
                torch.tensor([], device="cpu"),
                torch.tensor([[[1]], [[2]]], device="cpu"),
            ),
            dim=0,
        )
        output1_xpu = torch.cat(
            (torch.tensor([], device="xpu"), torch.tensor([1], device="xpu")), dim=0
        )
        output2_xpu = torch.cat(
            (torch.tensor([], device="xpu"), torch.tensor([1, 2], device="xpu")), dim=0
        )
        output3_xpu = torch.cat(
            (
                torch.tensor([], device="xpu"),
                torch.tensor([[1, 2], [3, 4]], device="xpu"),
            ),
            dim=0,
        )
        output4_xpu = torch.cat(
            (
                torch.tensor([], device="xpu"),
                torch.tensor([[[1]], [[2]]], device="xpu"),
            ),
            dim=0,
        )
        self.assertEqual(output1_cpu, output1_xpu.cpu())
        self.assertEqual(output2_cpu, output2_xpu.cpu())
        self.assertEqual(output3_cpu, output3_xpu.cpu())
        self.assertEqual(output4_cpu, output4_xpu.cpu())
