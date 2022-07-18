import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import copy
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_channels_last_1d(), reason="doesn't enable channels last 1d")
    def test_binary_op_channels_last_1d(self, dtype=torch.float):
        shapes = [(1, 2, 4), (2, 2, 3), (4, 4, 4), (4, 4, 1), (4, 1, 4),
                  (4, 1, 1), (1, 4, 4), (1, 4, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, W = shape[0], shape[1], shape[2]
            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is contiguous, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), False)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is channels_last_1d, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last_1d)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is channels_last_1d, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is contiguous, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last_1d)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), False)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is channels_last_1d, b is channels_last_1d alpha is 0.1:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last_1d)

            a_cpu.add_(b_cpu, alpha=0.1)
            a_xpu.add_(b_xpu, alpha=0.1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is contiguous, b is contiguous alpha is 0.1:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.add_(b_cpu, alpha=0.1)
            a_xpu.add_(b_xpu, alpha=0.1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), False)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            y_cpu = a_cpu + b_cpu
            print("\na is channels_last_1d, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            y_xpu = a_xpu + b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            y_cpu = a_cpu + b_cpu
            print("\na is channels_last_1d, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu + b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is contiguous, b is channels_last_1d:")
            y_cpu = a_cpu + b_cpu
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            y_xpu = a_xpu + b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last_1d), False)
            print("passed")

    def test_binary_op_channels_last(self, dtype=torch.float):
        shapes = [(1, 2, 3, 4), (2, 2, 3, 3), (4, 4, 4, 4), (4, 4, 1, 1), (4, 1, 4, 4),
                  (4, 1, 4, 1), (4, 1, 1, 4), (1, 4, 1, 4), (1, 4, 4, 1), (4, 1, 1, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is contiguous, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), False)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is channels_last, b is channels_last:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is channels_last, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is contiguous, b is channels_last:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)

            a_cpu.add_(b_cpu, alpha=1)
            a_xpu.add_(b_xpu, alpha=1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), False)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is channels_last, b is channels_last alpha is 0.1:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)

            a_cpu.add_(b_cpu, alpha=0.1)
            a_xpu.add_(b_xpu, alpha=0.1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is contiguous, b is contiguous alpha is 0.1:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.add_(b_cpu, alpha=0.1)
            a_xpu.add_(b_xpu, alpha=0.1)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), False)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu + b_cpu
            print("\na is channels_last, b is channels_last:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)
            y_xpu = a_xpu + b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu + b_cpu
            print("\na is channels_last, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu + b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            print("\na is contiguous, b is channels_last:")
            y_cpu = a_cpu + b_cpu
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)
            y_xpu = a_xpu + b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), False)
            print("passed")

    def test_binary_op(self, dtype=torch.float):
        x_cpu = torch.randn(5)

        x_dpcpp = x_cpu.to(dpcpp_device)
        # y_cpu1 = x_cpu.new_ones((2, 3))
        y_cpu1 = torch.randn(5)
        # y_cpu2 = x_cpu.new_ones((2, 3))
        y_cpu2 = torch.randn(5)

        y_cpu1_int = torch.tensor(
            [[3, 1, 2, 3], [2, 3, 4, 1]], dtype=torch.int32)
        # y_cpu2 = x_cpu.new_ones((2, 3))
        y_cpu2_int = torch.tensor(
            [[1, 5, 2, 4], [1, 1, 5, 5]], dtype=torch.int32)

        y_dpcpp1 = y_cpu1.to(dpcpp_device)
        y_dpcpp2 = y_cpu2.to(dpcpp_device)
        y_dpcpp1_int = y_cpu1_int.to(dpcpp_device)
        y_dpcpp2_int = y_cpu2_int.to(dpcpp_device)

        x_cpu_b_1 = torch.tensor([True, True])
        x_cpu_b_2 = torch.tensor([False, True])
        x_dpcpp_b_1 = x_cpu_b_1.to(dpcpp_device)
        x_dpcpp_b_2 = x_cpu_b_2.to(dpcpp_device)

        print("add y_cpu", y_cpu1.add(y_cpu2))
        print("add y_dpcpp", y_dpcpp1.add(y_dpcpp2).to(cpu_device))
        self.assertEqual(y_cpu1.add(y_cpu2),
                         y_dpcpp1.add(y_dpcpp2).to(cpu_device))

        print("sub y_cpu", y_cpu1.sub(y_cpu2))
        print("sub y_dpcpp", y_dpcpp1.sub(y_dpcpp2).to(cpu_device))
        self.assertEqual(y_cpu1.sub(y_cpu2),
                         y_dpcpp1.sub(y_dpcpp2).to(cpu_device))

        print("mul y_cpu", y_cpu1.mul(y_cpu2))
        print("mul y_dpcpp", y_dpcpp1.mul(y_dpcpp2).to(cpu_device))
        self.assertEqual(y_cpu1.mul(y_cpu2),
                         y_dpcpp1.mul(y_dpcpp2).to(cpu_device))

        print("div y_cpu", y_cpu1.div(y_cpu2))
        print("div y_dpcpp", y_dpcpp1.div(y_dpcpp2).to(cpu_device))
        self.assertEqual(y_cpu1.div(y_cpu2),
                         y_dpcpp1.div(y_dpcpp2).to(cpu_device))

        y_cpu_div = y_cpu1_int.div(y_cpu2_int)
        y_dpcpp_div = y_dpcpp1_int.div(y_dpcpp2_int).to(cpu_device)
        print("div y_cpu_int", y_cpu_div)
        print("div y_dpcpp_int", y_dpcpp_div)
        self.assertEqual(y_cpu_div.dtype, y_dpcpp_div.dtype)
        self.assertEqual(y_cpu_div, y_dpcpp_div)

        print("floor_divide y_cpu", y_cpu1.floor_divide(y_cpu2))
        print("floor_divide y_dpcpp", y_dpcpp1.floor_divide(y_dpcpp2).to(cpu_device))
        self.assertEqual(y_cpu1.floor_divide(y_cpu2),
                         (y_dpcpp1.floor_divide(y_dpcpp2)).to(cpu_device))

        print("__and__ y_cpu", y_cpu1_int.__and__(y_cpu2_int))
        print("__and__ y_dpcpp", y_dpcpp1_int.__and__(
            y_dpcpp2_int).to(cpu_device))
        self.assertEqual(y_cpu1_int.__and__(y_cpu2_int),
                         y_dpcpp1_int.__and__(y_dpcpp2_int).to(cpu_device))

        print("__and__ y_cpu", x_cpu_b_1.__and__(x_cpu_b_2))
        print("__and__ y_dpcpp", x_dpcpp_b_1.__and__(x_dpcpp_b_2).to(cpu_device))
        self.assertEqual(x_cpu_b_1.__and__(x_cpu_b_2),
                         x_dpcpp_b_1.__and__(x_dpcpp_b_2).to(cpu_device))

        print("__iand__ y_cpu", y_cpu1_int.__iand__(y_cpu2_int))
        print("__iand__ y_dpcpp", y_dpcpp1_int.__iand__(
            y_dpcpp2_int).to(cpu_device))
        self.assertEqual(y_cpu1_int.__iand__(y_cpu2_int),
                         y_dpcpp1_int.__iand__(y_dpcpp2_int).to(cpu_device))

        print("__iand__ y_cpu", x_cpu_b_1.__iand__(x_cpu_b_2))
        print("__iand__ y_dpcpp", x_dpcpp_b_1.__iand__(
            x_dpcpp_b_2).to(cpu_device))
        self.assertEqual(x_cpu_b_1.__iand__(x_cpu_b_2),
                         x_dpcpp_b_1.__iand__(x_dpcpp_b_2).to(cpu_device))

        print("__or__ y_cpu", y_cpu1_int.__or__(y_cpu2_int))
        print("__or__ y_dpcpp", y_dpcpp1_int.__or__(
            y_dpcpp2_int).to(cpu_device))
        self.assertEqual(y_cpu1_int.__or__(y_cpu2_int),
                         y_dpcpp1_int.__or__(y_dpcpp2_int).to(cpu_device))

        print("__or__ y_cpu", x_cpu_b_1.__or__(x_cpu_b_2))
        print("__or__ y_dpcpp", x_dpcpp_b_1.__or__(x_dpcpp_b_2).to(cpu_device))
        self.assertEqual(x_cpu_b_1.__or__(x_cpu_b_2),
                         x_dpcpp_b_1.__or__(x_dpcpp_b_2).to(cpu_device))

        print("__ior__ y_cpu", y_cpu1_int.__ior__(y_cpu2_int))
        print("__ior__ y_dpcpp", y_dpcpp1_int.__ior__(
            y_dpcpp2_int).to(cpu_device))
        self.assertEqual(y_cpu1_int.__ior__(y_cpu2_int),
                         y_dpcpp1_int.__ior__(y_dpcpp2_int).to(cpu_device))

        print("__ior__ y_cpu", x_cpu_b_1.__ior__(x_cpu_b_2))
        print("__ior__ y_dpcpp", x_dpcpp_b_1.__ior__(x_dpcpp_b_2).to(cpu_device))
        self.assertEqual(x_cpu_b_1.__ior__(x_cpu_b_2),
                         x_dpcpp_b_1.__ior__(x_dpcpp_b_2).to(cpu_device))

        print("__xor__ y_cpu", y_cpu1_int.__xor__(y_cpu2_int))
        print("__xor__ y_dpcpp", y_dpcpp1_int.__xor__(
            y_dpcpp2_int).to(cpu_device))
        self.assertEqual(y_cpu1_int.__xor__(y_cpu2_int),
                         y_dpcpp1_int.__xor__(y_dpcpp2_int).to(cpu_device))

        print("__xor__ x_cpu", x_cpu_b_1.__xor__(x_cpu_b_2))
        print("__xor__ x_dpcpp", x_dpcpp_b_1.__xor__(x_dpcpp_b_2).to(cpu_device))
        self.assertEqual(x_cpu_b_1.__xor__(x_cpu_b_2),
                         x_dpcpp_b_1.__xor__(x_dpcpp_b_2).to(cpu_device))

        print("remainder scalar y_cpu", torch.remainder(y_cpu1, 1.5))
        print("remainder scalar y_dpcpp", torch.remainder(
            y_dpcpp1, 1.5).to(cpu_device))
        self.assertEqual(torch.remainder(y_cpu1, 1.5),
                         torch.remainder(y_dpcpp1, 1.5).to(cpu_device))

        print("remainder tensor y_cpu", torch.remainder(y_cpu1, y_cpu2))
        print("remainder tensor y_dpcpp", torch.remainder(
            y_dpcpp1, y_dpcpp2).to(cpu_device))
        self.assertEqual(torch.remainder(y_cpu1, y_cpu2),
                         torch.remainder(y_dpcpp1, y_dpcpp2).to(cpu_device))

        print("fmod scalar y_cpu", torch.fmod(y_cpu1, 1.5))
        print("fmod scalar y_dpcpp", torch.fmod(y_dpcpp1, 1.5).to(cpu_device))
        self.assertEqual(torch.fmod(y_cpu1, 1.5),
                         torch.fmod(y_dpcpp1, 1.5).to(cpu_device))

        print("fmod tensor y_cpu", torch.fmod(y_cpu1, y_cpu2))
        print("fmod tensor y_dpcpp", torch.fmod(
            y_dpcpp1, y_dpcpp2).to(cpu_device))
        self.assertEqual(torch.fmod(y_cpu1, y_cpu2), torch.fmod(
            y_dpcpp1, y_dpcpp2).to(cpu_device))

    def test_add_with_alpha_block_format(self, dtype=torch.float):
        x1 = torch.randn(1, 2, 3, 3)
        x1_xpu = x1.to("xpu")
        x2 = torch.randn(1, 2, 3, 3)
        x2_xpu = x2.to("xpu")
        conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

        y = conv(x1)
        y.add_(x2, alpha=10)
        x1.add_(y, alpha=0.1)
        print(x1)

        conv.to("xpu")
        y_xpu = conv(x1_xpu)
        y_xpu.add_(x2_xpu, alpha=10)
        x1_xpu.add_(y_xpu, alpha=0.1)
        print(x1_xpu.cpu())

        self.assertEqual(x1, x1_xpu.cpu())

        a = torch.tensor([True, False], device="xpu")
        b = torch.tensor([False, True], device="xpu")
        t = torch.tensor([True, True], device="xpu")

        assert torch.equal(a + b, t)

    def test_binary_op_broadcast(self, dtype=torch.float):

        print('testing add broadcast')
        a = torch.randn(4, 16, 16, 512).to(dtype)
        b = torch.randn(4, 1, 1, 512).to(dtype)
        a_ = a.clone().xpu()
        b_ = b.clone().xpu()
        c = a + b
        c_ = a_ + b_
        self.assertEqual(c, c_.cpu())

        print('testing add_ broadcast')
        a = torch.randn(4, 16, 16, 512).to(dtype)
        b = torch.randn(4, 1, 1, 512).to(dtype)
        a_ = a.clone().xpu()
        b_ = b.clone().xpu()
        a += b
        a_ += b_
        self.assertEqual(a, a_.cpu())

        print('testing div broadcast')
        a = torch.randn(4, 16, 16, 512).to(dtype)
        b = torch.randn(4, 1, 1, 512).to(dtype)
        a_ = a.clone().xpu()
        b_ = b.clone().xpu()
        c = a / b
        c_ = a_ / b_
        self.assertEqual(c, c_.cpu())

        print('testing div_ broadcast')
        a = torch.randn(4, 16, 16, 512).to(dtype)
        b = torch.randn(4, 1, 1, 512).to(dtype)
        a_ = a.clone().xpu()
        b_ = b.clone().xpu()
        a /= b
        a_ /= b_
        self.assertEqual(a, a_.cpu())

    def test_add_block_format(self, dtype=torch.float):
        x1 = torch.randn(1, 2, 3, 3)
        x1_xpu = x1.to("xpu")
        x2 = torch.randn(1, 2, 3, 3)
        x2_xpu = x2.to("xpu")
        to_block_cpu = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)
        to_block_dpcpp = copy.deepcopy(to_block_cpu).xpu()
        with torch.xpu.onednn_layout():
            y = to_block_cpu(x1)
            y.add_(x2)
            x1.add_(y)
            print(x1)
            y_xpu = to_block_dpcpp(x1_xpu)
            y_xpu.add_(x2_xpu)
            x1_xpu.add_(y_xpu)
            print(x1_xpu.cpu())
            self.assertEqual(x1, x1_xpu.cpu())
            a = torch.tensor([True, False], device="xpu")
            b = torch.tensor([False, True], device="xpu")
            t = torch.tensor([True, True], device="xpu")
            assert torch.equal(a + b, t)

    def test_add_broadcast_block_format(self):
        with torch.xpu.onednn_layout():
            to_block_xpu = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1).xpu()
            _self = to_block_xpu(torch.rand(1, 3, 1, 1).xpu())
            _other = to_block_xpu(torch.rand(1, 3, 5, 5).xpu())
            print('block: [1, 3, 1, 1] + [1, 3, 5, 5]')
            _self + _other
            print('block: [1, 3, 5, 5] + [1, 3, 1, 1]')
            _other + _self

    def test_add_broadcast_plain_format(self):
        _self = torch.rand(1, 3, 1, 1).xpu()
        _other = torch.rand(1, 3, 5, 5).xpu()
        print('plain: [1, 3, 1, 1] + [1, 3, 5, 5]')
        _self + _other
        print('plain: [1, 3, 5, 5] + [1, 3, 1, 1]')
        _other + _self
