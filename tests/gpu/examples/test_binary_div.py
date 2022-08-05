import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_channels_last_1d() or torch.xpu.using_onednn_layout(),
                        reason="doesn't enable channels last 1d or channels last does not support onednn block format")
    def test_binary_div_channels_last_1d(self, dtype=torch.float):
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

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), False)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is channels_last_1d, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu")
            a_xpu = torch.xpu.to_channels_last_1d(a_xpu)
            b_xpu = b_cpu.to("xpu")
            b_xpu = torch.xpu.to_channels_last_1d(b_xpu)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is channels_last_1d, b is contiguous:")
            a_xpu = a_cpu.to("xpu")
            a_xpu = torch.xpu.to_channels_last_1d(a_xpu)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            print("\na is contiguous, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu")
            b_xpu = torch.xpu.to_channels_last_1d(b_xpu)

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
            self.assertEqual(a_cpu, a_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), False)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            y_cpu = a_cpu / b_cpu
            print("\na is contiguous, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), False)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            y_cpu = a_cpu / b_cpu
            print("\na is channels_last_1d, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu")
            a_xpu = torch.xpu.to_channels_last_1d(a_xpu)
            b_xpu = b_cpu.to("xpu")
            b_xpu = torch.xpu.to_channels_last_1d(b_xpu)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            y_cpu = a_cpu / b_cpu
            print("\na is channels_last_1d, b is contiguous:")
            a_xpu = a_cpu.to("xpu")
            a_xpu = torch.xpu.to_channels_last_1d(a_xpu)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), False)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            print("passed")

            a_cpu = torch.randn(N, C, W)
            b_cpu = torch.randn(N, C, W)
            y_cpu = a_cpu / b_cpu
            print("\na is contiguous, b is channels_last_1d:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu")
            b_xpu = torch.xpu.to_channels_last_1d(b_xpu)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or 1 == W:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(a_xpu), False)
            print("passed")

    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    def test_binary_div_channels_last(self, dtype=torch.float):
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

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
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

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
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

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
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

            a_cpu.div_(b_cpu)
            a_xpu.div_(b_xpu)
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
            y_cpu = a_cpu / b_cpu
            print("\na is contiguous, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), False)
            print("passed")

            a_cpu = torch.randn(N, C, H, W)
            b_cpu = torch.randn(N, C, H, W)
            y_cpu = a_cpu / b_cpu
            print("\na is channels_last, b is channels_last:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)
            y_xpu = a_xpu / b_xpu
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
            y_cpu = a_cpu / b_cpu
            print("\na is channels_last, b is contiguous:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.channels_last)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            y_xpu = a_xpu / b_xpu
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
            y_cpu = a_cpu / b_cpu
            print("\na is contiguous, b is channels_last:")
            a_xpu = a_cpu.to("xpu").to(memory_format=torch.contiguous_format)
            b_xpu = b_cpu.to("xpu").to(memory_format=torch.channels_last)
            y_xpu = a_xpu / b_xpu
            self.assertEqual(y_cpu, y_xpu.cpu())
            if 1 == C or (1 == H and 1 == W):
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(a_xpu.is_contiguous(), True)
                self.assertEqual(a_xpu.is_contiguous(memory_format=torch.channels_last), False)
            print("passed")

    def test_div_rounding_mode(self, dtype=torch.float):
        a_cpu = torch.randn([100, 100])
        a_xpu = a_cpu.to("xpu")

        div_trunc_cpu = torch.div(a_cpu, 5, rounding_mode="trunc")
        div_trunc_xpu = torch.div(a_xpu, 5, rounding_mode="trunc")

        print(div_trunc_cpu)
        print(div_trunc_xpu.to("cpu"))

        self.assertEqual(div_trunc_cpu, div_trunc_xpu.to("cpu"))

        div_floor_cpu = torch.div(a_cpu, 5, rounding_mode="floor")
        div_floor_xpu = torch.div(a_xpu, 5, rounding_mode="floor")

        print(div_floor_cpu)
        print(div_floor_xpu.to("cpu"))

        self.assertEqual(div_floor_cpu, div_floor_xpu.to("cpu"))
