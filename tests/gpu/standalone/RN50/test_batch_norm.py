import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest
import itertools

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
            (1, 64, 56, 56),
            (1, 64, 112, 112),
            (1, 512, 7, 7),
            (1, 128, 28, 28),
            (1, 128, 56, 56),
            (1, 256, 14, 14),
            (1, 256, 28, 28),
            (1, 256, 56, 56),
            (1, 512, 14, 14),
            (1, 512, 28, 28),
            (1, 2048, 7, 7),
            (1, 1024, 14, 14)
        ]


class TestNNMethod(TestCase):
    def test_batch_norm_half(self, dtype=torch.half):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_i = torch.randn([N, C, H, W], device=cpu_device)
            x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)

            bn = nn.BatchNorm2d(C)
            y_cpu = bn(x_i)
            bn.to(dpcpp_device).to(dtype)
            y_dpcpp = bn(x_dpcpp_i)
            self.assertEqual(y_cpu, y_dpcpp.cpu().float(), atol=1e-2, rtol=0)

    def test_batch_norm_half_backward(self, dtype=torch.float16):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_i = torch.randn([N, C, H, W], device=cpu_device)
            grad_i = torch.randn([N, C, H, W], device=cpu_device)

            x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)
            grad_dpcpp_i = grad_i.to(dpcpp_device).to(dtype)

            x_cpu = Variable(x_i, requires_grad=True)
            grad_cpu = Variable(grad_i, requires_grad=True)
            bn = nn.BatchNorm2d(C)
            y_cpu = bn(x_cpu)

            y_cpu.backward(grad_cpu)

            x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
            grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
            bn.to(dtype).to(dpcpp_device)
            y_dpcpp = bn(x_dpcpp)
            y_dpcpp.backward(grad_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).float(), rtol=10e-4, atol=10e-2)
            self.assertEqual(
                x_cpu.grad, x_dpcpp.grad.to(cpu_device).float(), rtol=10e-4, atol=10e-2
            )

    def test_batch_norm_bfloat16(self, dtype=torch.bfloat16):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_i = torch.randn([N, C, H, W], dtype=dtype, device=cpu_device)
            grad_i = torch.randn([N, C, H, W], dtype=dtype, device=cpu_device)

            x_dpcpp_i = x_i.to(dpcpp_device)
            grad_dpcpp_i = grad_i.to(dpcpp_device)

            x_cpu = Variable(x_i, requires_grad=True)
            grad_cpu = Variable(grad_i, requires_grad=True)
            bn = nn.BatchNorm2d(C)
            y_cpu = bn(x_cpu)

            y_cpu.backward(grad_cpu)

            x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
            grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
            bn.to(dtype).to(dpcpp_device)
            y_dpcpp = bn(x_dpcpp)
            y_dpcpp.backward(grad_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-3, atol=1e-1)
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-3, atol=1e-1)

    def test_batch_norm(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_i = torch.randn([N, C, H, W], device=cpu_device)
            grad_i = torch.randn([N, C, H, W], device=cpu_device)

            x_dpcpp_i = x_i.to(dpcpp_device)
            grad_dpcpp_i = grad_i.to(dpcpp_device)

            self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
            self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

            x_cpu = Variable(x_i, requires_grad=True)
            grad_cpu = Variable(grad_i, requires_grad=True)
            bn1 = nn.BatchNorm2d(C)
            bn2 = nn.BatchNorm2d(C)
            y_cpu1 = bn1(x_cpu)
            y_cpu = bn2(y_cpu1)

            y_cpu.backward(grad_cpu)

            x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
            grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
            bn1.to(dpcpp_device)
            bn2.to(dpcpp_device)

            y_dpcpp1 = bn1(x_dpcpp)
            y_dpcpp = bn2(y_dpcpp1)

            y_dpcpp.backward(grad_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bwd(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_i = torch.randn([N, C, H, W], device=cpu_device)
            grad_i = torch.randn([N, C, H, W], device=cpu_device)
            conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
            bn = nn.BatchNorm2d(C)

            x_dpcpp_i = x_i.to(dpcpp_device)
            grad_dpcpp_i = grad_i.to(dpcpp_device)

            self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
            self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

            x_cpu = Variable(x_i, requires_grad=True)
            grad_cpu = Variable(grad_i, requires_grad=True)
            y_cpu1 = conv(x_cpu)
            y_cpu = bn(y_cpu1)
            y_cpu.backward(grad_cpu)

            x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
            grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
            conv.to(dpcpp_device)
            bn.to(dpcpp_device)

            y_dpcpp1 = conv(x_dpcpp)
            y_dpcpp = bn(y_dpcpp1)
            y_dpcpp.backward(grad_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x = torch.randn([N, C, H, W], dtype=torch.float)
            conv = torch.nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
            bn = torch.nn.BatchNorm2d(C)

            relu = torch.nn.ReLU()
            ref = conv(x)
            ref = bn(ref)
            ref = relu(ref)

            x = x.to("xpu").to(memory_format=torch.channels_last)
            conv.to("xpu")
            bn.to("xpu")
            real = conv(x)
            real = bn(real)
            real = relu(real)
            real = real.contiguous().cpu()

            self.assertEqual(real, ref)

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            bn = nn.BatchNorm2d(C)
            x_i = torch.randn([N, C, H, W], device=cpu_device)
            grad_i = torch.randn([N, C, H, W], device=cpu_device)

            x_dpcpp_i = x_i.to(dpcpp_device).to(memory_format=torch.channels_last)
            grad_dpcpp_i = grad_i.to(dpcpp_device).to(memory_format=torch.channels_last)

            x_cpu = Variable(x_i, requires_grad=True)
            grad_cpu = Variable(grad_i, requires_grad=True)

            y_cpu1 = bn(x_cpu)
            y_cpu = bn(y_cpu1)

            y_cpu.backward(grad_cpu)

            x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
            grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
            bn.to(dpcpp_device)

            y_dpcpp1 = bn(x_dpcpp)
            y_dpcpp = bn(y_dpcpp1)

            y_dpcpp.backward(grad_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_fwd_and_bwd(self, dtype=torch.float):
        for dtype in [torch.bfloat16, torch.float32]:
            if dtype == torch.bfloat16:
                rtol = 1e-3
                atol = 1e-1
            else:
                rtol = 1e-4
                atol = 1e-5

            for shape in shapes:
                print(
                    "\n================== test shape: ",
                    shape,
                    ", dtype:",
                    dtype,
                    "==================",
                )
                N, C, H, W = shape[0], shape[1], shape[2], shape[3]
                bn = nn.BatchNorm2d(C)
                x_i = torch.randn([N, C, H, W], dtype=dtype, device=cpu_device)
                grad_i = torch.randn([N, C, H, W], dtype=dtype, device=cpu_device)

                x_dpcpp_i = x_i.to(dpcpp_device).to(memory_format=torch.channels_last)
                grad_dpcpp_i = grad_i.to(dpcpp_device).to(
                    memory_format=torch.channels_last
                )

                x_cpu = Variable(x_i, requires_grad=True)
                grad_cpu = Variable(grad_i, requires_grad=True)

                y_cpu1 = bn(x_cpu)
                y_cpu = bn(y_cpu1)

                y_cpu.backward(grad_cpu)

                x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
                grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
                bn.to(dpcpp_device)

                y_dpcpp1 = bn(x_dpcpp)
                y_dpcpp = bn(y_dpcpp1)
                y_dpcpp.backward(grad_dpcpp)

                if (
                    1 == y_dpcpp.shape[1]
                    or (1 == y_dpcpp.shape[2] and 1 == y_dpcpp.shape[3])
                    or (
                        1 == y_dpcpp.shape[1]
                        and 1 == y_dpcpp.shape[2]
                        and 1 == y_dpcpp.shape[3]
                    )
                ):
                    self.assertEqual(y_dpcpp.is_contiguous(), True)
                    self.assertEqual(
                        y_dpcpp.is_contiguous(memory_format=torch.channels_last), True
                    )
                else:
                    self.assertEqual(y_dpcpp.is_contiguous(), False)
                    self.assertEqual(
                        y_dpcpp.is_contiguous(memory_format=torch.channels_last), True
                    )

                if (
                    1 == x_dpcpp.grad.shape[1]
                    or (1 == x_dpcpp.grad.shape[2] and 1 == x_dpcpp.grad.shape[3])
                    or (
                        1 == x_dpcpp.grad.shape[1]
                        and 1 == x_dpcpp.grad.shape[2]
                        and 1 == x_dpcpp.grad.shape[3]
                    )
                ):
                    self.assertEqual(x_dpcpp.grad.is_contiguous(), True)
                    self.assertEqual(
                        x_dpcpp.grad.is_contiguous(memory_format=torch.channels_last),
                        True,
                    )
                else:
                    self.assertEqual(x_dpcpp.grad.is_contiguous(), False)
                    self.assertEqual(
                        x_dpcpp.grad.is_contiguous(memory_format=torch.channels_last),
                        True,
                    )

                self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=rtol, atol=atol)
                self.assertEqual(
                    x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=rtol, atol=atol
                )

    def test_batch_norm_gather_stats(self):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            input = torch.randn([N, C, H, W], device="xpu").to(
                memory_format=torch.channels_last
            )
            mean, invstd = torch.batch_norm_gather_stats(
                input,
                mean=torch.ones(64, 3, device="xpu"),
                invstd=torch.ones(64, 3, device="xpu"),
                running_mean=None,
                running_var=None,
                momentum=0.1,
                eps=1e-5,
                count=2,
            )
            self.assertEqual(mean, torch.ones(3, device="xpu"))
            self.assertEqual(invstd, torch.ones(3, device="xpu"))

    def test_batch_norm_gather_stats_running_mean_and_running_var(self):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            input = torch.randn([N, C, H, W], device="xpu").to(
                memory_format=torch.channels_last
            )

            running_mean_ptr = torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="xpu"
            )
            running_var_ptr = torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="xpu"
            )

            torch.batch_norm_gather_stats(
                input,
                mean=torch.ones(64, 8, device="xpu"),
                invstd=torch.ones(64, 8, device="xpu"),
                running_mean=running_mean_ptr,
                running_var=running_var_ptr,
                momentum=0.1,
                eps=1e-5,
                count=2,
            )
            running_mean_correct = torch.tensor(
                [1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3], device="xpu"
            )
            running_var_correct = torch.tensor(
                [1.0008, 1.9008, 2.8008, 3.7008, 4.6008, 5.5008, 6.4008, 7.3008],
                device="xpu",
            )
            self.assertTrue(torch.allclose(running_mean_ptr, running_mean_correct))
            self.assertTrue(torch.allclose(running_var_ptr, running_var_correct, atol=1e-4))

    def test_batch_norm_update_stats_simple(self):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            input_cpu = torch.randn([N, C, H, W], dtype=torch.float, device=cpu_device)
            n_input = input_cpu.size(1)
            running_mean_cpu = torch.randn(n_input, dtype=torch.float, device=cpu_device)
            running_var_cpu = torch.randn(n_input, dtype=torch.float, device=cpu_device)
            momentum = 0.1

            input_dpcpp = input_cpu.to(dpcpp_device)
            running_mean_dpcpp = running_mean_cpu.to(dpcpp_device)
            running_var_dpcpp = running_var_cpu.to(dpcpp_device)

            save_mean_cpu, save_var_cpu = torch.batch_norm_update_stats(
                input_cpu, running_mean_cpu, running_var_cpu, momentum
            )
            save_mean_dpcpp, save_var_dpcpp = torch.batch_norm_update_stats(
                input_dpcpp, running_mean_dpcpp, running_var_dpcpp, momentum
            )

            self.assertEqual(save_mean_cpu, save_mean_dpcpp.to(cpu_device))
            self.assertEqual(save_var_cpu, save_var_dpcpp.to(cpu_device))

