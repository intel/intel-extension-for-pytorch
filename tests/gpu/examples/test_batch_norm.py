import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest
import itertools

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_batch_norm_half(self, dtype=torch.half):
        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)

        bn = nn.BatchNorm2d(2)
        y_cpu = bn(x_i)
        bn.to(dpcpp_device).to(dtype)
        y_dpcpp = bn(x_dpcpp_i)
        self.assertEqual(y_cpu, y_dpcpp.cpu().float(), atol=1e-2, rtol=0)

    def test_batch_norm_half_bakcward(self, dtype=torch.float16):
        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        grad_i = torch.randn([2, 2, 3, 3], device=cpu_device)

        x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)
        grad_dpcpp_i = grad_i.to(dpcpp_device).to(dtype)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm2d(2)
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
        x_i = torch.randn([2, 2, 3, 3], dtype=dtype, device=cpu_device)
        grad_i = torch.randn([2, 2, 3, 3], dtype=dtype, device=cpu_device)

        x_dpcpp_i = x_i.to(dpcpp_device)
        grad_dpcpp_i = grad_i.to(dpcpp_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm2d(2)
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
        shapes = [
            (1, 2, 3, 3),
            (2, 2, 3, 3),
            (4, 4, 4, 4),
            (4, 4, 1, 1),
            (4, 1, 4, 4),
            (4, 1, 4, 1),
            (4, 1, 1, 4),
            (1, 4, 1, 4),
            (1, 4, 4, 1),
            (4, 1, 1, 1),
            (4, 64, 128, 1),
            (4, 32, 64, 64),
            (4, 1024, 16, 16),
        ]
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
        conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        bn = nn.BatchNorm2d(2)

        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        grad_i = torch.randn([2, 2, 3, 3], device=cpu_device)

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
        x = torch.randn(1, 2, 3, 3, dtype=torch.float)
        conv = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        bn = torch.nn.BatchNorm2d(2)

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
        bn = nn.BatchNorm2d(2)
        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        grad_i = torch.randn([2, 2, 3, 3], device=cpu_device)

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

    @pytest.mark.skipif(
        not torch.xpu.has_channels_last_1d(), reason="doesn't enable channels last 1d"
    )
    def test_channels_last_1d_fwd_and_bwd(self, dtype=torch.float):
        shapes = [
            (1, 4, 32),
            (1, 2, 3),
            (2, 2, 3),
            (4, 4, 4),
            (4, 4, 1),
            (4, 1, 4),
            (4, 1, 1),
            (1, 4, 4),
            (1, 32, 1024),
            (4, 1024, 256),
        ]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, W = shape[0], shape[1], shape[2]
            bn = nn.BatchNorm1d(C)
            x_i = torch.randn([N, C, W], device=cpu_device)
            grad_i = torch.randn([N, C, W], device=cpu_device)

            x_dpcpp_i = torch.xpu.to_channels_last_1d(x_i.to(dpcpp_device))
            grad_dpcpp_i = torch.xpu.to_channels_last_1d(grad_i.to(dpcpp_device))

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
                or 1 == y_dpcpp.shape[2]
                or (1 == y_dpcpp.shape[1] and 1 == y_dpcpp.shape[2])
            ):
                self.assertEqual(y_dpcpp.is_contiguous(), True)
                self.assertEqual(
                    torch.xpu.is_contiguous_channels_last_1d(y_dpcpp), True
                )
            else:
                self.assertEqual(y_dpcpp.is_contiguous(), False)
                self.assertEqual(
                    torch.xpu.is_contiguous_channels_last_1d(y_dpcpp), True
                )

            if (
                1 == x_dpcpp.grad.shape[1]
                or 1 == x_dpcpp.grad.shape[2]
                or (1 == x_dpcpp.grad.shape[1] and 1 == x_dpcpp.grad.shape[2])
            ):
                self.assertEqual(x_dpcpp.grad.is_contiguous(), True)
                self.assertEqual(
                    torch.xpu.is_contiguous_channels_last_1d(x_dpcpp.grad), True
                )
            else:
                self.assertEqual(x_dpcpp.grad.is_contiguous(), False)
                self.assertEqual(
                    torch.xpu.is_contiguous_channels_last_1d(x_dpcpp.grad), True
                )

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_fwd_and_bwd(self, dtype=torch.float):
        shapes = [
            (1, 2, 3, 3),
            (2, 2, 3, 3),
            (4, 4, 4, 4),
            (4, 4, 1, 1),
            (4, 1, 4, 4),
            (4, 1, 4, 1),
            (4, 1, 1, 4),
            (1, 4, 1, 4),
            (1, 4, 4, 1),
            (4, 1, 1, 1),
            (1, 8, 32, 32),
            (4, 32, 32, 32),
            (4, 8, 60, 128),
            (4, 1024, 16, 16),
            (24, 1024, 7, 7),
            (16, 32, 24, 24),
            (32, 256, 56, 56),
        ]
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
        input = torch.randn(1, 3, 3, 3, device="xpu").to(
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

    """
    tests on various batch sizes and feature sizes
    batch_size:
        3072           case1: wgroup_size_batch_dim = dpcppMaxWorkItemsPerEU
                              n_wgroups_batch_dim = evenly divisible by wgroup_size_batch_dim
        3073           case2: wgroup_size_batch_dim = dpcppMaxWorkItemsPerEU
                              n_wgroups_batch_dim = not evenly divisible by wgroup_size_batch_dim
        1, 2, 256      case3: wgroup_size_batch_dim = even batch_size < dpcppMaxWorkItemsPerEU
                              n_wgroups_batch_dim = evenly divisible by wgroup_size_batch_dim
        3              case4: wgroup_size_batch_dim = odd batch_size < dpcppMaxWorkItemsPerEU
                              n_wgroups_batch_dim = not evenly divisible by wgroup_size_batch_dim

    feature_size:
        1, 4, 7        case1: wgroup_size_feature_dim = feature_size
                              n_wgroups_feature_dim = evenly divisible by wgroup_size_feature_dim
        32, 64         case2: wgroup_size_feature_dim = 32 (SIMD wdith)
                              n_wgroups_feature_dim = evenly divisible by wgroup_size_feature_dim
        33             case3: wgroup_size_feature_dim = 32 (SIMD wdith)
                              n_wgroups_feature_dim = not evenly divisible by wgroup_size_feature_dim

    """

    def test_batch_norm_gather_stats_comprehensive(self):
        input = torch.randn(1, 3, 3, 3, device="xpu").to(
            memory_format=torch.channels_last
        )
        for [batch_size, feature_size] in itertools.product(
            [1, 256, 3072, 3073, 2, 3], [1, 4, 7, 32, 63, 64]
        ):
            print(
                "\n================== batch_size: ",
                batch_size,
                ", feature_size:",
                feature_size,
                "==================",
            )
            mean_in = (
                torch.arange(1, batch_size + 1, dtype=torch.float, device="xpu")
                .view(-1, 1)
                .repeat(1, feature_size)
            )
            invstd_in = (
                torch.arange(1, batch_size + 1, dtype=torch.float, device="xpu")
                .view(-1, 1)
                .repeat(1, feature_size)
            )
            mean_out, invstd_out = torch.batch_norm_gather_stats(
                input,
                mean_in,
                invstd_in,
                running_mean=None,
                running_var=None,
                momentum=0.1,
                eps=0,
                count=2,
            )
            if batch_size == 1:
                mean_correct = torch.tensor([1.0], device="xpu").repeat(feature_size)
                invstd_correct = torch.tensor([1.0], device="xpu").repeat(feature_size)
            elif batch_size == 256:
                mean_correct = torch.tensor([128.5000], device="xpu").repeat(
                    feature_size
                )
                invstd_correct = torch.tensor([0.0135], device="xpu").repeat(
                    feature_size
                )
            elif batch_size == 3072:
                mean_correct = torch.tensor([1536.5000], device="xpu").repeat(
                    feature_size
                )
                invstd_correct = torch.tensor([0.0011], device="xpu").repeat(
                    feature_size
                )
            elif batch_size == 3073:
                mean_correct = torch.tensor([1537.0], device="xpu").repeat(feature_size)
                invstd_correct = torch.tensor([0.0011], device="xpu").repeat(
                    feature_size
                )
            elif batch_size == 2:
                mean_correct = torch.tensor([1.5000], device="xpu").repeat(feature_size)
                invstd_correct = torch.tensor([1.0690], device="xpu").repeat(
                    feature_size
                )
            elif batch_size == 3:
                mean_correct = torch.tensor([2.0], device="xpu").repeat(feature_size)
                invstd_correct = torch.tensor([0.9448], device="xpu").repeat(
                    feature_size
                )

            self.assertTrue(torch.allclose(mean_out, mean_correct))
            self.assertTrue(torch.allclose(invstd_out, invstd_correct, atol=1e-4))

    def test_batch_norm_gather_stats_running_mean_and_running_var(self):
        input = torch.randn(1, 3, 3, 3, device="xpu").to(
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

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_sync_batchnorm_accuracy(self):
        def _batch_norm_stats(data, memory_format, mean_axes):
            mean1, _ = torch.batch_norm_stats(data, 1e-5)
            mean2, _ = torch.batch_norm_stats(
                data.to(memory_format=memory_format), 1e-5
            )
            print("mean2:", mean2)
            mean_ref = torch.mean(data, mean_axes, keepdim=False)
            print("mean ref end")

            self.assertEqual(mean_ref, mean1)
            self.assertEqual(mean_ref, mean2)

        _batch_norm_stats(
            torch.randn(1, 96, 112, 112, dtype=torch.float, device="xpu"),
            torch.channels_last,
            (0, 2, 3),
        )
        _batch_norm_stats(
            torch.randn(1, 96, 112, 112, 112, dtype=torch.float, device="xpu"),
            torch.channels_last_3d,
            (0, 2, 3, 4),
        )

    def test_batch_norm_update_stats_simple(self):
        input_cpu = torch.randn(1, 2, 3, 3, dtype=torch.float, device=cpu_device)
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

    def test_batch_norm_legit_simple(self):
        input_cpu = torch.randn(1, 2, 3, 3, dtype=torch.float, device=cpu_device)
        n_input = input_cpu.size(1)
        weight_cpu = torch.randn(2, dtype=torch.float, device=cpu_device)
        bias_cpu = torch.randn(2, dtype=torch.float, device=cpu_device)
        train = True
        momentum = 0.1
        epsilon = 1e-5

        input_dpcpp = input_cpu.to(dpcpp_device)
        weight_dpcpp = weight_cpu.to(dpcpp_device)
        bias_dpcpp = bias_cpu.to(dpcpp_device)

        def _batch_norm_legit_simple(track_stats):
            if track_stats:
                running_mean_cpu = torch.randn(
                    n_input, dtype=torch.float, device=cpu_device
                )
                running_var_cpu = torch.randn(
                    n_input, dtype=torch.float, device=cpu_device
                )

                running_mean_dpcpp = running_mean_cpu.to(dpcpp_device)
                running_var_dpcpp = running_var_cpu.to(dpcpp_device)

                (
                    out_cpu,
                    save_mean_cpu,
                    save_invstd_cpu,
                ) = torch._native_batch_norm_legit(
                    input_cpu,
                    weight_cpu,
                    bias_cpu,
                    running_mean_cpu,
                    running_var_cpu,
                    train,
                    momentum,
                    epsilon,
                )
                (
                    out_dpcpp,
                    save_mean_dpcpp,
                    save_invstd_dpcpp,
                ) = torch._native_batch_norm_legit(
                    input_dpcpp,
                    weight_dpcpp,
                    bias_dpcpp,
                    running_mean_dpcpp,
                    running_var_dpcpp,
                    train,
                    momentum,
                    epsilon,
                )
            else:
                (
                    out_cpu,
                    save_mean_cpu,
                    save_invstd_cpu,
                ) = torch._native_batch_norm_legit(
                    input_cpu, weight_cpu, bias_cpu, train, momentum, epsilon
                )
                (
                    out_dpcpp,
                    save_mean_dpcpp,
                    save_invstd_dpcpp,
                ) = torch._native_batch_norm_legit(
                    input_dpcpp, weight_dpcpp, bias_dpcpp, train, momentum, epsilon
                )

            self.assertEqual(out_cpu, out_dpcpp.to(cpu_device))
            self.assertEqual(save_mean_cpu, save_mean_dpcpp.to(cpu_device))
            self.assertEqual(save_invstd_cpu, save_invstd_dpcpp.to(cpu_device))

        _batch_norm_legit_simple(track_stats=True)
        _batch_norm_legit_simple(track_stats=False)

    def test_sync_batch_norm_elemt(self):
        input = torch.ones(1, 4, 2, 2, device=dpcpp_device)
        weight = torch.ones(4, device=dpcpp_device)
        bias = torch.zeros(4, device=dpcpp_device)
        mean = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=dpcpp_device)
        invstd = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=dpcpp_device)
        eps = 1e-5

        cuda_result = torch.tensor(
            [
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[-2.0, -2.0], [-2.0, -2.0]],
                    [[-6.0, -6.0], [-6.0, -6.0]],
                    [[-12.0, -12.0], [-12.0, -12.0]],
                ]
            ]
        )

        for memory_format in [torch.contiguous_format, torch.channels_last]:
            result = torch.batch_norm_elemt(
                input.contiguous(memory_format=memory_format),
                weight,
                bias,
                mean,
                invstd,
                eps,
            )
            self.assertEqual(result.to(cpu_device), cuda_result)

    def test_sync_batch_norm_backward_elemt(self):
        grad_output = torch.ones(1, 4, 2, 2, device=dpcpp_device)
        input = torch.ones(1, 4, 2, 2, device=dpcpp_device)
        mean = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=dpcpp_device)
        invstd = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=dpcpp_device)
        weight = torch.ones(4, device=dpcpp_device)
        sum_dy = torch.ones(4, device=dpcpp_device)
        sum_dy_xmu = torch.zeros(4, device=dpcpp_device)
        count = torch.tensor([[2], [2]], dtype=torch.int, device=dpcpp_device)

        cuda_result = torch.tensor(
            [
                [
                    [[0.75, 0.75], [0.75, 0.75]],
                    [[1.50, 1.50], [1.50, 1.50]],
                    [[2.25, 2.25], [2.25, 2.25]],
                    [[3.00, 3.00], [3.00, 3.00]],
                ]
            ]
        )

        for memory_format in [torch.contiguous_format, torch.channels_last]:
            result = torch.batch_norm_backward_elemt(
                grad_output.contiguous(memory_format=memory_format),
                input.contiguous(memory_format=memory_format),
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count,
            )
        self.assertEqual(result.to(cpu_device), cuda_result)
