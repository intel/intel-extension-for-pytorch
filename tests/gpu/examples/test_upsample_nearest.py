import torch
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_upsamle_nearest(self, dtype=torch.float):
        # #### upsample nearest 1D #####
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode="nearest", recompute_scale_factor=rsf
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode="nearest", recompute_scale_factor=rsf
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30), dtype=torch.float32, device=cpu_device)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # #### upsample nearest 2D #####
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode="nearest", recompute_scale_factor=rsf
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode="nearest", recompute_scale_factor=rsf
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn(
            (2, 3, 30, 40), dtype=torch.float32, device=cpu_device
        )
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # #### upsample nearest 3D #####
        input_cpu = torch.randn((2, 3, 2, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8, 1]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode="nearest", recompute_scale_factor=rsf
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode="nearest", recompute_scale_factor=rsf
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn(
            (2, 3, 12, 40, 5), dtype=torch.float32, device=cpu_device
        )
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_with_non_integral_scales(self, dtype=torch.float):
        # 2D, CF
        x = torch.rand(1, 3, 18, 32)
        shape = (36, 48)
        dummy_img = (x - x.min()) / (x.max() - x.min()) * 255.0
        dummy_img_xpu = dummy_img.to("xpu")
        dummy_img.requires_grad = True
        dummy_img_xpu.requires_grad = True
        y_cpu = torch.functional.F.interpolate(dummy_img, shape)
        y_xpu = torch.functional.F.interpolate(dummy_img_xpu, shape)
        self.assertEqual(y_cpu, y_xpu.cpu())
        grad_out_cpu = torch.randn(
            (1, 3, 36, 48), dtype=torch.float32, device=cpu_device
        )
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        y_cpu.backward(grad_out_cpu)
        y_xpu.backward(grad_out_dpcpp)
        grad_cpu = dummy_img.grad
        grad_dpcpp = dummy_img_xpu.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # 2D, CL
        x = torch.rand(1, 3, 18, 32).to(memory_format=torch.channels_last)
        shape = (36, 48)
        dummy_img = (x - x.min()) / (x.max() - x.min()) * 255.0
        dummy_img_xpu = dummy_img.to("xpu")
        dummy_img.requires_grad = True
        dummy_img_xpu.requires_grad = True
        y_cpu = torch.functional.F.interpolate(dummy_img, shape)
        y_xpu = torch.functional.F.interpolate(dummy_img_xpu, shape)
        self.assertEqual(y_cpu, y_xpu.cpu())
        grad_out_cpu = torch.randn(
            (1, 3, 36, 48), dtype=torch.float32, device=cpu_device
        ).to(memory_format=torch.channels_last)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        y_cpu.backward(grad_out_cpu)
        y_xpu.backward(grad_out_dpcpp)
        grad_cpu = dummy_img.grad
        grad_dpcpp = dummy_img_xpu.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # 1D, CF
        x = torch.rand(3, 18, 32)
        shape = 48
        dummy_img = (x - x.min()) / (x.max() - x.min()) * 255.0
        dummy_img_xpu = dummy_img.to("xpu")
        dummy_img.requires_grad = True
        dummy_img_xpu.requires_grad = True
        y_cpu = torch.functional.F.interpolate(dummy_img, shape)
        y_xpu = torch.functional.F.interpolate(dummy_img_xpu, shape)
        self.assertEqual(y_cpu, y_xpu.cpu())
        grad_out_cpu = torch.randn((3, 18, 48), dtype=torch.float32, device=cpu_device)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        y_cpu.backward(grad_out_cpu)
        y_xpu.backward(grad_out_dpcpp)
        grad_cpu = dummy_img.grad
        grad_dpcpp = dummy_img_xpu.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # 3D, CF
        x = torch.rand(1, 3, 18, 32, 32)
        shape = (36, 48, 48)
        dummy_img = (x - x.min()) / (x.max() - x.min()) * 255.0
        dummy_img_xpu = dummy_img.to("xpu")
        dummy_img.requires_grad = True
        dummy_img_xpu.requires_grad = True
        y_cpu = torch.functional.F.interpolate(dummy_img, shape)
        y_xpu = torch.functional.F.interpolate(dummy_img_xpu, shape)
        self.assertEqual(y_cpu, y_xpu.cpu())
        grad_out_cpu = torch.randn(
            (1, 3, 36, 48, 48), dtype=torch.float32, device=cpu_device
        )
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        y_cpu.backward(grad_out_cpu)
        y_xpu.backward(grad_out_dpcpp)
        grad_cpu = dummy_img.grad
        grad_dpcpp = dummy_img_xpu.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # 3D, CL
        x = torch.rand(1, 3, 18, 32, 32).to(memory_format=torch.channels_last_3d)
        shape = (36, 48, 48)
        dummy_img = (x - x.min()) / (x.max() - x.min()) * 255.0
        dummy_img_xpu = dummy_img.to("xpu")
        dummy_img.requires_grad = True
        dummy_img_xpu.requires_grad = True
        y_cpu = torch.functional.F.interpolate(dummy_img, shape)
        y_xpu = torch.functional.F.interpolate(dummy_img_xpu, shape)
        self.assertEqual(y_cpu, y_xpu.cpu())
        grad_out_cpu = torch.randn(
            (1, 3, 36, 48, 48), dtype=torch.float32, device=cpu_device
        ).to(memory_format=torch.channels_last_3d)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        y_cpu.backward(grad_out_cpu)
        y_xpu.backward(grad_out_dpcpp)
        grad_cpu = dummy_img.grad
        grad_dpcpp = dummy_img_xpu.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test__upsample_nearest_exact1d_out_f32(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

    def test__upsample_nearest_exact1d_out_bf16(self, dtype=torch.bfloat16):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.bfloat16)
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

    def test__upsample_nearest_exact1d_out_f16(self, dtype=torch.float16):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.float16)
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

    def test_upsample_nearest_exact1d(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_exact2d_f32(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_exact2d_f16(self, dtype=torch.float16):
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.float16)
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqualIgnoreType(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_exact2d_bf16(self, dtype=torch.bfloat16):
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.bfloat16)
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqualIgnoreType(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_exact3d_f32(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_exact3d_f16(self, dtype=torch.float16):
        input_cpu = torch.randn((2, 3, 5, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.float16)
        scales = [6, 8, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqualIgnoreType(grad_cpu, grad_dpcpp.cpu())

    def test_upsample_nearest_exact3d_bf16(self, dtype=torch.bfloat16):
        input_cpu = torch.randn((2, 3, 5, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.bfloat16)
        scales = [6, 8, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp,
            scale_factor=scales,
            mode="nearest-exact",
            recompute_scale_factor=rsf,
        )
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqualIgnoreType(grad_cpu, grad_dpcpp.cpu())

    def test_q_upsample_nearest_2d(self, dtype=torch.float):
        for shape in [(6, 12), (4, 8)]:
            dtype = torch.qint8
            for zp in [0]:
                scale = 0.04
                x_cpu = torch.randn([2, 2, 4, 8], device=torch.device("cpu"))
                x_gpu = x_cpu.to("xpu")

                q_cpu = torch.quantize_per_tensor(x_cpu, scale, zp, dtype)
                y_cpu = torch.functional.F.interpolate(q_cpu, shape)

                q_gpu = torch.quantize_per_tensor(x_gpu, scale, zp, dtype)
                y_gpu = torch.functional.F.interpolate(q_gpu, shape)

                self.assertEqual(y_cpu, y_gpu)
