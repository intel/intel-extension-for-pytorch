import torch
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

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
            input_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30), dtype=torch.float32, device=cpu_device)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # #### upsample nearest 2D #####
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30, 40), dtype=torch.float32, device=cpu_device)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

        # #### upsample nearest 1D #####
        input_cpu = torch.randn((2, 3, 2, 5, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8, 1]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 12, 40, 5), dtype=torch.float32, device=cpu_device)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test__upsample_nearest_exact1d_out_f32(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

    def test__upsample_nearest_exact1d_out_bf16(self, dtype=torch.bfloat16):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.bfloat16)
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

    def test__upsample_nearest_exact1d_out_f16(self, dtype=torch.float16):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device)
        input_dpcpp = input_cpu.to("xpu").to(torch.float16)
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
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
            input_cpu, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest-exact', recompute_scale_factor=rsf)
        print("cpu result = ", output_cpu)
        print("dpcpp result = ", output_dpcpp.cpu())
        print("fwd result diff = ", output_cpu - output_dpcpp.cpu())
        self.assertEqualIgnoreType(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.ones_like(output_cpu)
        grad_out_dpcpp = torch.ones_like(output_dpcpp)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        print("x_cpu grad = ", grad_cpu)
        print("x_dpcpp grad = ", grad_dpcpp.cpu())
        print("bwd result diff = ", grad_cpu - grad_dpcpp.cpu())
        self.assertEqualIgnoreType(grad_cpu, grad_dpcpp.cpu())

