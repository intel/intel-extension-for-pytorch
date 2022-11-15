import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa

xpu_device = torch.device("xpu")
cpu_device = torch.device("cpu")
checking_atol = 3e-2
checking_rtol = 3e-2


class TestTorchMethod(TestCase):
    def test_admm(self, dtype=torch.float):

        m1_cpu = torch.randn([3, 4], dtype=dtype)
        m2_cpu = torch.randn([4, 2], dtype=dtype)
        x_cpu = torch.ones([3, 2], dtype=dtype)
        x_cpu2 = torch.ones([1, 2], dtype=dtype)

        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu2 = x_cpu2.to(xpu_device)

        print("cpu addmm_ self", x_cpu)
        x_cpu.addmm_(m1_cpu, m2_cpu)
        print("cpu addmm_ result", x_cpu)

        print("xpu addmm_ self", x_xpu.cpu())
        x_xpu.addmm_(m1_xpu, m2_xpu)
        print("xpu addmm_ result", x_xpu.cpu())
        self.assertEqual(x_cpu, x_xpu.cpu())

        print("cpu addmm_ self", x_cpu2)
        y = x_cpu2.addmm(m1_cpu, m2_cpu)
        print("cpu addmm_ result", y)

        print("xpu addmm_ self", x_xpu2.cpu())
        y_xpu = x_xpu2.addmm(m1_xpu, m2_xpu)
        print("xpu addmm_ result", y_xpu.cpu())
        self.assertEqual(y, y_xpu.cpu())

    # This case is used to check opaque tensor's allocation size in reorder, so it is running in block format
    def test_admm_block(self, dtype=torch.float):
        bs = 64
        hidden = 17
        classNum = 2

        # source1
        src1_cpu = torch.randn(bs, hidden, device=cpu_device, dtype=torch.float32).requires_grad_()
        src1_xpu = torch.randn(bs, hidden, device=xpu_device, dtype=dtype).requires_grad_()

        # source2
        src2_cpu = torch.randn(hidden, classNum, device=cpu_device, dtype=torch.float32).requires_grad_()
        src2_xpu = torch.randn(hidden, classNum, device=xpu_device, dtype=dtype).requires_grad_()

        # bias
        bias_cpu = torch.randn(bs, classNum, device=cpu_device, dtype=torch.float32).requires_grad_()
        bias_xpu = torch.randn(bs, classNum, device=xpu_device, dtype=dtype).requires_grad_()

        # grad
        grad_cpu = torch.randn(bs, classNum, device=cpu_device, dtype=torch.float32).requires_grad_()
        grad_xpu = torch.randn(bs, classNum, device=xpu_device, dtype=dtype).requires_grad_()

        # align all
        src1_xpu.data = src1_cpu.data.to(device=xpu_device, dtype=dtype)
        src2_xpu.data = src2_cpu.data.to(device=xpu_device, dtype=dtype)
        bias_xpu.data = bias_cpu.data.to(device=xpu_device, dtype=dtype)
        grad_xpu.data = grad_cpu.data.to(device=xpu_device, dtype=dtype)

        # forward
        dst_cpu = torch.addmm(bias_cpu, src1_cpu, src2_cpu)
        dst_xpu = torch.addmm(bias_xpu, src1_xpu, src2_xpu)

        # backward
        dst_cpu.backward(grad_cpu)
        dst_xpu.backward(grad_xpu)

        # check output
        self.assertEqual(dst_cpu, dst_xpu.to(device=cpu_device, dtype=torch.float),
                         atol=checking_atol, rtol=checking_rtol)

        # check src1 grad
        self.assertEqual(src1_cpu.grad, src1_xpu.grad.to(device=cpu_device, dtype=torch.float),
                         atol=checking_atol, rtol=checking_rtol)

        # check src2 grad
        self.assertEqual(src2_cpu.grad, src2_xpu.grad.to(device=cpu_device, dtype=torch.float),
                         atol=checking_atol, rtol=checking_rtol)

        # check bias grad
        self.assertEqual(bias_cpu.grad, bias_xpu.grad.to(device=cpu_device, dtype=torch.float),
                         atol=checking_atol, rtol=checking_rtol)
