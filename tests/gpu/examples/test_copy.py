import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_copy_d_to_h_no_contiguous(self, dtype=torch.float):
        input = torch.randn([10000, 64, 3])
        input_xpu = input.to(dpcpp_device)
        output = torch.as_strided(input, (256, 64, 3), (640, 1, 64))
        output_xpu = torch.as_strided(input_xpu, (256, 64, 3), (640, 1, 64))
        self.assertEqual(output, output_xpu.cpu())

    def test_copy_D2D_8d(self, dtype=torch.float):
        input = torch.rand(256, 8, 8, 3, 3, 4, 5, 6, device=cpu_device)
        input_xpu = input.to(dpcpp_device)
        output = input.transpose(3, 4)
        output_xpu = input_xpu.transpose(3, 4)
        output.contiguous()
        output_xpu.contiguous()
        self.assertEqual(output, output_xpu.cpu())

    def test_copy_quantize_tensor(self, dtype=torch.qint8):
        qtensor1 = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0], device=dpcpp_device), 0.1, 10, dtype=dtype)
        qtensor2 = qtensor1.clone()
        self.assertEqual(qtensor1.to(cpu_device), qtensor2.to(cpu_device))

    def test_copy_big_numel(self, dtype=torch.float):
        tensor1 = torch.rand([8, 2048, 50304], dtype=dtype, device=dpcpp_device)
        tensor2 = tensor1.clone()
        self.assertEqual(tensor1.to(cpu_device), tensor2.to(cpu_device))
