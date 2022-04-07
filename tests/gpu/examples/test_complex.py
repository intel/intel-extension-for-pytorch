import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_complex_float(self, dtype=torch.float):
        img = torch.randn([5, 5])
        real = torch.randn([5, 5])
        y_cpu = torch.complex(real, img)
        img_xpu = img.to("xpu")
        real_xpu = real.to("xpu")
        y_xpu = torch.complex(real_xpu, img_xpu)

        self.assertEqual(y_cpu, y_xpu.to("cpu"))

    def test_complex_double(self, dtype=torch.double):
        img = torch.randn([5, 5], dtype=dtype)
        real = torch.randn([5, 5], dtype=dtype)
        y_cpu = torch.complex(real, img)
        img_xpu = img.to("xpu")
        real_xpu = real.to("xpu")
        y_xpu = torch.complex(real_xpu, img_xpu)

        self.assertEqual(y_cpu, y_xpu.to("cpu"))

    def test_real_imag(self, dtype=torch.float):
        input = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input_xpu = input.to("xpu")
        self.assertEqual(input.real, input_xpu.real.to("cpu"))
        self.assertEqual(input.imag, input_xpu.imag.to("cpu"))

    def test_conj(self, dtype=torch.float):
        input = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input_xpu = input.to("xpu")
        self.assertEqual(input.conj(), input_xpu.conj().to("cpu"))
