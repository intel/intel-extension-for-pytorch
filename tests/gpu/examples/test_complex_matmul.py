import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

@pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
class TestTorchMethod(TestCase):
    def test_addmm(self):
        mat1 = torch.complex(torch.randn([4, 2]), torch.randn([4, 2]))
        mat2 = torch.complex(torch.randn([2, 4]), torch.randn([2, 4]))
        add = torch.complex(torch.randn([4, 4]), torch.randn([4, 4]))

        mat1_xpu = mat1.to("xpu")
        mat2_xpu = mat2.to("xpu")
        add_xpu = add.to("xpu")

        output = torch.addmm(add, mat1, mat2)
        output_xpu = torch.addmm(add_xpu, mat1_xpu, mat2_xpu)
        self.assertEqual(output, output_xpu.to("cpu"))

    def test_matmul(self):
        mat1 = torch.complex(torch.randn(5, 1, 3), torch.randn(5, 1, 3))
        mat2 = torch.complex(torch.randn(3, 5), torch.randn(3, 5))

        mat1_xpu = mat1.to("xpu")
        mat2_xpu = mat2.to("xpu")

        output = mat1 @ mat2
        output_xpu = mat1_xpu @ mat2_xpu
        self.assertEqual(output, output_xpu.to("cpu"))

    def test_bmm(self):
        mat1 = torch.complex(torch.randn(2, 4, 2), torch.randn(2, 4, 2))
        mat2 = torch.complex(torch.randn(2, 2, 4), torch.randn(2, 2, 4))

        mat1_xpu = mat1.to("xpu")
        mat2_xpu = mat2.to("xpu")

        output = mat1 @ mat2
        output_xpu = mat1_xpu @ mat2_xpu
        self.assertEqual(output, output_xpu.to("cpu"))

    def test_bmm_no_contiguous(self):
        mat1 = torch.complex(torch.randn(2, 2, 3), torch.randn(2, 2, 3)).transpose(1, 2)
        mat2 = torch.complex(torch.randn(2, 4, 2), torch.randn(2, 4, 2)).transpose(1, 2)

        mat1_xpu = mat1.to("xpu")
        mat2_xpu = mat2.to("xpu")

        output = mat1 @ mat2
        output_xpu = mat1_xpu @ mat2_xpu
        self.assertEqual(output, output_xpu.to("cpu"))
