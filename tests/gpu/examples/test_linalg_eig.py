import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase
import pytest
import numpy as np

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):

    @pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
    def test_linalg_eig(self, dtype=torch.complex128):
        input = torch.randn(2, 2, dtype=torch.complex128)
        input_xpu = input.to(dpcpp_device)

        L, V = torch.linalg.eig(input)
        L_xpu, V_xpu = torch.linalg.eig(input_xpu)
        self.assertEqual(L, L_xpu.to(cpu_device))
        self.assertEqual(V, V_xpu.to(cpu_device))

    def test_tensordot(self, device="xpu"):
        a = torch.arange(60., device=device).reshape(3, 4, 5)
        b = torch.arange(24., device=device).reshape(4, 3, 2)
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=([1, 0], [0, 1])))
        self.assertEqual(c, cn)

        cout = torch.zeros((5, 2), device=device)
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        self.assertEqual(c, cout)

        a = torch.randn(2, 3, 4, 5, device=device)
        b = torch.randn(4, 5, 6, 7, device=device)
        c = torch.tensordot(a, b, dims=2).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=2))

        with self.assertRaisesRegex(RuntimeError, "expects dims >= 0"):
            torch.tensordot(a, b, dims=-1)

        self.assertEqual(c, cn)
        c = torch.tensordot(a, b).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        self.assertEqual(c, cn)

        a = torch.tensordot(torch.tensor(0.), torch.tensor(0.), 0)
        an = torch.from_numpy(np.tensordot(np.zeros((), dtype=np.float32), np.zeros((), dtype=np.float32), 0))
        self.assertEqual(a, an)
