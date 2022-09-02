import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# [Note:] This test is ported from experiment/test_linalg.py test_inv_errors_and_warnings().
# Retrive out this test, only because the singular error is not throw. Other tests could be passed.
class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_inv_with_errors(self, dtype=torch.float32, device=dpcpp_device):
        """ This function test raising exceptions.

        """
        a = torch.randn(2, 3, 4, 3, dtype=dtype, device=dpcpp_device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.inv(a)

        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.inv(a)

        # # The following singular input could not raise error. This is the same behavior with torch.inverse
        # # The root cause is because the calling of apply_inverse_dpcpp_ does not raise the error.
        # def run_test_singular_input(batch_dim, n):
        #       a = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
        #       a[n, -1, -1] = 0
        #     #   with self.assertRaisesRegex(RuntimeError, rf"\(Batch element {n}\): The diagonal element 3 is zero"):
        #       torch.linalg.inv(a)

        # for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
        #       run_test_singular_input(*params)


    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_inverse(self, dtype=torch.float):
        def _validate(A, A_):
            self.assertEqual(torch.matmul(A, A_), torch.eye(A.size(-1)).expand_as(A),
                             rtol=1.3e-6, atol=0.005)

        for size in [(3, 3), (2, 3, 3), (128, 64, 64)]:
            A = torch.randn(size, dtype=dtype)

            # CPU
            A_cpu = A.to('cpu')
            Ai_cpu = torch.inverse(A_cpu)
            _validate(A_cpu, Ai_cpu)

            # XPU
            A_xpu = A.clone().to('xpu')
            Ai_xpu = torch.inverse(A_xpu)
            _validate(A_xpu.cpu(), Ai_xpu.cpu())
            A_xpu2 = A.clone().to('xpu')
            Ai_xpu2 = torch.linalg.inv(A_xpu2)
            _validate(A_xpu2.cpu(), Ai_xpu2.cpu())

            self.assertEqual(Ai_xpu.cpu(), Ai_xpu2.cpu())
