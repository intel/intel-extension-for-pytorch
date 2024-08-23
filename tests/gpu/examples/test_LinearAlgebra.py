import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import pytest
import itertools
import numpy as np

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    @pytest.mark.skip(
        reason="PT2.5: Double and complex datatype matmul is not supported in oneDNN",
    )
    def test_cholesky_inverse(self, dtype=torch.float, device=dpcpp_device):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, upper, contiguous, dtype):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            actual_inverse = torch.cholesky_inverse(A, upper)
            expected_inverse = torch.cholesky_inverse(A, upper)

            self.assertEqual(actual_inverse, expected_inverse)

        shapes = (3, 5)
        batches = ((3,), (2, 2))
        dtypes = [torch.float, torch.cfloat]
        for shape, batch, upper, contiguous, dtype in list(
            itertools.product(shapes, batches, (True, False), (True, False), dtypes)
        ):
            run_test(shape, batch, upper, contiguous, dtype)

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_geqrf(self, dtype=torch.float):
        def run_test(shape, dtype):
            # numpy.linalg.qr with mode = 'raw' computes the same operation as torch.geqrf
            # so this test compares against that function
            from torch.testing import make_tensor

            A = make_tensor(shape, dtype=dtype, device=dpcpp_device)
            # numpy.linalg.qr doesn't work with batched input
            m, n = A.shape[-2:]
            tau_size = "n" if m > n else "m"
            np_dtype = A.cpu().numpy().dtype
            ot = [np_dtype, np_dtype]
            numpy_geqrf_batched = np.vectorize(
                lambda x: np.linalg.qr(x, mode="raw"),
                otypes=ot,
                signature=f"(m,n)->(n,m),({tau_size})",
            )

            expected = numpy_geqrf_batched(A.cpu())
            actual = torch.geqrf(A)

            # numpy.linalg.qr returns transposed result
            self.assertEqual(expected[0].swapaxes(-2, -1), actual[0])
            self.assertEqual(expected[1], actual[1])

        batches = [(), (0,), (2,), (2, 1)]
        ns = [5, 2, 0]
        dtypes = [torch.float, torch.cfloat]
        for batch, (m, n), dtype in itertools.product(
            batches, itertools.product(ns, ns), dtypes
        ):
            run_test((*batch, m, n), dtype)

    def test_ger(self, dtype=torch.float):
        v1 = torch.arange(1.0, 5.0, device=cpu_device)
        v2 = torch.arange(1.0, 4.0, device=cpu_device)

        A12 = torch.ger(v1, v2)
        print("cpu v1 ", v1.to(cpu_device))
        print("cpu v2 ", v2.to(cpu_device))
        print("cpu A12 ", A12.to(cpu_device))

        A21 = torch.ger(v2, v1)
        print("cpu v1 ", v2.to(cpu_device))
        print("cpu v2 ", v1.to(cpu_device))
        print("cpu A21 ", A21.to(cpu_device))

        v1 = torch.arange(1.0, 5.0, device=dpcpp_device)
        v2 = torch.arange(1.0, 4.0, device=dpcpp_device)

        A12_dpcpp = torch.ger(v1, v2)
        print("dpcpp v1 ", v1.to(cpu_device))
        print("dpcpp v2 ", v2.to(cpu_device))
        print("dpcpp A12_dpcpp ", A12_dpcpp.to(cpu_device))

        A21_dpcpp = torch.ger(v2, v1)
        print("dpcpp v1 ", v2.to(cpu_device))
        print("dpcpp v2 ", v1.to(cpu_device))
        print("dpcpp A21_dpcpp ", A21_dpcpp.to(cpu_device))

        self.assertEqual(A12, A12_dpcpp.to(cpu_device))
        self.assertEqual(A21, A21_dpcpp.to(cpu_device))

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_addr(self, dtype=torch.float):
        x1_cpu = torch.randn(3, dtype=torch.float)
        x2_cpu = torch.randn(2, dtype=torch.float)
        M_cpu = torch.randn(3, 2, dtype=torch.float)
        y_cpu = torch.addr(M_cpu, x1_cpu, x2_cpu)

        x1_xpu = x1_cpu.to(dpcpp_device)
        x2_xpu = x2_cpu.to(dpcpp_device)
        M_xpu = M_cpu.to(dpcpp_device)
        y_xpu = torch.addr(M_xpu, x1_xpu, x2_xpu)

        self.assertEqual(y_cpu, y_xpu.cpu())

        y_cpu = M_cpu.addr(x1_cpu, x2_cpu)
        y_xpu = M_xpu.addr(x1_xpu, x2_xpu)
        self.assertEqual(y_cpu, y_xpu.cpu())

        M_cpu.addr_(x1_cpu, x2_cpu)
        M_xpu.addr_(x1_xpu, x2_xpu)
        self.assertEqual(M_cpu, M_xpu.cpu())
