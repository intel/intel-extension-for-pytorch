from torch.testing._internal.common_utils import random_hermitian_matrix, TestCase, random_symmetric_matrix
from torch.testing._internal.common_device_type import dtypes
from torch.testing import make_tensor

import torch 
import warnings
import numpy as np
import itertools
import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_linalg_eigh(self, dtype=torch.float64):

        matrix_cpu = random_hermitian_matrix(3, 1, dtype=dtype, device=cpu_device)
        matrix_gpu = matrix_cpu.clone().detach().to(dpcpp_device)

        L, V = np.linalg.eigh(matrix_cpu.cpu().numpy(), UPLO="L")
        L_xpu, V_xpu = torch.linalg.eigh(matrix_gpu, "L")
        self.assertEqual(L, L_xpu.to(cpu_device))
        self.assertEqual(V, V_xpu.to(cpu_device))

    def test_eigvalsh(self, dtype=torch.float64, device=dpcpp_device):
        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)

            # check the out= variant
            out = torch.empty_like(actual_w)
            ans = torch.linalg.eigvalsh(matrix, UPLO=uplo, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, actual_w)

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)


    def test_eigh_lower_uplo(self, device=dpcpp_device, dtype=torch.float64):
        def run_test(shape, batch, uplo):
            # check lower case uplo
            # use non-symmetric input to check whether uplo argument is working as intended
            matrix = torch.randn(shape, shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            self.assertEqual(abs(actual_v), abs(expected_v))

        uplos = ["u", "l"]
        for uplo in uplos:
            run_test(3, (2, 2), uplo)

    @dtypes(torch.float32, torch.float64)
    def test_eigh_errors_and_warnings(self, dtype=torch.float64, device=dpcpp_device):

        # eigh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigh(t)

        # eigh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be \'L\' or \'U\'"):
                torch.linalg.eigh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be \'L\' or \'U\'"):
                np.linalg.eigh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = random_hermitian_matrix(3, dtype=dtype, device=device)
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        out_w = torch.empty(7, 7, dtype=real_dtype, device=device)
        out_v = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigh(a, out=(out_w, out_v))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[-2].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out_w = torch.empty(0, dtype=real_dtype, device=device)
        out_v = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvectors with dtype Int"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        out_w = torch.empty(0, dtype=torch.int, device=device)
        out_v = torch.empty(0, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvalues with dtype Int"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        # device should match
        if torch.xpu.is_available():
            wrong_device = cpu_device if device != cpu_device else dpcpp_device
            out_w = torch.empty(0, device=wrong_device, dtype=dtype)
            out_v = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=dtype)
            out_v = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))


    @dtypes(torch.float32, torch.float64)
    def test_eigh_non_contiguous(self, dtype=torch.float64, device=dpcpp_device):

        def run_test(matrix, uplo):
            self.assertFalse(matrix.is_contiguous())
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            # sign of eigenvectors is not unique and therefore absolute values are compared
            self.assertEqual(abs(actual_v), abs(expected_v))

        def run_test_permuted(shape, batch, uplo):
            # check for permuted / transposed inputs
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix.transpose(-2, -1)
            run_test(matrix, uplo)

        def run_test_skipped_elements(shape, batch, uplo):
            # check for inputs with skipped elements
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix[::2]
            run_test(matrix, uplo)

        shapes = (3, 5)
        batches = ((4, ), (4, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test_permuted(shape, batch, uplo)
            run_test_skipped_elements(shape, batch, uplo)


    @dtypes(torch.float64)
    def test_eigh_hermitian_grad(self, dtype=torch.float64, device=dpcpp_device):
        def run_test(dims, uplo):
            x = random_hermitian_matrix(dims[-1], *dims[:-2]).requires_grad_()
            w, v = torch.linalg.eigh(x)
            (w.sum() + abs(v).sum()).backward()
            self.assertEqual(x.grad, x.grad.conj().transpose(-1, -2))  # Check the gradient is Hermitian

        for dims, uplo in itertools.product([(3, 3), (1, 1, 3, 3)], ["L", "U"]):
            run_test(dims, uplo)

    @dtypes(torch.float32, torch.float64)
    def test_eigvalsh_non_contiguous(self, device=dpcpp_device, dtype=torch.float64):
        def run_test(matrix, uplo):
            self.assertFalse(matrix.is_contiguous())
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)

        def run_test_permuted(shape, batch, uplo):
            # check for permuted / transposed inputs
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix.transpose(-2, -1)
            run_test(matrix, uplo)

        def run_test_skipped_elements(shape, batch, uplo):
            # check for inputs with skipped elements
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix[::2]
            run_test(matrix, uplo)

        shapes = (3, 5)
        batches = ((4, ), (4, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test_permuted(shape, batch, uplo)
            run_test_skipped_elements(shape, batch, uplo)

    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64)
    def test_eigvals_numpy(self, device=dpcpp_device, dtype=torch.float64):
        def run_test(shape, *, symmetric=True):
            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)
            actual = torch.linalg.eigvals(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eigvals(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected, axis=-1)[::-1]
            expected = np.take_along_axis(expected, ind, axis=-1)

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            ind = np.argsort(actual.cpu().numpy(), axis=-1)[::-1]
            actual_np = actual.cpu().numpy()
            sorted_actual = np.take_along_axis(actual_np, ind, axis=-1)

            self.assertEqual(expected, sorted_actual, exact_dtype=False)

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @dtypes(torch.float32, torch.float64)
    def test_eigvalsh_errors_and_warnings(self, device=dpcpp_device, dtype=torch.float64):
        # eigvalsh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvalsh(t)

        # eigvalsh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be \'L\' or \'U\'"):
                torch.linalg.eigvalsh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be \'L\' or \'U\'"):
                np.linalg.eigvalsh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        real_dtype = t.real.dtype if dtype.is_complex else dtype
        out = torch.empty_like(t).to(real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigvalsh(t, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.eigvalsh(t, out=out)

        # device should match
        if torch.xpu.is_available():
            wrong_device = cpu_device if device != cpu_device else dpcpp_device
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigvalsh(t, out=out)
