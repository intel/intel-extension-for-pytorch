import torch
import intel_extension_for_pytorch # noqa
import warnings
import numpy as np
from torch.testing._internal.common_utils import (TestCase, freeze_rng_state)
from torch.testing._internal.common_device_type import dtypes

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_tensorinv_empty(self, device=dpcpp_device, dtype=torch.float64):
        for ind in range(1, 4):
            # Check for empty inputs. NumPy does not work for these cases.
            a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
            a_inv = torch.linalg.tensorinv(a, ind=ind)
            self.assertEqual(a_inv.shape, a.shape[ind:] + a.shape[:ind])

    @dtypes(torch.float, torch.double)
    def test_linalg_matrix_exp_batch(self, device=dpcpp_device, dtype=torch.float32):

        def run_test(*n):
            tensors_batch = torch.zeros(n, dtype=dtype, device=device)
            tensors_batch = tensors_batch.view(-1, n[-2], n[-1])

            num_matrices = tensors_batch.size(0)
            tensors_list = []
            for i in range(num_matrices):
                tensors_list.append(torch.randn(n[-2], n[-1], dtype=dtype, device=device))

            for i in range(num_matrices):
                tensors_batch[i, ...] = tensors_list[i]

            tensors_exp_map = (torch.linalg.matrix_exp(x) for x in tensors_list)
            tensors_exp_batch = torch.linalg.matrix_exp(tensors_batch)

            for i, tensor_exp in enumerate(tensors_exp_map):
                # compute on cpu avoid unexpected error on atsm
                self.assertEqual(tensors_exp_batch[i, ...].to('cpu'), tensor_exp)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)

    @dtypes(torch.complex64)
    def test_linalg_matrix_exp_no_warnings(self, device=dpcpp_device, dtype=torch.complex64):
        # this tests https://github.com/pytorch/pytorch/issues/80948
        with freeze_rng_state():
            torch.manual_seed(42)
            tens = 0.5 * torch.randn(10, 3, 3, dtype=dtype, device=device)
            tens = (0.5 * (tens.transpose(-1, -2) + tens))
            with warnings.catch_warnings(record=True) as w:
                tens.imag = torch.matrix_exp(tens.imag)
                self.assertFalse(len(w))

    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_linalg_matrix_exp_boundary_cases(self, device=dpcpp_device, dtype=torch.float32):
        expm = torch.linalg.matrix_exp

        with self.assertRaisesRegex(RuntimeError, "Expected a floating point or complex tensor"):
            expm(torch.randn(3, 3).type(torch.int))

        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            expm(torch.randn(3))

        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            expm(torch.randn(3, 2, 1))

        # check 1x1 matrices
        x = torch.randn(3, 3, 1, 1)
        self.assertEqual(expm(x), x.exp())

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_matrix_exp_analytic(self, device=dpcpp_device, dtype=torch.float32):
        expm = torch.linalg.matrix_exp
        # check zero matrix
        x = torch.zeros(20, 20, dtype=dtype, device=device)
        self.assertTrue((expm(x) == torch.eye(20, 20, dtype=dtype, device=device)).all().item())

        def normalize_to_1_operator_norm(sample, desired_norm):
            sample_norm, _ = sample.abs().sum(-2).max(-1)
            sample_to_1_norm = sample / sample_norm.unsqueeze(-1).unsqueeze(-1)
            return sample_to_1_norm * desired_norm

        def gen_good_cond_number_matrices(*n):
            """
            Generates a diagonally-domimant matrix
            with the eigenvalues centered at 1
            and the radii at most (n[-1] - 1) / (n[-2] ** 2)
            """
            identity = torch.eye(n[-2], n[-1], dtype=dtype, device=device).expand(*n)
            x = torch.rand(*n, dtype=dtype, device=device) / (n[-1] ** 2)
            x = (x - x * identity) + identity
            return x

        def run_test(*n):
            if dtype == torch.float:
                thetas = [
                    1.192092800768788e-07,  # deg 1
                    5.978858893805233e-04,  # deg 2
                    5.116619363445086e-02,  # deg 4
                    5.800524627688768e-01,  # deg 8
                    1.461661507209034e+00,  # deg 12
                    3.010066362817634e+00   # deg 18
                ]
            else:  # if torch.double
                thetas = [
                    2.220446049250313e-16,  # deg 1
                    2.580956802971767e-08,  # deg 2
                    3.397168839976962e-04,  # deg 4
                    4.991228871115323e-02,  # deg 8
                    2.996158913811580e-01,  # deg 12
                    1.090863719290036e+00   # deg 18
                ]

            # generate input
            q = gen_good_cond_number_matrices(*n)
            q_ = q.cpu().numpy()
            qinv = torch.inverse(q)
            qinv_ = qinv.cpu().numpy()
            d = torch.randn(n[:-1], dtype=dtype, device=device)
            x = torch.from_numpy(
                np.matmul(q_, np.matmul(torch.diag_embed(d).cpu().numpy(), qinv_))).to(device)
            x_norm, _ = x.abs().sum(-2).max(-1)

            # test simple analytic whatever norm generated
            mexp = expm(x)
            mexp_analytic = np.matmul(
                q_,
                np.matmul(
                    torch.diag_embed(d.exp()).cpu().numpy(),
                    qinv_
                )
            )
            self.assertEqual(mexp, mexp_analytic, atol=1e-3, rtol=0.0)

            # generate norms to test different degree expansions
            sample_norms = []
            for i in range(len(thetas) - 1):
                sample_norms.append(0.5 * (thetas[i] + thetas[i + 1]))
            sample_norms = [thetas[0] / 2] + sample_norms + [thetas[-1] * 2]

            # matrices to equal norm
            for sample_norm in sample_norms:
                x_normalized = normalize_to_1_operator_norm(x, sample_norm)

                mexp = expm(x_normalized)
                mexp_analytic = np.matmul(
                    q_,
                    np.matmul(
                        torch.diag_embed((d / x_norm.unsqueeze(-1) * sample_norm).exp()).cpu().numpy(),
                        qinv_
                    )
                )
                self.assertEqual(mexp, mexp_analytic, atol=1e-3, rtol=0.0)

        # single matrix
        run_test(2, 2)
        run_test(3, 3)
        run_test(4, 4)
        run_test(5, 5)
        run_test(100, 100)
        run_test(200, 200)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)
        run_test(3, 100, 100)
        run_test(3, 200, 200)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)
        run_test(3, 3, 100, 100)
        run_test(3, 3, 200, 200)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_matrix_exp_compare_with_taylor(self, device=dpcpp_device, dtype=torch.float32):

        def normalize_to_1_operator_norm(sample, desired_norm):
            sample_norm, _ = sample.abs().sum(-2).max(-1)
            sample_to_1_norm = sample / sample_norm.unsqueeze(-1).unsqueeze(-1)
            return sample_to_1_norm * desired_norm

        def gen_good_cond_number_matrices(*n):
            """
            Generates a diagonally-domimant matrix
            with the eigenvalues centered at 1
            and the radii at most (n[-1] - 1) / (n[-2] ** 2)
            """
            identity = torch.eye(n[-2], n[-1], dtype=dtype, device=device).expand(*n)
            x = torch.rand(*n, dtype=dtype, device=device) / (n[-1] ** 2)
            x = (x - x * identity) + identity
            return x

        def get_taylor_approximation(a, deg):
            a_ = a.cpu().numpy()
            identity = torch.eye(a.size(-2), a.size(-1), dtype=dtype, device=device).expand_as(a)
            res = identity.cpu().numpy()
            taylor_term = identity.cpu().numpy()

            for i in range(1, deg + 1):
                taylor_term = np.matmul(a_, taylor_term) / i
                res = res + taylor_term

            return res

        def scale_square(a, deg):
            if a.abs().pow(2).sum().sqrt() < 1.0:
                return get_taylor_approximation(a, 12)
            else:
                s = int(torch.log2(a.abs().pow(2).sum().sqrt()).ceil().item())
                b = a / (2 ** s)
                b = get_taylor_approximation(b, 18)
                for _ in range(s):
                    b = np.matmul(b, b)
                return torch.from_numpy(b).to(a.device)

        def run_test(*n):
            degs = [1, 2, 4, 8, 12, 18]
            if dtype == torch.float:
                thetas = [
                    1.192092800768788e-07,  # deg 1
                    5.978858893805233e-04,  # deg 2
                    5.116619363445086e-02,  # deg 4
                    5.800524627688768e-01,  # deg 8
                    1.461661507209034e+00,  # deg 12
                    3.010066362817634e+00   # deg 18
                ]
            else:  # if torch.double
                thetas = [
                    2.220446049250313e-16,  # deg 1
                    2.580956802971767e-08,  # deg 2
                    3.397168839976962e-04,  # deg 4
                    4.991228871115323e-02,  # deg 8
                    2.996158913811580e-01,  # deg 12
                    1.090863719290036e+00   # deg 18
                ]

            # generate norms to test different degree expansions
            sample_norms = []
            for i in range(len(thetas) - 1):
                sample_norms.append(0.5 * (thetas[i] + thetas[i + 1]))
            sample_norms = [thetas[0] / 2] + sample_norms + [thetas[-1] * 2]
            degs = [degs[0]] + degs

            for sample_norm, deg in zip(sample_norms, degs):
                x = gen_good_cond_number_matrices(*n)
                x = normalize_to_1_operator_norm(x, sample_norm)

                mexp = torch.linalg.matrix_exp(x)
                mexp_taylor = scale_square(x, deg)

                self.assertEqual(mexp.to('cpu'), mexp_taylor, atol=1e-2, rtol=0.0)

        # single matrix
        run_test(2, 2)
        run_test(3, 3)
        run_test(4, 4)
        run_test(5, 5)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)