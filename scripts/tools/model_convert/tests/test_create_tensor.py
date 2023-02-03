import unittest
import torch
import model_convert


class TorchCudaAPITests(unittest.TestCase):
    def test_empty(self):
        x = torch.empty(2, 3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_ones(self):
        x = torch.ones(2, 3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_logspace(self):
        x = torch.logspace(start=-10, end=10, steps=5,
                           device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_randperm(self):
        x = torch.randperm(4, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_asarray(self):
        a = torch.tensor([1, 2, 3]).to('cuda')
        b = torch.asarray(a)
        self.assertEqual(b.device.type, "xpu")

    def test_ones_like(self):
        input = torch.empty(2, 3)
        x = torch.ones_like(input, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_randn_like(self):
        ref_x = torch.ones(size=(3, 4))
        x = torch.randn_like(ref_x, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    @unittest.skip("not supported by XPU backend")
    def test_sparse_bsc_tensor(self):
        ccol_indices = (0, 1, 2)
        row_indices = (1, 0)
        values = ([[1]], [[2]])
        x = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, size=(
            2, 2), device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_rand_like(self):
        ref_x = torch.ones(size=(3, 4))
        x = torch.rand_like(ref_x, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_empty_like(self):
        a = torch.empty((2, 3), dtype=torch.int32, device='cuda')
        x = torch.empty_like(a)
        self.assertEqual(x.device.type, "xpu")

    def test_full_like(self):
        input = torch.randn(1, 4)
        x = torch.full_like(input, 0.3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    @unittest.skip("not supported by XPU backend")
    def test_sparse_compressed_tensor(self):
        compressed_indices = torch.tensor([0, 2, 4])
        plain_indices = torch.tensor([0, 1, 0, 1])
        values = torch.tensor([1, 2, 3, 4])
        x = torch.sparse_compressed_tensor(
            compressed_indices, plain_indices, values, layout=torch.sparse_csr, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_linspace(self):
        x = torch.linspace(3, 10, steps=5, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    @unittest.skip("not supported by XPU backend")
    def test_sparse_csr_tensor(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (0, 1)
        actual_values = (1, 2)
        x = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(
            2, 2), device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_arange(self):
        x = torch.arange(1, 4, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_kaiser_window(self):
        x = torch.kaiser_window(0, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_randint(self):
        x = torch.randint(4, (5, 2), device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_hamming_window(self):
        x = torch.hamming_window(10, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_rand(self):
        x = torch.rand(5, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    @unittest.skip("not supported by XPU backend")
    def test_sparse_bsr_tensor(self):
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        x = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(
            2, 2), device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_blackman_window(self):
        x = torch.blackman_window(10, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_scalar_tensor(self):
        x = torch.scalar_tensor(4, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_randint_like(self):
        a = torch.rand(4, 5)
        x = torch.randint_like(a, 2, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    @unittest.skip("not supported by XPU backend")
    def test_sparse_csc_tensor(self):
        ccol_indices = (0, 1, 2)
        row_indices = (1, 0)
        values = (1, 2)
        x = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=(
            2, 2), device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_range(self):
        x = torch.range(1, 4, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_full(self):
        x = torch.full((2, 3), 3.141592, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_tril_indices(self):
        x = torch.tril_indices(3, 3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_eye(self):
        x = torch.eye(3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_tensor(self):
        x = torch.tensor([[1., -1.], [1., -1.]], device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_empty_strided(self):
        x = torch.empty_strided((2, 3), (1, 2), device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_triu_indices(self):
        x = torch.triu_indices(3, 3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_hann_window(self):
        x = torch.hann_window(4, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_empty_quantized(self):
        x_q = torch.quantize_per_tensor(
            torch.tensor([3.0]), 1.0, 0, torch.qint32)
        x_pin = torch.empty_quantized(
            [3], x_q, pin_memory=True, dtype=torch.qint32, device=torch.device("cuda"))
        self.assertEqual(x_pin.device.type, "xpu")

    def test_sparse_coo_tensor(self):
        x = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [
                                    1], device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_zeros(self):
        x = torch.zeros(2, 3, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_randn(self):
        x = torch.randn(2, 3, device=torch.device("cuda"))

    def test_zeros_like(self):
        a = torch.rand(3, 4)
        x = torch.zeros_like(a, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")

    def test_bartlett_window(self):
        x = torch.bartlett_window(
            0, periodic=False, device=torch.device("cuda"))
        self.assertEqual(x.device.type, "xpu")


if __name__ == "__main__":
    unittest.main()
