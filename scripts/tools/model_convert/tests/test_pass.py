import unittest
import torch
import model_convert


class TorchCudaAPITests(unittest.TestCase):
    def test_is_nccl_available(self):
        x = torch.distributed.is_nccl_available()
        self.assertEqual(x, True)

    def test_is_bf16_supported(self):
        x = torch.cuda.is_bf16_supported()
        self.assertEqual(x, True)


if __name__ == "__main__":
    unittest.main()
