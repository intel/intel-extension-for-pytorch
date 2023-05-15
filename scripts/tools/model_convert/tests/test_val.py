import unittest
import torch
import model_convert


class TorchCudaAPITests(unittest.TestCase):
    def test_has_cuda(self):
        x = torch.has_cuda
        self.assertEqual(x, True)

    def test_version_cuda(self):
        x = torch.version.cuda
        self.assertEqual(x, "11.7")


if __name__ == "__main__":
    unittest.main()
