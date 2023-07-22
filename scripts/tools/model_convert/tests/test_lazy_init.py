import unittest
import torch
import model_convert


class TorchCudaAPITests(unittest.TestCase):
    def test_is_initialized(self):
        if torch.cuda.is_available():
            torch.cuda.init()
            x = torch.cuda.is_initialized()
            self.assertEqual(x, True)

    def test_is_in_bad_fork(self):
        x = torch.cuda._is_in_bad_fork()
        self.assertEqual(x, False)

    def test_lazy_init(self):
        torch.cuda._lazy_init()


if __name__ == "__main__":
    unittest.main()
