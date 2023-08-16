import unittest
import torch
import model_convert


class TorchCudaAPITests(unittest.TestCase):
    def test_is_current_stream_capturing(self):
        x = torch.cuda.is_current_stream_capturing()
        self.assertEqual(x, False)


if __name__ == "__main__":
    unittest.main()
