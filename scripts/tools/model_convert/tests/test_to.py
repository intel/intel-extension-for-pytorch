import unittest
import torch
import model_convert


class TorchCudaAPITests(unittest.TestCase):
    def test_to(self):
        x = torch.empty(4, 5).to("cuda:0")
        self.assertEqual(x.device.type, "xpu")

    def test_cuda_str(self):
        x = torch.empty(4, 5).cuda("cuda:0")
        self.assertEqual(x.device.type, "xpu")

    def test_cuda_int(self):
        n = torch.cuda.device_count()
        x = torch.empty(4, 5).cuda(n-1)
        self.assertEqual(x.device.type, "xpu")

    def test_cuda_none(self):
        x = torch.empty(4, 5).cuda()
        self.assertEqual(x.device.type, "xpu")


if __name__ == "__main__":
    unittest.main()
