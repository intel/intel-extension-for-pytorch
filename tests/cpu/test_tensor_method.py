import torch
import unittest
from common_utils import TestCase


class TestTesorMethod(TestCase):
    def test_numpy(self):
        # float tensor, numpy array will share memory with torch tensor.
        x = torch.randn(2, 3)
        y = torch.from_numpy(x.numpy())
        self.assertEqual(x, y)
        self.assertEqual(x.data_ptr(), y.data_ptr())
        # bfloat16 tensor, numpy array will not share memory with torch tensor.
        x = torch.randn(2, 3).bfloat16()
        y = torch.from_numpy(x.numpy())
        self.assertEqual(x, y.bfloat16())
        self.assertNotEqual(x.data_ptr(), y.data_ptr())


if __name__ == "__main__":
    test = unittest.main()
