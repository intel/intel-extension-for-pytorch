import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_gemv(self, dtype=torch.float):
        mat = torch.randn((3, 2), device=cpu_device)
        vec = torch.randn((2), device=cpu_device)

        print("mat: ", mat)
        print("vec: ", vec)

        res = torch.mv(mat, vec)
        res_1 = mat.mv(vec)

        mat_dpcpp = mat.to(dpcpp_device)
        vec_dpcpp = vec.to(dpcpp_device)
        res_dpcpp = torch.mv(mat_dpcpp, vec_dpcpp)
        res_dpcpp_1 = mat_dpcpp.mv(vec_dpcpp)

        self.assertEqual(res, res_dpcpp.to(cpu_device))
        self.assertEqual(res, res_dpcpp_1.to(cpu_device))
        self.assertEqual(res_1, res_dpcpp_1.to(cpu_device))

