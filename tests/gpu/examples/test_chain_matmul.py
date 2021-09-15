import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_chain_matmul_one_matrix(self, dtype=torch.float):
        data_x = torch.randn([4, 5], device=cpu_device)
        data_x_dpcpp = data_x.to(dpcpp_device)
        res = torch.chain_matmul(data_x)
        res_dpcpp = torch.chain_matmul(data_x_dpcpp)
        print("cpu input ", data_x)
        print("cpu chain_matmul result ", res)
        print("dpcpp input ", data_x_dpcpp.cpu())
        print("dpcpp chain_matmul result ", res_dpcpp.cpu())
        self.assertEqual(res.to(cpu_device), res_dpcpp.to(cpu_device))

    def test_chain_matmul_two_matrix(self, dtype=torch.bfloat16):
        data_x = torch.randn([2, 3], device=cpu_device)
        data_x_dpcpp = data_x.to(dpcpp_device)
        data_y = torch.randn([3, 4], device=cpu_device)
        data_y_dpcpp = data_y.to(dpcpp_device)
        res = torch.chain_matmul(data_x, data_y)
        res_dpcpp = torch.chain_matmul(data_x_dpcpp, data_y_dpcpp)
        print("cpu input ", data_x)
        print("cpu chain_matmul result ", res)
        print("dpcpp input ", data_x_dpcpp.cpu())
        print("dpcpp chain_matmul result ", res_dpcpp.cpu())
        self.assertEqual(res.to(cpu_device), res_dpcpp.to(cpu_device))

    def test_chain_matmul_three_matrix(self, dtype=torch.bfloat16):
        data_x = torch.randn([2, 3], device=cpu_device)
        data_x_dpcpp = data_x.to(dpcpp_device)
        data_y = torch.randn([3, 4], device=cpu_device)
        data_y_dpcpp = data_y.to(dpcpp_device)
        data_z = torch.randn([4, 5], device=cpu_device)
        data_z_dpcpp = data_z.to(dpcpp_device)
        res = torch.chain_matmul(data_x, data_y, data_z)
        res_dpcpp = torch.chain_matmul(data_x_dpcpp, data_y_dpcpp, data_z_dpcpp)
        print("cpu input ", data_x)
        print("cpu chain_matmul result ", res)
        print("dpcpp input ", data_x_dpcpp.cpu())
        print("dpcpp chain_matmul result ", res_dpcpp.cpu())
        self.assertEqual(res.to(cpu_device), res_dpcpp.to(cpu_device))

    def test_chain_matmul_four_matrix(self, dtype=torch.bfloat16):
        data_x = torch.randn([2, 3], device=cpu_device)
        data_x_dpcpp = data_x.to(dpcpp_device)
        data_y = torch.randn([3, 4], device=cpu_device)
        data_y_dpcpp = data_y.to(dpcpp_device)
        data_z = torch.randn([4, 5], device=cpu_device)
        data_z_dpcpp = data_z.to(dpcpp_device)
        data_n = torch.randn([5, 6], device=cpu_device)
        data_n_dpcpp = data_n.to(dpcpp_device)
        res = torch.chain_matmul(data_x, data_y, data_z, data_n)
        res_dpcpp = torch.chain_matmul(data_x_dpcpp, data_y_dpcpp, data_z_dpcpp, data_n_dpcpp)
        print("cpu input x", data_x)
        print("cpu input y", data_y)
        print("cpu input z", data_z)
        print("cpu input n", data_n)

        print("dpcpp input x", data_x_dpcpp.cpu())
        print("dpcpp input y", data_y_dpcpp.cpu())
        print("dpcpp input z", data_z_dpcpp.cpu())
        print("dpcpp input n", data_n_dpcpp.cpu())

        print("cpu chain_matmul result ", res)
        print("dpcpp chain_matmul result ", res_dpcpp.cpu())
        self.assertEqual(res.to(cpu_device), res_dpcpp.to(cpu_device))
