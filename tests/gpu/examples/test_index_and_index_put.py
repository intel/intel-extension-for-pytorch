import torch
from torch.testing._internal.common_utils import TestCase
import ipex
import numpy as np
import pytest

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    #@pytest.mark.skipif("not ipex._onedpl_is_enabled()")
    @pytest.mark.skip(reason="skip due to bugs caused by oneDPL and compiler upgrades")
    def test_index_and_index_put(self, dtype=torch.float):
        x_cpu = torch.randn([3, 3], dtype=torch.float, device=cpu_device)
        y_cpu = torch.randn([3, 3], dtype=torch.float, device=cpu_device)
        mask_cpu = y_cpu.gt(0)
        print("x_cpu:")
        print(x_cpu)
        print("mask_cpu:")
        print(mask_cpu)
        print("x_cpu[mask_cpu]:")
        print(x_cpu[mask_cpu])

        # dpcpp part
        x_dpcpp = x_cpu.to("xpu")
        mask_dpcpp = mask_cpu.to("xpu")
        print("mask index:")
        print(mask_dpcpp.to(cpu_device).nonzero())
        print("x_dpcpp[mask_dpcpp]:")
        print(x_dpcpp[mask_dpcpp].to("cpu"))
        self.assertEqual(mask_cpu.nonzero(),
                         mask_dpcpp.to(cpu_device).nonzero())
        self.assertEqual(x_cpu[mask_cpu], x_dpcpp[mask_dpcpp].to(cpu_device))

        # index put
        input = torch.ones([1], dtype=torch.float, device=cpu_device)
        indcies = torch.tensor([0, 0])
        x_cpu[indcies] = input
        print("index_put")
        print(x_cpu)
        x_cpu.index_put_([indcies], input, True)
        print("index_put accmulate=true")
        print(x_cpu)

        input = input.to("xpu")
        indcies = indcies.to("xpu")
        x_dpcpp[indcies] = input
        print("dpcpp index_put")
        print(x_dpcpp.cpu())
        x_dpcpp.index_put_([indcies], input, True)
        print("dpcpp  index_put accmulate=true")
        print(x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
