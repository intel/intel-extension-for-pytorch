import torch
import torch_ipex
from torch.testing._internal.common_utils import TestCase
import pytest

cpu_device = torch.device('cpu')
dpcpp_device = torch.device('dpcpp')


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("torch_ipex._double_kernel_disabled()")     
    def test_logical_xor(self, dtype=torch.float):
        input1 = torch.tensor(
            [0, 1, 10, 0], device=torch.device("cpu"), dtype=torch.int8)
        input2 = torch.tensor(
            [4, 0, 1, 0], device=torch.device("cpu"), dtype=torch.int8)

        # TODO: check for diferent dtype
        array1 = [input1, input1.double()]  # , input1.double()]
        array2 = [input2, input2.double()]  # , input2]

        for i in range(len(array1)):
            print("Testing logical_xor on", array1[i], "and", array2[i])
            result = torch.logical_xor(array1[i], array2[i])
            print("CPU result:")
            print(result)

            print("SYCL result:")
            print("--torch.logical_xor--")
            input1_dpcpp = array1[i].to("dpcpp")
            input2_dpcpp = array2[i].to("dpcpp")
            result_dpcpp1 = torch.logical_xor(input1_dpcpp, input2_dpcpp)
            print(result_dpcpp1.to("cpu"))

            print("--tensor.logical_xor--")
            result_dpcpp2 = input1_dpcpp.logical_xor(input2_dpcpp)
            print(result_dpcpp2.to("cpu"))
            print("--tensor.logical_xor_--")
            result_dpcpp3 = input2_dpcpp.logical_xor_(input1_dpcpp)
            print(result_dpcpp3.to("cpu"))
            print("\n")
            self.assertEqual(result, result_dpcpp1.cpu())
            self.assertEqual(result, result_dpcpp2.cpu())
            self.assertEqual(result, result_dpcpp3.cpu())

        print("Additional Test with out=torch.empty(4, dtype=torch.int8)")
        print("on", input1, input2)
        out = torch.empty(4, dtype=torch.int8)
        result = torch.logical_xor(input1, input2, out=out)
        print("CPU result:")
        print(result)
        print("SYCL result:")
        result_dpcpp = torch.logical_xor(
            input1.to("dpcpp"), input2.to("dpcpp"), out=out.to("dpcpp"))
        print(result_dpcpp.to("cpu"))
        self.assertEqual(result, result_dpcpp)
