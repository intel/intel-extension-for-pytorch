import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class ForeachBinaryTensorTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, other, device):
        input_tensor_for_func = []
        input_other_for_func = []
        for i in range(len(input)):
            input_tensor_for_func.append(input[i].clone().to(device))
            input_other_for_func.append(other[i].clone().to(device))
        return self.func(input_tensor_for_func, input_other_for_func)

class ForeachBinaryTensorTestInplace:
     def __init__(self, func):
         self.func = func

     def __call__(self, input, other, device):
         input_tensor_for_func = []
         input_other_for_func = []
         for i in range(len(input)):
             input_tensor_for_func.append(input[i].clone().to(device))
             input_other_for_func.append(other[i].clone().to(device))
         self.func(input_tensor_for_func, input_other_for_func)
         return input_tensor_for_func


class TestTorchMethod(TestCase):
    def test_foreach_mul_tensor(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachBinaryTensorTest(torch._foreach_mul)
        cpu1 = test(x1, x2, "cpu")
        xpu1 = test(x1, x2, "xpu")

        test = ForeachBinaryTensorTestInplace(torch._foreach_mul_)
        cpu2 = test(x1, x2, "cpu")
        xpu2 = test(x1, x2, "xpu")

        self.result_compare(cpu1, xpu1)
        self.result_compare(cpu2, xpu2)
        self.result_compare(cpu1, cpu2)
        self.result_compare(xpu1, xpu2)

    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())
