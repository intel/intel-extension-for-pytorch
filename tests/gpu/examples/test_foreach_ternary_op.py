import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import (TestCase)
from typing import List
import numpy as np

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class ForeachPointWiseScalarListTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, input1, input2, scalarlist, device):
        input_tensor_for_func: List[torch.Tensor] = []
        input_tensor1_for_func: List[torch.Tensor] = []
        input_tensor2_for_func: List[torch.Tensor] = []
        input_scalar_for_func = []
        for i in range(len(input)):
            input_tensor_for_func.append(input[i].clone().to(device))
            input_tensor1_for_func.append(input1[i].clone().to(device))
            input_tensor2_for_func.append(input2[i].clone().to(device))
            input_scalar_for_func.append(scalarlist[i])
        return self.func(input_tensor_for_func, input_tensor1_for_func, input_tensor2_for_func, input_scalar_for_func)

class ForeachPointWiseScalarTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, input1, input2, scalar, device):
        input_tensor_for_func: List[torch.Tensor] = []
        input_tensor1_for_func: List[torch.Tensor] = []
        input_tensor2_for_func: List[torch.Tensor] = []
        input_scalar_for_func = []
        for i in range(len(input)):
            input_tensor_for_func.append(input[i].clone().to(device))
            input_tensor1_for_func.append(input1[i].clone().to(device))
            input_tensor2_for_func.append(input2[i].clone().to(device))
        return self.func(input_tensor_for_func, input_tensor1_for_func, input_tensor2_for_func, scalar)

class TestTorchMethod(TestCase):

    def test_foreach_addcmul_scalar(self, dtype=torch.float):
        shape = [5, 8]
        x1 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x3 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        scalar = np.random.random()

        test = ForeachPointWiseScalarTest(torch._foreach_addcmul)
        cpu = test(x1, x2, x3, scalar, 'cpu')
        xpu = test(x1, x2, x3, scalar, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_addcdiv_scalar(self, dtype=torch.float):
        shape = [5, 8]
        x1 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x3 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        scalar = np.random.random()

        test = ForeachPointWiseScalarTest(torch._foreach_addcdiv)
        cpu = test(x1, x2, x3, scalar, 'cpu')
        xpu = test(x1, x2, x3, scalar, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_addcmul_scalarlist(self, dtype=torch.float):
        shape = [5, 8]
        x1 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x3 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        scalarlist = np.random.randn(250)

        test = ForeachPointWiseScalarListTest(torch._foreach_addcmul)
        cpu = test(x1, x2, x3, scalarlist, 'cpu')
        xpu = test(x1, x2, x3, scalarlist, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_addcdiv_scalarlist(self, dtype=torch.float):
        shape = [5, 8]
        x1 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        x3 = [torch.randn(shape, dtype=torch.float) for _ in range(250)]
        scalarlist = np.random.randn(250)

        test = ForeachPointWiseScalarListTest(torch._foreach_addcdiv)
        cpu = test(x1, x2, x3, scalarlist, 'cpu')
        xpu = test(x1, x2, x3, scalarlist, 'xpu')
        self.result_compare(cpu, xpu)

    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())
