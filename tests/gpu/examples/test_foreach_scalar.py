import torch
import random
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class ForeachTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, scalar, device, is_inplace=False):
        input_for_func = []
        for i in input:
            input_for_func.append(i.clone().to(device))
        if is_inplace:
            self.func(input_for_func, scalar)
            return input_for_func
        else:
            return self.func(input_for_func, scalar)

class TestTorchMethod(TestCase):
    def test_foreach_add(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_add)
        cpu = test(x1, scalar, 'cpu')
        xpu = test(x1, scalar, 'xpu')
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_add_)
        cpu_inplace = test_(x1, scalar, 'cpu', is_inplace=True)
        xpu_inplace = test_(x1, scalar, 'xpu', is_inplace=True)
        self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_sub(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_sub)
        cpu = test(x1, scalar, 'cpu')
        xpu = test(x1, scalar, 'xpu')
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_sub_)
        cpu_inplace = test_(x1, scalar, 'cpu', is_inplace=True)
        xpu_inplace = test_(x1, scalar, 'xpu', is_inplace=True)
        self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_mul(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_mul)
        cpu = test(x1, scalar, 'cpu')
        xpu = test(x1, scalar, 'xpu')
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_mul_)
        cpu_inplace = test_(x1, scalar, 'cpu', is_inplace=True)
        xpu_inplace = test_(x1, scalar, 'xpu', is_inplace=True)
        self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_div(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)
        if scalar == 0:
            scalar += 0.1

        test = ForeachTest(torch._foreach_div)
        cpu = test(x1, scalar, 'cpu')
        xpu = test(x1, scalar, 'xpu')
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_div_)
        cpu_inplace = test_(x1, scalar, 'cpu', is_inplace=True)
        xpu_inplace = test_(x1, scalar, 'xpu', is_inplace=True)
        self.result_compare(cpu_inplace, xpu_inplace)

    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())
