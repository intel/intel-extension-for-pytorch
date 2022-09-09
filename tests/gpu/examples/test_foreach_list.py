import torch
import random
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")



class ForeachTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input1, input2, device, is_inplace=False, scalar=None):
        input1_for_func = []
        input2_for_func = []
        for i in input1:
            input1_for_func.append(i.clone().to(device))
        for i in input2:
            input2_for_func.append(i.clone().to(device))
        if is_inplace:
            if scalar is None:
                self.func(input1_for_func, input2_for_func)
            else:
                self.func(input1_for_func, input2_for_func, alpha=scalar)
            return input1_for_func
        else:
            if scalar is None:
                return self.func(input1_for_func, input2_for_func)
            else:
                return self.func(input1_for_func, input2_for_func, alpha=scalar)

class TestTorchMethod(TestCase):
    def test_foreach_add(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_add)
        cpu = test(x1, x2, 'cpu', is_inplace=False, scalar=scalar)
        xpu = test(x1, x2, 'xpu', is_inplace=False, scalar=scalar)
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_add_)
        cpu_inplace = test_(x1, x2, 'cpu', is_inplace=True, scalar=scalar)
        xpu_inplace = test_(x1, x2, 'xpu', is_inplace=True, scalar=scalar)
        self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_sub(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_sub)
        cpu = test(x1, x2, 'cpu', is_inplace=False, scalar=scalar)
        xpu = test(x1, x2, 'xpu', is_inplace=False, scalar=scalar)
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_sub_)
        cpu_inplace = test_(x1, x2, 'cpu', is_inplace=True, scalar=scalar)
        xpu_inplace = test_(x1, x2, 'xpu', is_inplace=True, scalar=scalar)
        self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_mul(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_sub)
        cpu = test(x1, x2, 'cpu')
        xpu = test(x1, x2, 'xpu')
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_sub_)
        cpu_inplace = test_(x1, x2, 'cpu', is_inplace=True)
        xpu_inplace = test_(x1, x2, 'xpu', is_inplace=True)
        self.result_compare(cpu_inplace, xpu_inplace)

    def test_foreach_div(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        x2 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalar = random.uniform(-5, 5)

        test = ForeachTest(torch._foreach_div)
        cpu = test(x1, x2, 'cpu')
        xpu = test(x1, x2, 'xpu')
        self.result_compare(cpu, xpu)

        test_ = ForeachTest(torch._foreach_div_)
        cpu_inplace = test_(x1, x2, 'cpu', is_inplace=True)
        xpu_inplace = test_(x1, x2, 'xpu', is_inplace=True)
        self.result_compare(cpu_inplace, xpu_inplace)

    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())
