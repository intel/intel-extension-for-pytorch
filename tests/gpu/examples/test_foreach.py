import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import (TestCase)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class ForeachTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, device):
        input_for_func = []
        for i in input:
            input_for_func.append(i.clone().to(device))
        return self.func(input_for_func)

class TestTorchMethod(TestCase):
    # @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_foreach_abs(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_abs)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_sigmoid(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_sigmoid)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_round(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_round)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_frac(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_frac)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_reciprocal(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_reciprocal)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_erfc(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_erfc)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_expm1(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_expm1)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_lgamma(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_lgamma)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_traunc(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_trunc)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_floor(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_floor)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_ceil(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_ceil)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_acos(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_acos)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_asin(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_asin)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_atan(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_atan)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_cosh(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_cosh)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_tan(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_tan)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_sin(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_sin)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_sinh(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_sinh)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_exp(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_exp)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_tanh(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_tanh)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_log(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_log)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_log10(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_log10)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_log2(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_log2)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_cos(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_cos)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_sqrt(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_sqrt)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_log1p(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_log1p)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_erf(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_erf)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def test_foreach_neg(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]

        test = ForeachTest(torch._foreach_neg)
        cpu = test(x1, 'cpu')
        xpu = test(x1, 'xpu')
        self.result_compare(cpu, xpu)

    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())
