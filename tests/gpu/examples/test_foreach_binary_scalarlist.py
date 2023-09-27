import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class ForeachBinaryScalarListTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, input, scalarlist, device, is_inplace=False):
        input_tensor_for_func = []
        input_scalar_for_func = []

        for i in range(len(input)):
            input_tensor_for_func.append(input[i].clone().to(device))
            input_scalar_for_func.append(scalarlist[i].clone().to(device))

        if is_inplace:
            self.func(input_tensor_for_func, input_scalar_for_func)
            return input_tensor_for_func
        else :
            return self.func(input_tensor_for_func, input_scalar_for_func)


class TestTorchMethod(TestCase):
    def test_foreach_add_scalarlist(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalarlist = torch.randn(250)

        test = ForeachBinaryScalarListTest(torch._foreach_add)
        cpu = test(x1, scalarlist, "cpu")
        xpu = test(x1, scalarlist, "xpu")
        self.result_compare(cpu, xpu)

        test_ = ForeachBinaryScalarListTest(torch._foreach_add_)
        cpu_ = test_(x1, scalarlist, "cpu", True)
        xpu_ = test_(x1, scalarlist, "xpu", True)
        self.result_compare(cpu_, xpu_)

    def test_foreach_mul_scalarlist(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalarlist = torch.randn(250)

        test = ForeachBinaryScalarListTest(torch._foreach_mul)
        cpu = test(x1, scalarlist, "cpu")
        xpu = test(x1, scalarlist, "xpu")
        self.result_compare(cpu, xpu)

        test_ = ForeachBinaryScalarListTest(torch._foreach_mul_)
        cpu_ = test_(x1, scalarlist, "cpu", True)
        xpu_ = test_(x1, scalarlist, "xpu", True)
        self.result_compare(cpu_, xpu_)

    def test_foreach_sub_scalarlist(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalarlist = torch.randn(250)

        test = ForeachBinaryScalarListTest(torch._foreach_sub)
        cpu = test(x1, scalarlist, "cpu")
        xpu = test(x1, scalarlist, "xpu")
        self.result_compare(cpu, xpu)

        test_ = ForeachBinaryScalarListTest(torch._foreach_sub_)
        cpu_ = test_(x1, scalarlist, "cpu", True)
        xpu_ = test_(x1, scalarlist, "xpu", True)
        self.result_compare(cpu_, xpu_)

    def test_foreach_div_scalarlist(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalarlist = torch.randn(250)

        test = ForeachBinaryScalarListTest(torch._foreach_div)
        cpu = test(x1, scalarlist, "cpu")
        xpu = test(x1, scalarlist, "xpu")
        self.result_compare(cpu, xpu)

        test_ = ForeachBinaryScalarListTest(torch._foreach_div_)
        cpu_ = test_(x1, scalarlist, "cpu", True)
        xpu_ = test_(x1, scalarlist, "xpu", True)
        self.result_compare(cpu_, xpu_)

    def test_foreach_clamp_min_scalarlist(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalarlist = torch.randn(250)

        test = ForeachBinaryScalarListTest(torch._foreach_clamp_min)
        cpu = test(x1, scalarlist, "cpu")
        xpu = test(x1, scalarlist, "xpu")
        self.result_compare(cpu, xpu)

        test_ = ForeachBinaryScalarListTest(torch._foreach_clamp_min_)
        cpu_ = test_(x1, scalarlist, "cpu", True)
        xpu_ = test_(x1, scalarlist, "xpu", True)
        self.result_compare(cpu_, xpu_)


    def test_foreach_clamp_max_scalarlist(self, dtype=torch.float):
        x1 = [torch.randn([5, 8], dtype=torch.float) for _ in range(250)]
        scalarlist = torch.randn(250)

        test = ForeachBinaryScalarListTest(torch._foreach_clamp_max)
        cpu = test(x1, scalarlist, "cpu")
        xpu = test(x1, scalarlist, "xpu")
        self.result_compare(cpu, xpu)

        test_ = ForeachBinaryScalarListTest(torch._foreach_clamp_max_)
        cpu_ = test_(x1, scalarlist, "cpu", True)
        xpu_ = test_(x1, scalarlist, "xpu", True)
        self.result_compare(cpu_, xpu_)


    def result_compare(self, x1, x2):
        for i in range(len(x1)):
            self.assertEqual(x1[i].cpu(), x2[i].cpu())
