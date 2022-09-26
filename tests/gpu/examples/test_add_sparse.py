import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_sparse(self, dtype=torch.float):
        i_cpu = torch.LongTensor([[0, 1, 1], [2, 0, 0]])
        v_cpu = torch.FloatTensor([3, 4, 5])
        s_cpu = torch.torch._sparse_coo_tensor_unsafe(i_cpu, v_cpu, torch.Size([2, 3]))
        x_cpu = torch.randn([2, 3], dtype=dtype)

        s_xpu = s_cpu.to(dpcpp_device)
        x_xpu = x_cpu.to(dpcpp_device)

        print("s_cpu = ", s_cpu)
        print("s_xpu = ", s_xpu)
        print("x_cpu = ", x_cpu)
        print("x_xpu = ", x_xpu)

        y_cpu = torch.add(x_cpu, s_cpu)
        y_xpu = torch.add(x_xpu, s_xpu)

        print("sum cpu = ", y_cpu)
        print("sum xpu = ", y_xpu)

        self.assertEqual(y_cpu, y_xpu.cpu())
