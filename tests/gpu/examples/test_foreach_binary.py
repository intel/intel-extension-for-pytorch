import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestForeachBinary(TestCase):
    def test_maximum(self, dtype=torch.float):
        x1_cpu = torch.tensor([1.0, 3.0, 9.0, 6.0, 56.0, 99.0, 20.0, 30.0, 45.0]).to(
            cpu_device
        )
        x2_cpu = torch.tensor([6.0, 9.0, 3.0, 7.0, 0.0, 56.0, 97.0, 50.0, 14.0]).to(
            cpu_device
        )

        x1_xpu = torch.tensor([1.0, 3.0, 9.0, 6.0, 56.0, 99.0, 20.0, 30.0, 45.0]).to(
            xpu_device
        )
        x2_xpu = torch.tensor([6.0, 9.0, 3.0, 7.0, 0.0, 56.0, 97.0, 50.0, 14.0]).to(
            xpu_device
        )

        self.assertEqual(
            torch._foreach_maximum((x1_cpu,), (x2_cpu,)),
            torch._foreach_maximum((x1_xpu,), (x2_xpu,)),
        )
        self.assertEqual(
            torch._foreach_maximum_((x1_cpu,), (x2_cpu,)),
            torch._foreach_maximum_((x1_xpu,), (x2_xpu,)),
        )

    def test_minimum(self, dtype=torch.float):
        x1_cpu = torch.tensor([1, 3, 9, 6]).to(cpu_device)
        x2_cpu = torch.tensor([6, 9, 3, 7]).to(cpu_device)

        x1_xpu = torch.tensor([1, 3, 9, 6]).to(xpu_device)
        x2_xpu = torch.tensor([6, 9, 3, 7]).to(xpu_device)

        self.assertEqual(
            torch._foreach_minimum((x1_cpu,), (x2_cpu,)),
            torch._foreach_minimum((x1_xpu,), (x2_xpu,)),
        )
        self.assertEqual(
            torch._foreach_minimum_((x1_cpu,), (x2_cpu,)),
            torch._foreach_minimum_((x1_xpu,), (x2_xpu,)),
        )

        self.assertEqual(torch._foreach_minimum((x1_cpu, ), 3), torch._foreach_minimum((x1_xpu, ), 3))
        self.assertEqual(torch._foreach_minimum_((x1_cpu, ), 3), torch._foreach_minimum_((x1_xpu, ), 3))

        self.assertEqual(torch._foreach_minimum((x1_cpu, ), (3, )), torch._foreach_minimum((x1_xpu, ), (3, )))
        self.assertEqual(torch._foreach_minimum_((x1_cpu, ), (3, )), torch._foreach_minimum_((x1_xpu, ), (3, )))

    def test_foreach_norm(self, dtype=torch.float):
        shape = [1024, 1024]
        x1 = [torch.randn(shape, dtype=torch.float) for _ in range(18)]
        x1_xpu = [x1[i].clone().to('xpu') for i in range(len(x1))]

        for scalar in [1, 2]:
            out_x1_xpu = torch._foreach_norm(x1_xpu, scalar)
            for j in range(len(x1)):
                out_x1_ref = torch.norm(x1_xpu[j], scalar)
                self.assertEqual(out_x1_ref.to('cpu'), out_x1_xpu[j].to('cpu'), atol=1e-6, rtol=1e-5)

if __name__ == "__main__":
    x1_cpu = torch.tensor([1, 3, 9, 6]).to(cpu_device)
    x2_cpu = torch.tensor([6, 9, 3, 7]).to(cpu_device)

    x1_xpu = torch.tensor([1, 3, 9, 6]).to(xpu_device)
    x2_xpu = torch.tensor([6, 9, 3, 7]).to(xpu_device)

    x1_tuple = (x1_xpu,)
    x2_tuple = (x2_xpu,)
    print(torch._foreach_maximum(x1_tuple, x2_tuple))
    print(torch._foreach_maximum((x1_cpu,), (x2_cpu,)))

    t1 = torch.rand(16)
    t2 = torch.rand(16)
    t1_xpu = t1.to(xpu_device)
    t2_xpu = t2.to(xpu_device)
    print(torch._foreach_maximum((t1,), (t2,)))
    print(torch._foreach_maximum((t1_xpu,), (t2_xpu,)))
