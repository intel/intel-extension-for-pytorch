import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestForeachBinary(TestCase):
    @repeat_test_for_types([torch.float, torch.int8, torch.half, torch.bfloat16])
    def test_maximum(self, dtype=torch.float):
        x1_cpu = torch.tensor([1., 3., 9., 6., 56., 99., 20., 30., 45.]).to(cpu_device)
        x2_cpu = torch.tensor([6., 9., 3., 7., 0., 56., 97., 50., 14.]).to(cpu_device)

        x1_xpu = torch.tensor([1., 3., 9., 6., 56., 99., 20., 30., 45.]).to(xpu_device)
        x2_xpu = torch.tensor([6., 9., 3., 7., 0., 56., 97., 50., 14.]).to(xpu_device)

        self.assertEqual(torch._foreach_maximum((x1_cpu, ), (x2_cpu, )), torch._foreach_maximum((x1_xpu, ), (x2_xpu,)))

    @repeat_test_for_types([torch.float, torch.int8, torch.half, torch.bfloat16])
    def test_minimum(self, dtype=torch.float):
        x1_cpu = torch.tensor([1, 3, 9, 6]).to(cpu_device)
        x2_cpu = torch.tensor([6, 9, 3, 7]).to(cpu_device)

        x1_xpu = torch.tensor([1, 3, 9, 6]).to(xpu_device)
        x2_xpu = torch.tensor([6, 9, 3, 7]).to(xpu_device)

        self.assertEqual(torch._foreach_minimum((x1_cpu, ), (x2_cpu, )), torch._foreach_minimum((x1_xpu, ), (x2_xpu, )))


if __name__ == "__main__":

    with torch.autograd.profiler_legacy.profile(enabled=True, use_xpu=True, record_shapes=False) as prof:
        x1_cpu = torch.tensor([1, 3, 9, 6]).to(cpu_device)
        x2_cpu = torch.tensor([6, 9, 3, 7]).to(cpu_device)

        x1_xpu = torch.tensor([1, 3, 9, 6]).to(xpu_device)
        x2_xpu = torch.tensor([6, 9, 3, 7]).to(xpu_device)

        x1_tuple = (x1_xpu, )
        x2_tuple = (x2_xpu, )
        print(torch._foreach_maximum(x1_tuple, x2_tuple))
        print(torch._foreach_maximum((x1_cpu,), (x2_cpu,)))

        t1 = torch.rand(16)
        t2 = torch.rand(16)
        t1_xpu = t1.to(xpu_device)
        t2_xpu = t2.to(xpu_device)
        print(torch._foreach_maximum((t1,), (t2,)))
        print(torch._foreach_maximum((t1_xpu,), (t2_xpu,)))



    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.tile.' + 'pt')
