import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_distance(self, dtype=torch.float):
        pdist = nn.PairwiseDistance(p=2)
        input1 = torch.randn(100, 128, device=cpu_device,
                             dtype=dtype, requires_grad=True)
        input2 = torch.randn(100, 128, device=cpu_device,
                             dtype=dtype, requires_grad=True)

        input1_dpcpp = input1.to("xpu")
        input2_dpcpp = input2.to("xpu")
        output = pdist(input1, input2)
        pdist_dpcpp = pdist.to(dpcpp_device)
        output_dpcpp = pdist_dpcpp(input1_dpcpp, input2_dpcpp)
        self.assertEqual(output, output_dpcpp.to(cpu_device))

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(input1, input2)
        cos_dpcpp = cos.to(dpcpp_device)
        output_dpcpp = cos_dpcpp(input1_dpcpp, input2_dpcpp)
        self.assertEqual(output, output_dpcpp.to(cpu_device))

    def test_pdist(self, dtype=torch.float):
        for p in (0, 1, 2, 3, float('inf')):
            print("p = ", p)

            a = torch.randn([10, 15], requires_grad=True)
            y = torch.pdist(a, p)
            g = torch.ones_like(y, requires_grad=True)
            y.backward(g)
            grad_cpu = a.grad.detach().clone()
            a.grad.zero_()

            a_xpu = a.to('xpu')
            a_xpu.retain_grad()
            y_xpu = torch.pdist(a_xpu, p)
            g_xpu = torch.ones_like(y_xpu, requires_grad=True).to('xpu')
            y_xpu.backward(g_xpu)
            grad_xpu = a_xpu.grad.detach().clone()
            a_xpu.grad.zero_()

            self.assertEqual(y, y_xpu)
            self.assertEqual(grad_cpu, grad_xpu)

    def test_cdist(self, dtype=torch.float):
        for p in (0, 1, 2, 3, float('inf')):
            print("p = ", p)

            # small test: P < 25 & R < 25
            a = torch.randn([3, 10, 15], requires_grad=True)
            b = torch.randn([3, 5, 15], requires_grad=True)
            y = torch.cdist(a, b, p)
            g = torch.ones_like(y, requires_grad=True)
            y.backward(g)
            grad_cpu = a.grad.detach().clone()
            a.grad.zero_()

            a_xpu = a.to('xpu')
            b_xpu = b.to('xpu')
            a_xpu.retain_grad()
            b_xpu.retain_grad()
            y_xpu = torch.cdist(a_xpu, b_xpu, p)
            g_xpu = torch.ones_like(y_xpu, requires_grad=True).to('xpu')
            y_xpu.backward(g_xpu)
            grad_xpu = a_xpu.grad.detach().clone()
            a_xpu.grad.zero_()

            self.assertEqual(y, y_xpu)
            self.assertEqual(grad_cpu, grad_xpu)

            # large test: P > 25 & R > 25
            a = torch.randn([3, 30, 15], requires_grad=True)
            b = torch.randn([3, 35, 15], requires_grad=True)
            y = torch.cdist(a, b, p)
            g = torch.ones_like(y, requires_grad=True)
            y.backward(g)
            grad_cpu = a.grad.detach().clone()
            a.grad.zero_()

            a_xpu = a.to('xpu')
            b_xpu = b.to('xpu')
            a_xpu.retain_grad()
            b_xpu.retain_grad()
            y_xpu = torch.cdist(a_xpu, b_xpu, p)
            g_xpu = torch.ones_like(y_xpu, requires_grad=True).to('xpu')
            y_xpu.backward(g_xpu)
            grad_xpu = a_xpu.grad.detach().clone()
            a_xpu.grad.zero_()

            self.assertEqual(y, y_xpu)
            self.assertEqual(grad_cpu, grad_xpu)
