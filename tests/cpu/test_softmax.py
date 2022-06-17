import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
from torch.testing._internal.jit_utils import JitTestCase
import unittest
IPEX_SOFTMAX = 'ipex::softmax'
IPEX_SOFTMAX_ = 'ipex::softmax_'
ATEN_SOFTMAX = 'aten::softmax'

class softmax_with_multiuse_input(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = nn.Softmax(dim=-1)(x)
        x2 = x + x1 
        return x1, x2

class softmax_with_alias_input(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x
        x2 = nn.Softmax(dim=-1)(x)
        return x1, x2

class inplace_softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x + 1
        x2 = nn.Softmax(dim=-1)(x1)
        return x2

class inplace_softmax_with_TE_group(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x + 1
        x2 = x + 2
        x3 = x + 3
        x4 = x + 4
        x5 = x + 5
        y1 = (x1 / x2).softmax(dim = -1)
        y2 = ((x4 - x3) / x5).softmax(dim = -1)
        return y1, y2


class SoftmaxTester(JitTestCase):
    def test_softmax(self):
        for dtype in ["fp32", "bf16"]:
            test1 = torch.tensor([[2.0,2.0],[2.0,2.0]])
            test2 = torch.tensor([[2.0,2.0],[2.0,2.0]])
            test3 = torch.tensor([[1.0,1.0],[1.0,1.0]])
            test4 = torch.tensor([[1.0,1.0],[1.0,1.0]]).transpose(1,0)
            test5 = torch.tensor([[2.0,2.0],[2.0,2.0]]).transpose(1,0)
            test6 = torch.tensor([[1.0,1.0],[1.0,1.0]])

            if dtype == "bf16":
                test1 = test1.bfloat16()
                test2 = test2.bfloat16()
                test3 = test3.bfloat16()
                test4 = test4.bfloat16()
                test5 = test5.bfloat16()
                test6 = test6.bfloat16()

            model1 = softmax_with_multiuse_input().eval()
            model2 = softmax_with_alias_input().eval()
            model3 = inplace_softmax().eval()
            model4 = inplace_softmax().eval()
            model5 = softmax_with_multiuse_input().eval()
            model6 = inplace_softmax_with_TE_group().eval()

            with torch.no_grad():
                model1 = torch.jit.trace(model1, test1)
                res1 = model1(test1)
                model2 = torch.jit.trace(model2, test2)
                res2 = model2(test2)
                model3 = torch.jit.trace(model3, test3)
                res3 = model3(test3)
                model4 = torch.jit.trace(model4, test4)
                res4 = model4(test4)
                model5 = torch.jit.trace(model5, test5)
                res5 = model5(test5)
                model6_traced = torch.jit.trace(model6, test6)
                res6_traced = model6_traced(test6)
                res6 = model6(test6)


            # should be outplace since multi-use
            graph1 = model1.graph_for(test1)
            self.assertGraphContainsExactly(graph1, IPEX_SOFTMAX, 1)
            # should be outplace since alias
            graph2 = model2.graph_for(test2)
            self.assertGraphContainsExactly(graph2, IPEX_SOFTMAX, 1)
            # should be inplace
            graph3 = model3.graph_for(test3)
            self.assertGraphContainsExactly(graph3, IPEX_SOFTMAX_, 1)
            # inplace test, but should be aten::softmax due to non-contiguous input
            graph4 = model4.graph_for(test4)
            self.assertGraphContainsExactly(graph4, ATEN_SOFTMAX, 1)
            # outplace test, but should be aten::softmax due to non-contiguous input
            graph5 = model5.graph_for(test5)
            self.assertGraphContainsExactly(graph5, ATEN_SOFTMAX, 1)
            # should be inplace
            graph6 = model6_traced.graph_for(test6)
            self.assertGraphContainsExactly(graph6, IPEX_SOFTMAX_, 2)

            # the output results of above inplace/outplace softmax should be the same
            self.assertEqual(res1[0], res2[1], 0)
            self.assertEqual(res1[0], res3, 0)
            self.assertEqual(res1[0], res4, 0)
            self.assertEqual(res1[0], res5[0], 0)
            self.assertEqual(res6[0], res6_traced[0], 0)
            self.assertEqual(res6[1], res6_traced[1], 0)


if __name__ == '__main__':
    test = unittest.main()
