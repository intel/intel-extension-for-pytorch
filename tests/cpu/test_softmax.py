import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
from torch.testing._internal.jit_utils import JitTestCase
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
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

class inplace_softmax_with_blocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, flag):
        if flag:
          if flag:
            x1 = x + 1
          else:
            x1 = x + 2
        else:
            x1 = x + 3
        x2 = torch.softmax(x1, dim=-1)
        return x2

class softmax_MHA(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        attention_scores = torch.matmul(x, x.transpose(-1, -2))
        attention_scores = attention_scores / 64
        attention_scores = attention_scores + x
        attention_scores = nn.Softmax(dim=-1)(attention_scores)
        return attention_scores

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

class softmax_dtype(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.nn.functional.softmax(x, dtype=x.dtype, dim=1)

class inplace_softmax_dtype(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x + 1
        return torch.nn.functional.softmax(x1, dtype=x.dtype, dim=1)

class SoftmaxTester(JitTestCase):
    def test_softmax(self):
        for dtype in ["fp32", "bf16"]:
            test1 = torch.tensor([[2.0,2.0],[2.0,2.0]])
            test2 = torch.tensor([[2.0,2.0],[2.0,2.0]])
            test3 = torch.tensor([[1.0,1.0],[1.0,1.0]])
            test4 = torch.tensor([[1.0,1.0],[1.0,1.0]]).transpose(1,0)
            test5 = torch.tensor([[2.0,2.0],[2.0,2.0]]).transpose(1,0)
            test6 = torch.rand(1,16,64,64)
            test7 = torch.tensor([[1.0,1.0],[1.0,1.0]])
            test8 = torch.rand(2,3)
            test9 = test8 - 1
            test10 = torch.tensor([[1.0,1.0],[1.0,1.0]])

            if dtype == "bf16":
                test1 = test1.bfloat16()
                test2 = test2.bfloat16()
                test3 = test3.bfloat16()
                test4 = test4.bfloat16()
                test5 = test5.bfloat16()
                test7 = test7.bfloat16()
                test8 = test8.bfloat16()
                test9 = test9.bfloat16()
                test10 = test10.bfloat16()

            model1 = softmax_with_multiuse_input().eval()
            model2 = softmax_with_alias_input().eval()
            model3 = inplace_softmax().eval()
            model4 = inplace_softmax().eval()
            model5 = softmax_with_multiuse_input().eval()
            model6 = softmax_MHA().eval()
            model7 = inplace_softmax_with_blocks().eval()
            model8 = softmax_dtype().eval()
            model9 = inplace_softmax_dtype().eval()
            model10 = inplace_softmax_with_TE_group().eval()

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
                model7 = torch.jit.script(model7)
                res7 = model7(test7, torch.BoolTensor([True]))
                model8 = torch.jit.trace(model8, test8)
                res8 = model8(test8)
                model9 = torch.jit.trace(model9, test9)
                res9 = model9(test9)
                model10_traced = torch.jit.trace(model10, test10)
                res10_traced = model10_traced(test10)
                res10 = model10(test10)

            # int8 case, testing inplac with llga fusion group
            qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8), weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
            ipex.nn.utils._model_convert.replace_dropout_with_identity(model6)
            prepared_model = prepare(model6, qconfig, example_inputs=test6, inplace=False)
            prepared_model(test6)
            converted_model = convert(prepared_model)
            converted_model = torch.jit.trace(converted_model, test6)
            converted_model = torch.jit.freeze(converted_model)

            with torch.no_grad():
                res6 = converted_model(test6)
                res6 = converted_model(test6)
            with torch.no_grad():
                res6 = converted_model(test6)
                res6_ori = model6(test6)

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
            # should be inplace and pass the checking in llga fusion group
            graph6 = converted_model.graph_for(test3)
            self.assertGraphContainsExactly(graph6, IPEX_SOFTMAX_, 1)
            # should be inplace
            graph7 = model7.graph_for(test7, torch.BoolTensor([True]))
            self.assertGraphContainsExactly(graph7, IPEX_SOFTMAX_, 1)
            # should be outplace
            graph8 = model8.graph_for(test8)
            self.assertGraphContainsExactly(graph8, IPEX_SOFTMAX, 1)
            # should be inplace
            graph9 = model9.graph_for(test9)
            self.assertGraphContainsExactly(graph9, IPEX_SOFTMAX_, 1)
            # should be inplace
            graph10 = model10_traced.graph_for(test10)
            self.assertGraphContainsExactly(graph10, IPEX_SOFTMAX_, 2)

            # the output results of above inplace/outplace softmax should be the same
            self.assertEqual(res1[0], res2[1], 0)
            self.assertEqual(res1[0], res3, 0)
            self.assertEqual(res1[0], res4, 0)
            self.assertEqual(res1[0], res5[0], 0)
            self.assertEqual(res1[0], res7, 0)
            self.assertEqual(res8, res9, 0)
            self.assertEqual(res10[0], res10_traced[0], 0)
            self.assertEqual(res10[1], res10_traced[1], 0)


if __name__ == '__main__':
    test = unittest.main()
