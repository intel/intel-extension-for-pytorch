import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from torch.testing._internal.jit_utils import JitTestCase
import unittest
import torch.nn.functional as F

conv_module = {1: torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

class EltwiseFusionOp:
    def __init__(self, ipex_eltwise_op, op_input_list={}):
        self.ipex_eltwise_op = ipex_eltwise_op
        self.op_input_list = op_input_list

unary_PyTorch_op_to_IPEX_op_map = {
    torch.relu: EltwiseFusionOp("relu"),
    torch.relu_: EltwiseFusionOp("relu_"),
    nn.ELU(inplace=False): EltwiseFusionOp("elu"),
    nn.ELU(inplace=True): EltwiseFusionOp("elu_"),
    torch.abs: EltwiseFusionOp("abs"),
    torch.abs_: EltwiseFusionOp("abs_"),
    torch.exp: EltwiseFusionOp("exp"),
    torch.exp_: EltwiseFusionOp("exp_"),
    nn.Hardswish(inplace=False): EltwiseFusionOp("hardswish"),
    nn.Hardswish(inplace=True): EltwiseFusionOp("hardswish_"),
    torch.log: EltwiseFusionOp("log"),
    torch.log_: EltwiseFusionOp("log_"),
    nn.Mish(inplace=False): EltwiseFusionOp("mish"),
    nn.Mish(inplace=True): EltwiseFusionOp("mish_"),
    torch.sigmoid: EltwiseFusionOp("sigmoid"),
    torch.sigmoid_: EltwiseFusionOp("sigmoid_"),
    torch.round: EltwiseFusionOp("round"),
    torch.round_: EltwiseFusionOp("round_"),
    torch.sqrt: EltwiseFusionOp("sqrt"),
    torch.sqrt_: EltwiseFusionOp("sqrt_"),
    torch.square: EltwiseFusionOp("square"),
    torch.square_: EltwiseFusionOp("square_"),
    torch.tanh: EltwiseFusionOp("tanh"),
    torch.tanh_: EltwiseFusionOp("tanh_"),
    nn.SiLU(inplace=False): EltwiseFusionOp("silu"),
    nn.SiLU(inplace=True): EltwiseFusionOp("silu_"),
}

non_unary_PyTorch_op_to_IPEX_op_map = {
    torch.clamp: EltwiseFusionOp("clamp", op_input_list={"min": -2, "max": 3}),
    torch.clamp_: EltwiseFusionOp("clamp_", op_input_list={"min": -2, "max": 3}),
    nn.GELU(approximate="none"): EltwiseFusionOp("gelu(none)"),
    nn.GELU(approximate="tanh"): EltwiseFusionOp("gelu(tanh)"),
    torch.pow: EltwiseFusionOp("pow", op_input_list={"exponent": 2}),
    lambda t: t.pow_(2): EltwiseFusionOp("pow_"),
    nn.LeakyReLU(negative_slope=0.02, inplace=False): EltwiseFusionOp("leaky_relu"),
    nn.LeakyReLU(negative_slope=0.02, inplace=True): EltwiseFusionOp("leaky_relu_"),
}

class ConvEltwise(nn.Module):
    def __init__(self, eltwise_fn, dim, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvEltwise, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.conv(x)
        b = self.eltwise(a, **self.kwargs)
        return b

class IPEXConvAddRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvAddRelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = F.relu(self.conv1(x))
        b = self.conv2(x)
        return F.relu(a.add_(b), inplace=True)

class IPEXConvConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvConvRelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        return F.relu(res, inplace=True)

class IPEXMatmulDiv(nn.Module):
    def __init__(self):
        super(IPEXMatmulDiv, self).__init__()
        seed = 2018
        torch.manual_seed(seed)

    def forward(self, x1, x2, x3):
        return torch.matmul(x1, x2) / x3 + x3

class TestTE(JitTestCase):
    def test_ipex_unary_conv_fusion(self, op_list=unary_PyTorch_op_to_IPEX_op_map):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        ipex._C.enable_custom_op_2_nnc_fuser()
        dim = 2
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        for eltwise in op_list:
            fusion_op = op_list[eltwise]
            ipex_eltwise_op = fusion_op.ipex_eltwise_op
            print("\nTEST conv2d+%s" % ipex_eltwise_op)
            for use_channels_last in [0, 1]:
                for batch_size, image_size in [[8, 20], [3, 256]]:
                    input_size = [batch_size, in_channels, image_size, image_size]
                    x = torch.randn(input_size)
                    te_model = ConvEltwise(eltwise, dim, in_channels, out_channels, kernel_size, image_size).eval()
                    if use_channels_last:
                        x = x.to(memory_format=torch.channels_last)
                        te_model = te_model.to(memory_format=torch.channels_last)
                    te_model_traced = torch.jit.trace(te_model, (x))
                    te_model_traced = torch.jit.freeze(te_model_traced)
                    te_model_traced(x)
                    self.assertAllFused(te_model_traced.graph_for(x))

                    res_jit = te_model_traced(x)
                    res_imperative = te_model(x)
                    self.assertEqual(res_jit, res_imperative, "{}, {}".format(res_jit, res_imperative))
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_non_unary_conv_fusion(self, op_list=non_unary_PyTorch_op_to_IPEX_op_map):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        ipex._C.enable_custom_op_2_nnc_fuser()
        dim = 2
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        for eltwise in op_list:
            fusion_op = op_list[eltwise]
            ipex_eltwise_op = fusion_op.ipex_eltwise_op
            print("\nTEST conv2d+%s" % ipex_eltwise_op)
            for use_channels_last in [0, 1]:
                for batch_size, image_size in [[8, 20], [3, 256]]:
                    input_size = [batch_size, in_channels, image_size, image_size]
                    x = torch.randn(input_size)
                    op_input_list = fusion_op.op_input_list
                    te_model = ConvEltwise(eltwise, dim, in_channels, out_channels, kernel_size, image_size, **op_input_list).eval()
                    if use_channels_last:
                        x = x.to(memory_format=torch.channels_last)
                        te_model = te_model.to(memory_format=torch.channels_last)
                    te_model_traced = torch.jit.trace(te_model, (x))
                    te_model_traced = torch.jit.freeze(te_model_traced)
                    te_model_traced(x)
                    self.assertAllFused(te_model_traced.graph_for(x))

                    res_jit = te_model_traced(x)
                    res_imperative = te_model(x)
                    self.assertEqual(res_jit, res_imperative, "{}, {}".format(res_jit, res_imperative))
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_add_relu(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        ipex._C.enable_custom_op_2_nnc_fuser()

        print("\nTEST conv2d+add+relu")
        for use_channels_last in [0, 1]:
            te_model = IPEXConvAddRelu(3, 2, kernel_size=(3, 3)).eval()
            x = torch.randn(1, 3, 10, 10)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                te_model = te_model.to(memory_format=torch.channels_last)
            te_model_traced = torch.jit.trace(te_model, (x))
            te_model_traced = torch.jit.freeze(te_model_traced)
            te_model_traced(x)
            self.assertAllFused(te_model_traced.graph_for(x))

            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

            x = torch.randn(3, 3, 20, 20)
            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_conv_relu(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        ipex._C.enable_custom_op_2_nnc_fuser()

        print("\nTEST conv bottleneck")
        for use_channels_last in [0, 1]:
            te_model = IPEXConvConvRelu(3, 10, kernel_size=(3, 3)).eval()
            x = torch.randn(1, 3, 224, 224)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                te_model = te_model.to(memory_format=torch.channels_last)
            te_model_traced = torch.jit.script(te_model)
            te_model_traced = torch.jit.freeze(te_model_traced)
            te_model_traced(x)

            self.assertAllFused(te_model_traced.graph_for(x))

            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

            x = torch.randn(3, 3, 500, 500)
            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_matmul_div(self):
        ipex._C.enable_custom_op_2_nnc_fuser()
        print("TEST conv matmul+div")
        te_matmul_div = IPEXMatmulDiv()
        x1 = torch.randn(5, 5)
        x2 = torch.randn(5, 5)
        x3 = torch.randn(5, 5)
        te_matmul_div_traced = torch.jit.script(te_matmul_div).eval()
        te_matmul_div_traced = torch.jit.freeze(te_matmul_div_traced)
        te_matmul_div_traced(x1, x2, x3)
        self.assertAllFused(te_matmul_div_traced.graph_for(x1, x2, x3))
        res_jit = te_matmul_div_traced(x1, x2, x3)
        res_imperative = te_matmul_div(x1, x2, x3)
        self.assertEqual(res_jit, res_imperative)

if __name__ == '__main__':
    test = unittest.main()
