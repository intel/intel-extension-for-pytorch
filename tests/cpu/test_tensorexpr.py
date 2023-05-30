import torch
import torch.nn as nn
from torch.testing._internal.jit_utils import JitTestCase
import unittest
import torch.nn.functional as F
import time


def get_rand_seed():
    return int(time.time() * 1000000000)


conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

from typing import Dict, NamedTuple


class EltwiseFusionOp(NamedTuple):
    ipex_eltwise_op: str
    op_input_list: Dict = {}


unary_PyTorch_op_to_IPEX_op_map = {
    torch.relu: EltwiseFusionOp("relu"),
    torch.relu_: EltwiseFusionOp("relu_"),
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
    nn.Hardsigmoid(inplace=False): EltwiseFusionOp("hardsigmoid"),
    nn.Hardsigmoid(inplace=True): EltwiseFusionOp("hardsigmoid_"),
}

non_unary_PyTorch_op_to_IPEX_op_map = {
    torch.clamp: EltwiseFusionOp("clamp", op_input_list={"min": -2, "max": 3}),
    torch.clamp_: EltwiseFusionOp("clamp_", op_input_list={"min": -2, "max": 3}),
    nn.GELU(approximate="none"): EltwiseFusionOp("gelu(none)"),
    nn.GELU(approximate="tanh"): EltwiseFusionOp("gelu(tanh)"),
    nn.ELU(inplace=False): EltwiseFusionOp("elu"),
    nn.ELU(inplace=True): EltwiseFusionOp("elu_"),
    torch.pow: EltwiseFusionOp("pow", op_input_list={"exponent": 2}),
    lambda t: t.pow_(2): EltwiseFusionOp("pow_"),
    nn.LeakyReLU(negative_slope=0.02, inplace=False): EltwiseFusionOp("leaky_relu"),
    nn.LeakyReLU(negative_slope=0.02, inplace=True): EltwiseFusionOp("leaky_relu_"),
}


class ConvEltwise(nn.Module):
    def __init__(
        self,
        eltwise_fn,
        dim,
        in_channels,
        out_channels,
        kernel_size,
        image_size,
        **kwargs
    ):
        super(ConvEltwise, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.conv(x)
        b = self.eltwise(a, **self.kwargs)
        return b


class IPEXConvAdd(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvAdd, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return a.add_(b)


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


class IPEXConvSigmoidMul(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvSigmoidMul, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = torch.sigmoid(a)
        return a.mul_(b)


class LinearEltwise(nn.Module):
    def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
        super(LinearEltwise, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.linear(x)
        a = a / 2
        b = self.eltwise(a, **self.kwargs)
        return b


class IPEXLinearAdd(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(IPEXLinearAdd, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        a = self.linear1(x)
        b = self.linear2(x)
        return a.add_(b)


class IPEXLinearAddRelu(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(IPEXLinearAddRelu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        a = F.relu(self.linear(x))
        b = self.linear(x)
        return F.relu(a.add_(b), inplace=True)


class IPEXLinearSigmoidMul(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(IPEXLinearSigmoidMul, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        a = self.linear(x)
        b = torch.sigmoid(a)
        return a.mul_(b)


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
        dim = 2
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        for eltwise in op_list:
            rand_seed = int(get_rand_seed())
            torch.manual_seed(rand_seed)
            fusion_op = op_list[eltwise]
            ipex_eltwise_op = fusion_op.ipex_eltwise_op
            print("TEST conv2d+%s" % ipex_eltwise_op)
            for use_channels_last in [0, 1]:
                for batch_size, image_size in [[8, 20], [3, 256]]:
                    input_size = [batch_size, in_channels, image_size, image_size]
                    x = torch.randn(input_size)
                    te_model = ConvEltwise(
                        eltwise, dim, in_channels, out_channels, kernel_size, image_size
                    ).eval()
                    if use_channels_last:
                        x = x.to(memory_format=torch.channels_last)
                        te_model = te_model.to(memory_format=torch.channels_last)
                    te_model_traced = torch.jit.trace(te_model, (x))
                    te_model_traced = torch.jit.freeze(te_model_traced)
                    te_model_traced(x)
                    # self.assertAllFused(te_model_traced.graph_for(x))

                    res_jit = te_model_traced(x)
                    res_imperative = te_model(x)
                    self.assertEqual(
                        res_jit,
                        res_imperative,
                        "{}, {}".format(res_jit, res_imperative),
                    )
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_non_unary_conv_fusion(
        self, op_list=non_unary_PyTorch_op_to_IPEX_op_map
    ):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        dim = 2
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        for eltwise in op_list:
            rand_seed = int(get_rand_seed())
            torch.manual_seed(rand_seed)
            fusion_op = op_list[eltwise]
            ipex_eltwise_op = fusion_op.ipex_eltwise_op
            print("TEST conv2d+%s" % ipex_eltwise_op)
            for use_channels_last in [0, 1]:
                for batch_size, image_size in [[8, 20], [3, 256]]:
                    input_size = [batch_size, in_channels, image_size, image_size]
                    x = torch.randn(input_size)
                    op_input_list = fusion_op.op_input_list
                    te_model = ConvEltwise(
                        eltwise,
                        dim,
                        in_channels,
                        out_channels,
                        kernel_size,
                        image_size,
                        **op_input_list
                    ).eval()
                    if use_channels_last:
                        x = x.to(memory_format=torch.channels_last)
                        te_model = te_model.to(memory_format=torch.channels_last)
                    te_model_traced = torch.jit.trace(te_model, (x))
                    te_model_traced = torch.jit.freeze(te_model_traced)
                    te_model_traced(x)
                    # self.assertAllFused(te_model_traced.graph_for(x))

                    res_jit = te_model_traced(x)
                    res_imperative = te_model(x)
                    self.assertEqual(
                        res_jit,
                        res_imperative,
                        "{}, {}".format(res_jit, res_imperative),
                    )
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_add(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        print("TEST conv2d+add")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for use_channels_last in [0, 1]:
            te_model = IPEXConvAdd(3, 2, kernel_size=(3, 3)).eval()
            x = torch.randn(1, 3, 10, 10)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                te_model = te_model.to(memory_format=torch.channels_last)
            te_model_traced = torch.jit.trace(te_model, (x))
            te_model_traced = torch.jit.freeze(te_model_traced)
            te_model_traced(x)
            # self.assertAllFused(te_model_traced.graph_for(x))

            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

            x = torch.randn(3, 3, 20, 20)
            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_add_relu(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        print("TEST conv2d+add+relu")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for use_channels_last in [0, 1]:
            te_model = IPEXConvAddRelu(3, 2, kernel_size=(3, 3)).eval()
            x = torch.randn(1, 3, 10, 10)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                te_model = te_model.to(memory_format=torch.channels_last)
            te_model_traced = torch.jit.trace(te_model, (x))
            te_model_traced = torch.jit.freeze(te_model_traced)
            te_model_traced(x)
            # self.assertAllFused(te_model_traced.graph_for(x))

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
        print("TEST conv bottleneck")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for use_channels_last in [0, 1]:
            te_model = IPEXConvConvRelu(3, 10, kernel_size=(3, 3)).eval()
            x = torch.randn(1, 3, 224, 224)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                te_model = te_model.to(memory_format=torch.channels_last)
            te_model_traced = torch.jit.script(te_model)
            te_model_traced = torch.jit.freeze(te_model_traced)
            te_model_traced(x)

            # self.assertAllFused(te_model_traced.graph_for(x))

            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

            x = torch.randn(3, 3, 500, 500)
            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_sigmoid_mul(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        print("TEST conv2d+sigmoid+mul")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for use_channels_last in [0, 1]:
            te_model = IPEXConvSigmoidMul(3, 2, kernel_size=(3, 3)).eval()
            x = torch.randn(1, 3, 10, 10)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                te_model = te_model.to(memory_format=torch.channels_last)
            te_model_traced = torch.jit.trace(te_model, (x))
            te_model_traced = torch.jit.freeze(te_model_traced)
            te_model_traced(x)
            # self.assertAllFused(te_model_traced.graph_for(x))

            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

            x = torch.randn(3, 3, 20, 20)
            res_jit = te_model_traced(x)
            res_imperative = te_model(x)
            self.assertEqual(res_jit, res_imperative)

        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_matmul_div(self):
        print("TEST conv matmul+div")
        te_matmul_div = IPEXMatmulDiv()
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        x1 = torch.randn(5, 5)
        x2 = torch.randn(5, 5)
        x3 = torch.randn(5, 5)
        te_matmul_div_traced = torch.jit.script(te_matmul_div).eval()
        te_matmul_div_traced = torch.jit.freeze(te_matmul_div_traced)
        te_matmul_div_traced(x1, x2, x3)
        # self.assertAllFused(te_matmul_div_traced.graph_for(x1, x2, x3))
        res_jit = te_matmul_div_traced(x1, x2, x3)
        res_imperative = te_matmul_div(x1, x2, x3)
        self.assertEqual(res_jit, res_imperative)

    def test_ipex_unary_linear_fusion(self, op_list=unary_PyTorch_op_to_IPEX_op_map):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        batch_size = 3
        out_channels = 32
        in_channels = 3
        for eltwise in op_list:
            rand_seed = int(get_rand_seed())
            torch.manual_seed(rand_seed)
            fusion_op = op_list[eltwise]
            ipex_eltwise_op = fusion_op.ipex_eltwise_op
            """ # Issue of "round" 
                The OP "round" in ideep has numeric issue when input is exactly 0.500,
                so we fix the seed here for "round".
                For example:
                    x = torch.Tensor([0.500])
                    ideep: 1.0 = torch.round(x)
                    expected: 0.0 = torch.round(x)
                The seed to reproduce the failure: 1665593217573048320
            """
            if "round" in ipex_eltwise_op:
                torch.manual_seed(1665594679504775936)
            print("TEST linear+%s" % ipex_eltwise_op)
            for bias in [True, False]:
                input_size = [batch_size, in_channels]
                x = torch.randn(input_size)
                # linear fusion only supports bf16
                with torch.cpu.amp.autocast(
                    enabled=True, dtype=torch.bfloat16
                ), torch.no_grad():
                    te_model = LinearEltwise(
                        eltwise, in_channels, out_channels, bias
                    ).eval()
                    te_model_traced = torch.jit.trace(te_model, (x))
                    te_model_traced = torch.jit.freeze(te_model_traced)
                    te_model_traced(x)
                    # self.assertAllFused(te_model_traced.graph_for(x))

                    res_jit = te_model_traced(x)
                    res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_non_unary_linear_fusion(
        self, op_list=non_unary_PyTorch_op_to_IPEX_op_map
    ):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        batch_size = 3
        out_channels = 32
        in_channels = 3
        for eltwise in op_list:
            rand_seed = int(get_rand_seed())
            torch.manual_seed(rand_seed)
            fusion_op = op_list[eltwise]
            ipex_eltwise_op = fusion_op.ipex_eltwise_op
            print("TEST linear+%s" % ipex_eltwise_op)
            for bias in [True, False]:
                input_size = [batch_size, in_channels]
                x = torch.randn(input_size)
                op_input_list = fusion_op.op_input_list
                # linear fusion only supports bf16
                with torch.cpu.amp.autocast(
                    enabled=True, dtype=torch.bfloat16
                ), torch.no_grad():
                    te_model = LinearEltwise(
                        eltwise, in_channels, out_channels, bias, **op_input_list
                    ).eval()
                    te_model_traced = torch.jit.trace(te_model, (x))
                    te_model_traced = torch.jit.freeze(te_model_traced)
                    te_model_traced(x)
                    # self.assertAllFused(te_model_traced.graph_for(x))

                    res_jit = te_model_traced(x)
                    res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_linear_add(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        print("TEST linear+add")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for bias in [True, False]:
            with torch.cpu.amp.autocast(
                enabled=True, dtype=torch.bfloat16
            ), torch.no_grad():
                te_model = IPEXLinearAdd(3, 32, bias).eval()
                x = torch.randn(3, 3)
                te_model_traced = torch.jit.trace(te_model, (x))
                te_model_traced = torch.jit.freeze(te_model_traced)
                te_model_traced(x)
                # self.assertAllFused(te_model_traced.graph_for(x))

                res_jit = te_model_traced(x)
                res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )

                x = torch.randn(8, 3)
                res_jit = te_model_traced(x)
                res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )

    def test_ipex_linear_add_relu(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        print("TEST linear+add+relu")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for bias in [True, False]:
            with torch.cpu.amp.autocast(
                enabled=True, dtype=torch.bfloat16
            ), torch.no_grad():
                te_model = IPEXLinearAddRelu(3, 32, bias).eval()
                x = torch.randn(3, 3)
                te_model_traced = torch.jit.trace(te_model, (x))
                te_model_traced = torch.jit.freeze(te_model_traced)
                te_model_traced(x)
                # self.assertAllFused(te_model_traced.graph_for(x))

                res_jit = te_model_traced(x)
                res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )

                x = torch.randn(8, 3)
                res_jit = te_model_traced(x)
                res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )

    def test_ipex_linear_sigmoid_mul(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        print("TEST linear+sigmoid+mul")
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)
        for bias in [True, False]:
            with torch.cpu.amp.autocast(
                enabled=True, dtype=torch.bfloat16
            ), torch.no_grad():
                te_model = IPEXLinearSigmoidMul(3, 32, bias).eval()
                x = torch.randn(3, 3)
                te_model_traced = torch.jit.trace(te_model, (x))
                te_model_traced = torch.jit.freeze(te_model_traced)
                te_model_traced(x)
                # self.assertAllFused(te_model_traced.graph_for(x))

                res_jit = te_model_traced(x)
                res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )

                x = torch.randn(8, 3)
                res_jit = te_model_traced(x)
                res_imperative = te_model(x)
                self.assertEqual(
                    res_jit,
                    res_imperative,
                    rtol=0.02,
                    atol=0.01,
                    msg="{}, {}".format(res_jit, res_imperative),
                )


if __name__ == "__main__":
    # ipex._C.enable_custom_op_2_nnc_fuser()
    test = unittest.main()
