# from turtle import forward
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import copy

import intel_extension_for_pytorch  # noqa

from torch.quantization.quantize_jit import (convert_jit, prepare_jit)
from torch.jit._recursive import wrap_cpp_module

import pytest

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
print_graph = True


def conv2d_fusion(input1, input2, model, print_graph=False, dtype=torch.float):
    y = model(input1, input2)
    # print("raw: ", y)

    input1 = input1.to("xpu")
    input2 = input2.to("xpu")
    input3 = input2.clone()
    model = model.to("xpu")
    modelJit = torch.jit.script(model)
    with torch.no_grad():
        if print_graph:
            print(modelJit.graph_for(input1, input2))
        y_script = modelJit(input1, input3)
        # print("fusion:", y_script)
    del modelJit
    return y, y_script


def _conv_fusion(input1, input2, model, print_graph=False, dtype=torch.float):
    y = model(input1, input2)
    # print("half raw: ", y)
    input1 = input1.to("xpu").half()
    input2 = input2.to("xpu").half()
    model = model.to("xpu").half()
    jit_model = torch.jit.trace(model, (input1, input2), check_trace=False)
    jit_model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_model._c))
    with torch.no_grad():
        if print_graph:
            print(jit_model.graph_for(input1, input2))
        y_script = jit_model(input1, input2)
    # print("fusion: ", y_script)
    del jit_model
    return y, y_script.to(torch.float32)

class MulAdd(torch.nn.Module):
    def __init__(self) -> None:
        super(MulAdd, self).__init__()
        self.conv = nn.Conv2d(2, 2, 1, 1)

    def forward(self, input, m1, m2):
        input = F.relu(self.conv(input))
        m2 += input * 3.0
        return m2


class MatmulSum(torch.nn.Module):
    def __init__(self):
        super(MatmulSum, self).__init__()

    def forward(self, m1, m2, a):
        y = torch.matmul(m1, m2)
        y += a
        return y


class TransMatmul(torch.nn.Module):
    def __init__(self):
        super(TransMatmul, self).__init__()

    def forward(self, m1, m2):
        return torch.matmul(m1, m2.transpose(-1, -2))


class TransMatmulScale(torch.nn.Module):
    def __init__(self):
        super(TransMatmulScale, self).__init__()

    def forward(self, m1, m2):
        return torch.matmul(m1, m2.transpose(-1, -2)) / 8


class TransMatmulAddAdd(torch.nn.Module):
    def __init__(self):
        super(TransMatmulAddAdd, self).__init__()

    def forward(self, m1, m2, add1, add2):
        return torch.add(torch.matmul(m1, m2.t()), add1, alpha=2.0) + add2


class TransMatmulAdd(torch.nn.Module):
    def __init__(self):
        super(TransMatmulAdd, self).__init__()

    def forward(self, m1, m2, add1):
        output = torch.matmul(m1, m2.t())
        output += add1
        return output


class TransMatmulAddGelu(torch.nn.Module):
    def __init__(self):
        super(TransMatmulAddGelu, self).__init__()

    def forward(self, m1, m2, add):
        return F.gelu(torch.add(torch.matmul(m1, m2.t()), add, alpha=2.0))


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return F.relu(self.conv(x))


class Conv2dSum(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dSum, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        x = self.conv(x).add_(a)
        return x


class Conv2dSumRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dSumRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return F.relu(self.conv(x).add_(a))


class Conv2dAbs(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dAbs, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.abs(self.conv(x))


class Conv2dLeakyrelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dLeakyrelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.neg = random.random()

    def forward(self, x, a):
        return F.leaky_relu(self.conv(x), self.neg)


class Conv2dSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dSigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.sigmoid(self.conv(x))


class Conv2dSqrt(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dSqrt, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))

    def forward(self, x, a):
        return torch.sqrt(self.conv(x))


class Conv2dTanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dTanh, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))

    def forward(self, x, a):
        return torch.tanh(self.conv(x))


class Conv2dSquare(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dSquare, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))

    def forward(self, x, a):
        return torch.square(self.conv(x))


class Conv2dExp(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dExp, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))

    def forward(self, x, a):
        return torch.exp(self.conv(x))


class Conv2dLog(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dLog, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))

    def forward(self, x, a):
        return torch.log(self.conv(x))


class Conv2dRound(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dRound, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))

    def forward(self, x, a):
        return torch.round(self.conv(x))


class Conv2dLogSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dLogSigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.LogSigmoid()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dHardswish(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dHardswish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.Hardswish()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dMish(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.Mish()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dSilu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dSilu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.SiLU()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dGelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dGelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.GELU()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dHardsigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dHardsigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.Hardsigmoid()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dPow(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dPow, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.exponent = 1.0

    def forward(self, x, a):
        x = self.conv(x)
        # print("x:res: ", x)
        return torch.pow(x, self.exponent)


class Conv2dRelu6(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dRelu6, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.ReLU6()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dElu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dElu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv.weight = torch.nn.parameter.Parameter(torch.abs(self.conv.weight))
        self.activation = torch.nn.ELU()

    def forward(self, x, a):
        return self.activation(self.conv(x))


class Conv2dBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMul, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.mul(self.conv(x), a)


class PadConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PadConv2d, self).__init__()
        self.pad = nn.ConstantPad2d((0, 1, 0, 2), 0.0)
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)


class PermuteContiguous(torch.nn.Module):
    def __init__(self) -> None:
        super(PermuteContiguous, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 126, (1, 1))
        )

    def forward(self, x):
        x = self.block(x)
        x = torch.permute(x, [0, 2, 3, 1])
        return x.contiguous()


class LinearGELU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearGELU, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.linear(x))
        return x


class LinearAdd(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearAdd, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x1 = torch.ones(x.shape).to(x.device)
        x = self.linear(x)
        y = x + x1
        return y


class LinearReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x))
        return x


class LinearSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSigmoid, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear(x))
        return x

class LinearSqrt(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSqrt, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.sqrt(self.linear(x))
        return x

class LinearSquare(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSquare, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.square(self.linear(x))
        return x

class LinearAbs(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearAbs, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.abs(self.linear(x))
        return x

class LinearExp(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearExp, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.exp(self.linear(x))
        return x

class LinearLog(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLog, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.log(self.linear(x))
        return x

class LinearRound(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearRound, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.round(self.linear(x))
        return x

class LinearSilu(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSilu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = F.silu(self.linear(x))
        return x

class LinearLogSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLogSigmoid, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = F.logsigmoid(self.linear(x))
        return x

class LinearHardswish(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHardswish, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = F.hardswish(self.linear(x))
        return x

class LinearMish(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearMish, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.activation = torch.nn.Mish()

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x

class LinearHardSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHardSigmoid, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = F.hardsigmoid(self.linear(x))
        return x

class LinearTanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearTanh, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.tanh(self.linear(x))
        return x

class LinearLeakyRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLeakyRelu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.linear(x), 0.01)
        return x

class LinearPow(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearPow, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.pow(self.linear(x), 2)
        return x

class LinearHardtanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHardtanh, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.activation = nn.ReLU6()

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x

class LinearElu(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearElu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = F.elu(self.linear(x), 1.2)
        return x

class TestNNMethod(TestCase):
    def test_matmul_sum_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)
        acc = torch.randn([2], device=cpu_device)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        acc_dpcpp = acc.to(dpcpp_device)
        model = MatmulSum()
        raw = model(m1, m2, acc)
        print("raw: ", raw)
        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real = modelJit(m1_dpcpp, m2_dpcpp, acc_dpcpp)
            print("real: ", real.cpu())
        self.assertEqual(raw, real.to(cpu_device))
        del modelJit

    def test_trans_baddbmm_fusion(self, dtype=torch.float):
        m1 = torch.randn((2, 2, 3), device=cpu_device)
        m2 = torch.randn((2, 2, 3), device=cpu_device)

        model = TransMatmul()
        raw1 = model(m1, m2)
        raw2 = model(m1, m2)
        print("raw1: ", raw1)
        print("raw2: ", raw2)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real1 = modelJit(m1_dpcpp, m2_dpcpp)
            real2 = modelJit(m1_dpcpp, m2_dpcpp)
            print("real1:", real1.to(cpu_device))
            print("real2:", real2.to(cpu_device))
        self.assertEqual(raw1, real1.to(cpu_device))
        self.assertEqual(raw2, real2.to(cpu_device))
        del modelJit

    def test_trans_baddbmm_scale_fusion(self, dtype=torch.float):
        m1 = torch.randn((2, 2, 3), device=cpu_device)
        m2 = torch.randn((2, 2, 3), device=cpu_device)

        model = TransMatmulScale()
        raw1 = model(m1, m2)
        raw2 = model(m1, m2)
        print("raw1: ", raw1)
        print("raw2: ", raw2)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real1 = modelJit(m1_dpcpp, m2_dpcpp)
            real2 = modelJit(m1_dpcpp, m2_dpcpp)
            print("real1:", real1.to(cpu_device))
            print("real2:", real2.to(cpu_device))
        self.assertEqual(raw1, real1.to(cpu_device))
        self.assertEqual(raw2, real2.to(cpu_device))
        del modelJit


    def test_mul_add(self, dtype=torch.float):
        m1 = torch.randn((4, 2, 2, 2), device=cpu_device)
        m2 = torch.randn((4, 2, 2, 2), device=cpu_device)
        add1 = torch.randn((4, 2, 2, 2), device=cpu_device)
        add2 = add1.clone()

        model = MulAdd()
        model1 = copy.deepcopy(model)
        raw = model(m1, m2, add1)
        print("raw: ", raw)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        add1_dpcpp = add2.to(dpcpp_device)
        model1 = model1.to("xpu")

        modelJit = torch.jit.script(model1)
        with torch.no_grad():
            print(modelJit.graph_for(m1_dpcpp, m2_dpcpp, add1_dpcpp))
            real = modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp)
            print("real:", real.to(cpu_device))
        self.assertEqual(raw, real.to(cpu_device))
        del modelJit

    def test_trans_matmul_add(self, dtype=torch.float):
        m1 = torch.randn((4, 2), device=cpu_device)
        m2 = torch.randn((4, 2), device=cpu_device)
        add1 = torch.randn((4, 4), device=cpu_device)

        model = TransMatmulAdd()
        raw = model(m1, m2, add1)
        print("raw: ", raw)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        add1_dpcpp = add1.to(dpcpp_device)

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real = modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp)
            print("real:", real.to(cpu_device))
        self.assertEqual(raw, real.to(cpu_device))
        del modelJit

    def test_trans_matmul_add_add(self, dtype=torch.float):
        m1 = torch.randn((4, 2), device=cpu_device)
        m2 = torch.randn((4, 2), device=cpu_device)
        add1 = torch.randn((4, 4), device=cpu_device)
        add2 = torch.randn((4, 4), device=cpu_device)

        model = TransMatmulAddAdd()
        raw = model(m1, m2, add1, add2)
        print("raw: ", raw)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        add1_dpcpp = add1.to(dpcpp_device)
        add2_dpcpp = add2.to(dpcpp_device)

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real = modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp, add2_dpcpp)
            print("real:", real.to(cpu_device))
        self.assertEqual(raw, real.to(cpu_device))
        del modelJit

    def test_trans_3d_matmul_add_add(self, dtype=torch.float):
        m1 = torch.randn((4, 3, 2), device=cpu_device)
        m2 = torch.randn((4, 2), device=cpu_device)
        add1 = torch.randn((4, 3, 4), device=cpu_device)
        add2 = torch.randn((4), device=cpu_device)

        model = TransMatmulAddAdd()
        raw = model(m1, m2, add1, add2)
        print("raw: ", raw)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        add1_dpcpp = add1.to(dpcpp_device)
        add2_dpcpp = add2.to(dpcpp_device)

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real = modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp, add2_dpcpp)
            print("real:", real.to(cpu_device))
        self.assertEqual(raw, real.to(cpu_device))
        del modelJit

    def test_trans_matmul_gelu(self, dtype=torch.float):
        m1 = torch.randn((4, 2), device=cpu_device)
        m2 = torch.randn((4, 2), device=cpu_device)
        add = torch.randn((4, 4), device=cpu_device)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        add_dpcpp = add.to(dpcpp_device)

        model = TransMatmulAddGelu()
        model_dpcpp = model.to(dpcpp_device)
        raw = model_dpcpp(m1_dpcpp, m2_dpcpp, add_dpcpp)
        print("raw: ", raw.cpu())

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            real = modelJit(m1_dpcpp, m2_dpcpp, add_dpcpp)
            print("real:", real.to(cpu_device))
            print(modelJit.graph_for(m1_dpcpp, m2_dpcpp, add_dpcpp))
        self.assertEqual(raw, real.to(cpu_device), atol=1e-3, rtol=1.3e-6)
        del modelJit

    def test_conv_sqrt_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)
        x = torch.abs(x)
        model = Conv2dSqrt(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_abs_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dAbs(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_relu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dRelu(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_sum_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSum(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_sum_relu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSumRelu(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_sigmoid_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSigmoid(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_leaky_relu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dLeakyrelu(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_tanh_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dTanh(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_square_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSquare(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_exp_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dExp(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_log_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)
        x = torch.abs(x)
        model = Conv2dLog(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_round_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dRound(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_logsigmoid_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dLogSigmoid(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_hardswish_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dHardswish(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_mish_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dMish(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_silu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSilu(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_gelu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dGelu(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script, atol=1e5, rtol=1e5)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_hardsigmoid_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dHardsigmoid(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_pow_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dPow(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_hardtanh_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dRelu6(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_elu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dElu(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_mul(self, dtype=torch.float):
        x = torch.randn([1, 64, 512, 512], device=cpu_device)
        a1 = torch.randn([1, 64, 512, 512], device=cpu_device)
        a2 = torch.randn([1, 64, 1, 1], device=cpu_device)
        a3 = torch.randn([1, 1, 1, 1], device=cpu_device)
        model = Conv2dBinaryMul(64, 64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        other = [a1, a2, a3]
        for a in other:
            y = model(x, a)

            x_xpu = x.clone().to("xpu")
            a_xpu = a.clone().to("xpu")
            model_xpu = copy.deepcopy(model).to("xpu")
            modelJit = torch.jit.script(model_xpu)
            with torch.no_grad():
                y_dpcpp = modelJit(x_xpu, a_xpu)
                # print(modelJit.graph_for(x, a))
                # print("fusion:", y_dpcpp.cpu())
            self.assertEqual(y, y_dpcpp.to(cpu_device))
            del modelJit

    def test_pad_conv_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)

        model = PadConv2d(2, 2, kernel_size=1, stride=1, padding=(1, 2), bias=True)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.script(model)
        # modelJit.to("xpu")
        with torch.no_grad():
            # print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    @pytest.mark.skip("quantize convolution have some misalignment with pytorch")
    def test_permute_contiguous_fusion(self, dtype=torch.float):
        model = PermuteContiguous()
        input_cpu = torch.rand([1, 32, 128, 128])
        input_xpu = input_cpu.clone().to("xpu")

        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)

        # cpu int8
        modelJit = torch.jit.trace(model, input_cpu)
        modelJit.eval()
        print(modelJit)
        print("finish jit...")

        print("start calibration ...")
        qconfig_u8 = torch.quantization.QConfig(
            activation=torch.quantization.observer.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
                dtype=torch.quint8
            ),
            weight=torch.quantization.default_weight_observer
        )

        modelJit = prepare_jit(modelJit, {'': qconfig_u8}, True)

        # do calibration
        for i in range(1):
            calib_input = input_cpu
            modelJit(calib_input)
        print("start cpu convert")
        modelJit = convert_jit(modelJit, True)
        print(modelJit.graph_for(input_cpu))
        print("--modelJit={}".format(modelJit))

        # inference
        print("start inference ...")
        for i in range(5):
            output_cpu = modelJit(input_cpu)

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # xpu
        print("-------start xpu path-------")
        print("start jit ...")
        model = model.to("xpu")
        model = torch.jit.trace(model, input_cpu.to("xpu"))
        modelJit = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
        modelJit.eval()
        print("finish jit ...")

        modelJit = modelJit.to("xpu")
        print("start calibration ...")
        # calibration
        # default with per_tensor quantization
        qconfig_u8 = torch.quantization.QConfig(
            activation=torch.quantization.observer.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
                dtype=torch.quint8
            ),
            weight=torch.quantization.default_weight_observer
        )

        modelJit = prepare_jit(modelJit, {'': qconfig_u8}, True)
        modelJit = modelJit.to("xpu")

        # do calibration
        for i in range(1):
            calib_input = input_xpu
            print(calib_input.size())
            modelJit(calib_input)
        modelJit = convert_jit(modelJit, True)
        print(modelJit.graph_for(input_xpu))

        print("start inference ...")
        for i in range(5):

            output = modelJit(input_xpu)
            torch.xpu.synchronize()
        self.assertEqual(output.cpu(), output_cpu)

    def test_linear_relu_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearReLU(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_gelu_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearGELU(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device), atol=1e-3, rtol=1.3e-6)
        del modelJit

    def test_linear_add_fusion(self, dtype=torch.float):
        x = torch.randn([1, 384, 1024], device=cpu_device)
        model = LinearAdd(1024, 1024)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device), atol=1e-3, rtol=1.3e-6)
        del modelJit

    def test_linear_sigmoid_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearSigmoid(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_sqrt_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearSqrt(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_square_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearSquare(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_abs_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearAbs(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_exp_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearExp(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_log_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearLog(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_round_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearRound(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_silu_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearSilu(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_logsigmoid_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearLogSigmoid(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_hardswish_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearHardswish(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    @pytest.mark.skip("oneDNN's post-op implementation will get incorrect result on this case. need skip")
    def test_linear_mish_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearMish(4, 4)
        print("raw model input: ", x)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.script(model)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            print("jit model input: ", x)
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_hardsigmoid_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearHardSigmoid(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    @pytest.mark.skip("oneDNN's post-op implementation will get incorrect result on this case. need skip")
    def test_linear_tanh_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearTanh(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x, check_trace=True)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_leakyrelu_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearLeakyRelu(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_pow_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearPow(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))


    def test_linear_elu_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearElu(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_hardtanh(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearHardtanh(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit
