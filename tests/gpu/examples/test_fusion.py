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
        for i in range(5):
            y_script = modelJit(input1, input2)
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
    input3 = input2.clone()
    model = model.to("xpu").half()
    jit_model = torch.jit.trace(model, (input1, input2), check_trace=False)
    jit_model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_model._c))
    with torch.no_grad():
        for i in range(5):
            y_script = jit_model(input1, input2)
        if print_graph:
            print(jit_model.graph_for(input1, input2))
        y_script = jit_model(input1, input3)
    # print("fusion: ", y_script)
    del jit_model
    return y, y_script.to(torch.float32)


class MulAddScalar(torch.nn.Module):
    def __init__(self) -> None:
        super(MulAddScalar, self).__init__()
        self.conv = nn.Conv2d(2, 2, 1, 1)

    def forward(self, input, m1, m2):
        input = F.relu(self.conv(input))
        ret = input * 3.0 + m2
        return ret

class MulAdd(torch.nn.Module):
    def __init__(self) -> None:
        super(MulAdd, self).__init__()
        self.conv = nn.Conv2d(2, 2, 1, 1)

    def forward(self, input, m1, m2):
        ret = F.relu(self.conv(input))
        ret = ret * m1 + m2
        return ret

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class MatmulSum(torch.nn.Module):
    def __init__(self):
        super(MatmulSum, self).__init__()

    def forward(self, m1, m2, a):
        y = torch.matmul(m1, m2)
        y += a
        return y

class MatmulSqrt(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulSqrt, self).__init__()

    def forward(self, m1, m2):
        return  torch.sqrt(torch.matmul(m1, m2))

class TMatmulSqrt(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulSqrt, self).__init__()

    def forward(self, m1, m2):
        return torch.sqrt(torch.matmul(m1, m2.t()))
class MatmulAbs(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulAbs, self).__init__()

    def forward(self, m1, m2):
        return  torch.abs(torch.matmul(m1, m2))

class TMatmulAbs(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulAbs, self).__init__()

    def forward(self, m1, m2):
        return  torch.abs(torch.matmul(m1, m2.t()))

class MatmulTanh(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulTanh, self).__init__()

    def forward(self, m1, m2):
        return  torch.tanh(torch.matmul(m1, m2))

class TMatmulTanh(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulTanh, self).__init__()

    def forward(self, m1, m2):
        return  torch.tanh(torch.matmul(m1, m2.t()))

class MatmulSquare(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulSquare, self).__init__()

    def forward(self, m1, m2):
        return  torch.square(torch.matmul(m1, m2))

class TMatmulSquare(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulSquare, self).__init__()

    def forward(self, m1, m2):
        return  torch.square(torch.matmul(m1, m2.t()))

class MatmulRelu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulRelu, self).__init__()

    def forward(self, m1, m2):
        return  torch.relu(torch.matmul(m1, m2))

class TMatmulRelu(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulRelu, self).__init__()

    def forward(self, m1, m2):
        return  torch.relu(torch.matmul(m1, m2.t()))

class MatmulSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulSigmoid, self).__init__()

    def forward(self, m1, m2):
        return  torch.sigmoid(torch.matmul(m1, m2))

class TMatmulSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulSigmoid, self).__init__()

    def forward(self, m1, m2):
        return  torch.sigmoid(torch.matmul(m1, m2.t()))

class MatmulExp(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulExp, self).__init__()

    def forward(self, m1, m2):
        return  torch.exp(torch.matmul(m1, m2))

class TMatmulExp(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulExp, self).__init__()

    def forward(self, m1, m2):
        return  torch.exp(torch.matmul(m1, m2.t()))

class MatmulLog(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulLog, self).__init__()

    def forward(self, m1, m2):
        return  torch.log(torch.matmul(m1, m2))

class TMatmulLog(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulLog, self).__init__()

    def forward(self, m1, m2):
        return  torch.log(torch.matmul(m1, m2.t()))

class MatmulRound(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulRound, self).__init__()

    def forward(self, m1, m2):
        return  torch.round(torch.matmul(m1, m2))

class TMatmulRound(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulRound, self).__init__()

    def forward(self, m1, m2):
        return  torch.round(torch.matmul(m1, m2.t()))

class MatmulLogsigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulLogsigmoid, self).__init__()

    def forward(self, m1, m2):
        return  F.logsigmoid(torch.matmul(m1, m2))

class TMatmulLogsigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulLogsigmoid, self).__init__()

    def forward(self, m1, m2):
        return  F.logsigmoid(torch.matmul(m1, m2.t()))

class MatmulHardsiwsh(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulHardsiwsh, self).__init__()

    def forward(self, m1, m2):
        return  F.hardswish(torch.matmul(m1, m2))

class TMatmulHardswish(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulHardswish, self).__init__()

    def forward(self, m1, m2):
        return  F.hardswish(torch.matmul(m1, m2.t()))

class MatmulMish(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulMish, self).__init__()

    def forward(self, m1, m2):
        return  F.mish(torch.matmul(m1, m2))

class TMatmulMish(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulMish, self).__init__()

    def forward(self, m1, m2):
        return  F.mish(torch.matmul(m1, m2.t()))

class MatmulSilu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulSilu, self).__init__()

    def forward(self, m1, m2):
        return  F.silu(torch.matmul(m1, m2))

class TMatmulSilu(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulSilu, self).__init__()

    def forward(self, m1, m2):
        return  F.silu(torch.matmul(m1, m2.t()))

class MatmulGelu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulGelu, self).__init__()

    def forward(self, m1, m2):
        return  F.gelu(torch.matmul(m1, m2))

class TMatmulGelu(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulGelu, self).__init__()

    def forward(self, m1, m2):
        return  F.gelu(torch.matmul(m1, m2.t()))

class MatmulHardsigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulHardsigmoid, self).__init__()

    def forward(self, m1, m2):
        return  F.hardsigmoid(torch.matmul(m1, m2))

class TMatmulHardsigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulHardsigmoid, self).__init__()

    def forward(self, m1, m2):
        return  F.hardsigmoid(torch.matmul(m1, m2.t()))

class MatmulElu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulElu, self).__init__()

    def forward(self, m1, m2):
        return  F.elu(torch.matmul(m1, m2))

class TMatmulElu(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulElu, self).__init__()

    def forward(self, m1, m2):
        return  F.elu(torch.matmul(m1, m2.t()))

class MatmulPow(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulPow, self).__init__()

    def forward(self, m1, m2):
        return  torch.pow(torch.matmul(m1, m2), 2.0)

class TMatmulPow(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulPow, self).__init__()

    def forward(self, m1, m2):
        return  F.elu(torch.matmul(m1, m2.t()), 2.0)

class MatmulLeakyrelu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulLeakyrelu, self).__init__()

    def forward(self, m1, m2):
        return  F.leaky_relu(torch.matmul(m1, m2), 0.01)

class TMatmulLeakyrelu(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulLeakyrelu, self).__init__()

    def forward(self, m1, m2):
        return  F.leaky_relu(torch.matmul(m1, m2.t()), 0.01)

class MatmulRelu6(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulRelu6, self).__init__()

    def forward(self, m1, m2):
        return  F.relu6(torch.matmul(m1, m2))

class TMatmulRelu6(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulRelu6, self).__init__()

    def forward(self, m1, m2):
        return  F.relu6(torch.matmul(m1, m2.t()))

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
        x = self.conv(x).add(a)
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


class Conv2dSigmoidBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dSigmoidBinaryMul, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        ret = torch.sigmoid(self.conv(x))
        return ret * a

class Conv2dSigmoidBinaryMulAdd(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dSigmoidBinaryMulAdd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        ret = torch.sigmoid(self.conv(x))
        return ret * a + x

class Conv2dSigmoidBinaryMulAddRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dSigmoidBinaryMulAddRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        ret = torch.sigmoid(self.conv(x))
        return torch.relu(ret * a + x)

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

class Conv2dMishYolo(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dMishYolo, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = Mish()

    def forward(self, x, a):
        return self.activation(self.conv(x))

class Conv2dMishAddYolo(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv2dMishAddYolo, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = Mish()

    def forward(self, x, a):
        return self.activation(self.conv(x)) + a


class Conv2dBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMul, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.mul(self.conv(x), x)

class Conv2dBinaryMulScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMulScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.mul(self.conv(x), 3.)

class Conv2dBinaryMulAdd(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMulAdd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.mul(self.conv(x), a) + a


class Conv2dBinaryDiv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryDiv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.div(self.conv(x), x)

class Conv2dBinaryDivScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryDivScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.div(self.conv(x), 3.)

class Conv2dBinarySub(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinarySub, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return self.conv(x) - a

class Conv2dBinarySubScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinarySubScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return self.conv(x) - 3.

class Conv2dBinaryEqual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryEqual, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.eq(self.conv(x), a)

class Conv2dBinaryEqualScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryEqualScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.eq(self.conv(x), 3.)

class Conv2dBinaryNotEqual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryNotEqual, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.ne(self.conv(x), a)

class Conv2dBinaryNotEqualScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryNotEqualScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.ne(self.conv(x), 3.)

class Conv2dBinaryMin(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMin, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.min(self.conv(x), a)

class Conv2dBinaryMinScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMinScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.min(self.conv(x), torch.tensor(3.))

class Conv2dBinaryMax(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMax, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.max(self.conv(x), a)

class Conv2dBinaryMaxScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryMaxScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.max(self.conv(x), torch.tensor(3.))

class Conv2dBinaryGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryGE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.ge(self.conv(x), a)

class Conv2dBinaryGEScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryGEScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.ge(self.conv(x), 3.)

class Conv2dBinaryGT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryGT, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.gt(self.conv(x), a)

class Conv2dBinaryGTScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryGTScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.gt(self.conv(x), 3.)

class Conv2dBinaryLE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryLE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.le(self.conv(x), a)

class Conv2dBinaryLEScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryLEScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.le(self.conv(x), 3.)

class Conv2dBinaryLT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryLT, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.lt(self.conv(x), a)

class Conv2dBinaryLTScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBinaryLTScalar, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.lt(self.conv(x), 3.)


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
            nn.Conv2d(2, 2, (1, 1))
        )

    def forward(self, x):
        x = self.block(x)
        x = torch.permute(x, [0, 2, 3, 1])
        return x.contiguous()


class LinearGELU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearGELU, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.linear(x))
        return x


class LinearAdd(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearAdd, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x1 = torch.ones(x.shape).to(x.device)
        x = self.linear(x)
        y = x + x1
        return y


class LinearReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x))
        return x


class LinearSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSigmoid, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear(x))
        return x

class LinearSqrt(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSqrt, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.sqrt(self.linear(x))
        return x

class LinearSquare(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSquare, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.square(self.linear(x))
        return x

class LinearAbs(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearAbs, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.abs(self.linear(x))
        return x

class LinearExp(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearExp, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.exp(self.linear(x))
        return x

class LinearLog(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLog, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.log(self.linear(x))
        return x

class LinearRound(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearRound, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.round(self.linear(x))
        return x

class LinearSilu(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSilu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = F.silu(self.linear(x))
        return x

class LinearLogSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLogSigmoid, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = F.logsigmoid(self.linear(x))
        return x

class LinearHardswish(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHardswish, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = F.hardswish(self.linear(x))
        return x

class LinearMish(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearMish, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Mish()

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x

class LinearHardSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHardSigmoid, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = F.hardsigmoid(self.linear(x))
        return x

class LinearTanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearTanh, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.tanh(self.linear(x))
        return x

class LinearLeakyRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLeakyRelu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.linear(x), 0.01)
        return x

class LinearPow(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearPow, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = torch.pow(self.linear(x), 2)
        return x

class LinearHardtanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHardtanh, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU6()

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x

class LinearElu(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearElu, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = F.elu(self.linear(x), 1.2)
        return x

class LinearBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryMul, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.mul(self.linear(x), a)

class LinearBinaryMulScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryMulScalar, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.mul(self.linear(x), 3.0)

class LinearBinaryDiv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryDiv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.div(self.linear(x), a)

class LinearBinaryDivScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryDivScalar, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.div(self.linear(x), 3.0)

class LinearBinarySub(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinarySub, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return self.linear(x) - a

class LinearBinarySubScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinarySubScalar, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return self.linear(x) - 3.0

class LinearBinaryAdd(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryAdd, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return self.linear(x) + a

class LinearBinaryAddScalar(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryAddScalar, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return self.linear(x) + 3.0

class LinearBinaryEqual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryEqual, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.eq(self.linear(x), a)

class LinearBinaryNotEqual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryNotEqual, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.ne(self.linear(x), a)

class LinearBinaryMin(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryMin, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.min(self.linear(x), a)

class LinearBinaryMax(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryMax, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.max(self.linear(x), a)

class LinearBinaryGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryGE, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.ge(self.linear(x), a)

class LinearBinaryGT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryGT, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.gt(self.linear(x), a)

class LinearBinaryLE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryLE, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.le(self.linear(x), a)

class LinearBinaryLT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryLT, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return torch.lt(self.linear(x), a)

class TestNNMethod(TestCase):

    def matmul_fusion(self, model, t_model, m1, m2):
        raw = model(m1, m2)
        raw_t = t_model(m1, m2)
        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        model = model.to(dpcpp_device)
        t_model = t_model.to(dpcpp_device)
        modelJit = torch.jit.script(model)
        tmodelJit = torch.jit.script(t_model)
        for i in range(2):
            modelJit(m1_dpcpp, m2_dpcpp)
            tmodelJit(m1_dpcpp, m2_dpcpp)
        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(m1_dpcpp, m2_dpcpp))
                print(tmodelJit.graph_for(m1_dpcpp, m2_dpcpp))
            real = modelJit(m1_dpcpp, m2_dpcpp)
            t_real = tmodelJit(m1_dpcpp, m2_dpcpp)
        self.assertEqual(raw, real.to(cpu_device))
        self.assertEqual(raw_t, t_real.to(cpu_device))

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
    
    def test_matmul_sqrt_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)
        model = MatmulSqrt()
        t_model = TMatmulSqrt()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_square_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulSquare()
        t_model = TMatmulSquare()
        self.matmul_fusion(model, t_model, m1, m2)


    def test_matmul_abs_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulAbs()
        t_model = TMatmulAbs()
        self.matmul_fusion(model, t_model, m1, m2)


    def test_matmul_exp_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulExp()
        t_model = TMatmulExp()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_log_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulLog()
        t_model = TMatmulLog()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_round_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulRound()
        t_model = TMatmulRound()
        self.matmul_fusion(model, t_model, m1, m2)
        
    # result incorrect
    def test_matmul_silu_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulSilu()
        t_model = TMatmulSilu()
        self.matmul_fusion(model, t_model, m1, m2)

    # result incorrect
    def test_matmul_gelu_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulGelu()
        t_model = TMatmulGelu()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_log_sigmoid_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulLogsigmoid()
        t_model = TMatmulLogsigmoid()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_hardswish_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulHardsiwsh()
        t_model = TMatmulHardswish()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_mish_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulMish()
        t_model = TMatmulMish()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_hardsigmoid_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulHardsigmoid()
        t_model = TMatmulHardsigmoid()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_tanh_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulTanh()
        t_model = TMatmulTanh()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_leaky_relu_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulLeakyrelu()
        t_model = TMatmulLeakyrelu()
        self.matmul_fusion(model, t_model, m1, m2)  

    def test_matmul_pow_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulPow()
        t_model = TMatmulPow()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_elu_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulElu()
        t_model = TMatmulElu()
        self.matmul_fusion(model, t_model, m1, m2)

    # op different 
    def test_matmul_hardtanh_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulRelu6()
        t_model = TMatmulRelu6()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_sigmoid_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulSigmoid()
        t_model = TMatmulSigmoid()
        self.matmul_fusion(model, t_model, m1, m2)

    def test_matmul_relu_fusion(self, dtype=torch.float):
        m1 = torch.randn([4, 2], device=cpu_device)
        m2 = torch.randn([2, 2], device=cpu_device)

        model = MatmulRelu()
        t_model = TMatmulRelu()
        self.matmul_fusion(model, t_model, m1, m2)

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
        model_scalar = MulAddScalar()
        model_scalar1 = copy.deepcopy(model_scalar)
        raw = model(m1, m2, add1)
        raw_scalar = model_scalar(m1, m2, add1)
        print("raw: ", raw)

        m1_dpcpp = m1.to(dpcpp_device)
        m2_dpcpp = m2.to(dpcpp_device)
        add1_dpcpp = add2.to(dpcpp_device)
        add2_dpcpp = add1_dpcpp.clone()
        model1 = model1.to("xpu")
        model_scalar1 = model_scalar1.to("xpu")

        modelJit = torch.jit.script(model1)
        modelJit_scalar = torch.jit.script(model_scalar1)
        with torch.no_grad():
            for i in range(5):
                modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp)
                modelJit_scalar(m1_dpcpp, m2_dpcpp, add1_dpcpp)
            print(modelJit.graph_for(m1_dpcpp, m2_dpcpp, add1_dpcpp))
            print(modelJit_scalar.graph_for(m1_dpcpp, m2_dpcpp, add1_dpcpp))
            real = modelJit(m1_dpcpp, m2_dpcpp, add2_dpcpp)
            real_scalar = modelJit_scalar(m1_dpcpp, m2_dpcpp, add2_dpcpp)
            print("real:", real.to(cpu_device))
        self.assertEqual(raw, real.to(cpu_device))
        self.assertEqual(raw_scalar, real_scalar.to(cpu_device))
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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)


    def test_conv_sigmoid_binary_mul_add_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 3, 3], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSigmoidBinaryMulAdd(2, 2, kernel_size=1, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

    def test_conv_sigmoid_binary_mul_add_relu_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 3, 3], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSigmoidBinaryMulAddRelu(2, 2, kernel_size=1, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

    def test_conv_sigmoid_binary_mul_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dSigmoidBinaryMul(2, 2, kernel_size=1, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_conv_mish_yolo_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dMishYolo(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_conv_mish_add_yolo_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dMishAddYolo(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e-1, rtol=1e-1)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

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
        self.assertEqual(y, y_script, atol=1e-3, rtol=1e-3)

    def test_conv_binary_mul_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryMul(2, 2, kernel_size=1, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryMulScalar(2, 2, kernel_size=1, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_mul_add_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryMulAdd(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_sub_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinarySub(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinarySubScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_div_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryDiv(2, 2, kernel_size=1, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryDivScalar(2, 2, kernel_size=1, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_eq_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryEqual(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryEqualScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)

    def test_conv_binary_ne_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryNotEqual(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryNotEqualScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)

    def test_conv_binary_max_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryMax(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryMaxScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_min_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryMin(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryMinScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script)
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script, atol=1e3, rtol=1e3)

    def test_conv_binary_gt_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryGT(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryGTScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)

    def test_conv_binary_ge_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryGE(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryGEScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)

    def test_conv_binary_lt_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryLT(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryLTScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)

    def test_conv_binary_le_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 8, 8], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dBinaryLE(2, 2, kernel_size=3, stride=1, bias=True)
        model1 = copy.deepcopy(model)
        model_scalar = Conv2dBinaryLEScalar(2, 2, kernel_size=3, stride=1, bias=True)
        model_scalar1 = copy.deepcopy(model_scalar)
        y, y_script = conv2d_fusion(x, a1, model, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)
        y, y_script = conv2d_fusion(x, a1, model_scalar, print_graph)
        self.assertEqual(y, y_script.bool())
        y, y_script = _conv_fusion(x, a1, model_scalar1, print_graph)
        self.assertEqual(y, y_script.bool(), atol=1e3, rtol=1e3)

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

    def test_permute_contiguous_fusion(self, dtype=torch.float):
        model = PermuteContiguous()
        input_cpu = torch.rand([1, 2, 9, 9])
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
                    print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

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
            for i in range(3):
                if print_graph and i==2:
                    print(modelJit.graph_for(x))
            print("jit model input: ", x.cpu())
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
            for i in range(3):
                if print_graph and i==2:
                    print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    def test_linear_tanh_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        model = LinearTanh(4, 4)
        y = model(x)
        print("raw: ", y)

        x = x.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, x)

        with torch.no_grad():
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
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
            for i in range(3):
                if print_graph and i==2:
                    print(modelJit.inlined_graph)
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
            for i in range(3):
                if print_graph and i==2:
                    print(modelJit.graph_for(x))
            y_dpcpp = modelJit(x)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit


    def test_linear_binary_mul_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        a = torch.randn([2, 4], device=cpu_device)
        model = LinearBinaryMul(4, 4)
        model_scalar = LinearBinaryMulScalar(4, 4)
        y = model(x, a)
        y_scalar = model_scalar(x, a)
        print("raw: ", y)

        x = x.to("xpu")
        a = a.to("xpu")
        model.to("xpu")
        model_scalar.to("xpu")
        modelJit = torch.jit.script(model)
        modelJit_scalar = torch.jit.script(model_scalar)

        with torch.no_grad(): 
            for i in range(2):
                modelJit(x, a)
                modelJit_scalar(x, a)
            if print_graph:
                print(modelJit.graph_for(x, a))
                print(modelJit_scalar.graph_for(x, a))
            y_dpcpp = modelJit(x, a)
            y_dpcpp_scalar = modelJit_scalar(x, a)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        self.assertEqual(y_scalar, y_dpcpp_scalar.to(cpu_device))
        del modelJit

    def test_linear_binary_sub_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        a = torch.randn([2, 4], device=cpu_device)
        model = LinearBinarySub(4, 4)
        model_scalar = LinearBinarySubScalar(4, 4)
        y = model(x, a)
        y_scalar = model_scalar(x, a)
        print("raw: ", y)

        x = x.to("xpu")
        a = a.to("xpu")
        a_ = a.clone()
        model.to("xpu")
        model_scalar.to("xpu")
        modelJit = torch.jit.script(model)
        modelJit_scalar = torch.jit.script(model_scalar)

        with torch.no_grad(): 
            for i in range(2):
                modelJit(x, a)
                modelJit_scalar(x, a)
            if print_graph:
                print(modelJit.graph_for(x, a))
                print(modelJit_scalar.graph_for(x, a))
            y_dpcpp = modelJit(x, a_)
            y_dpcpp_scalar = modelJit_scalar(x, a_)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        self.assertEqual(y_scalar, y_dpcpp_scalar.to(cpu_device))
        del modelJit

    def test_linear_binary_div_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        a = torch.randn([2, 4], device=cpu_device)
        model = LinearBinaryDiv(4, 4)
        model_scalar = LinearBinaryDivScalar(4, 4)
        y = model(x, a)
        y_scalar = model_scalar(x, a)
        print("raw: ", y)

        x = x.to("xpu")
        a = a.to("xpu")
        model.to("xpu")
        model_scalar.to("xpu")
        modelJit = torch.jit.script(model)
        modelJit_scalar = torch.jit.script(model_scalar)

        with torch.no_grad(): 
            for i in range(2):
                modelJit(x, a)
                modelJit_scalar(x, a)
            if print_graph:
                print(modelJit.graph_for(x, a))
                print(modelJit_scalar.graph_for(x, a))
            y_dpcpp = modelJit(x, a)
            y_dpcpp_scalar = modelJit_scalar(x, a)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        self.assertEqual(y_scalar, y_dpcpp_scalar.to(cpu_device))
        del modelJit

    def test_linear_binary_add_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        a = torch.randn([2, 4], device=cpu_device)
        model = LinearBinaryAdd(4, 4)
        model_scalar = LinearBinaryAddScalar(4, 4)
        y = model(x, a)
        y_scalar = model_scalar(x, a)
        print("raw: ", y)

        x = x.to("xpu")
        a = a.to("xpu")
        a_ = a.clone()
        model.to("xpu")
        model_scalar.to("xpu")
        modelJit = torch.jit.script(model)
        modelJit_scalar = torch.jit.script(model_scalar)

        with torch.no_grad(): 
            for i in range(2):
                modelJit(x, a)
                modelJit_scalar(x, a)
            if print_graph:
                print(modelJit.graph_for(x, a))
                print(modelJit_scalar.graph_for(x, a))
            y_dpcpp = modelJit(x, a_)
            y_dpcpp_scalar = modelJit_scalar(x, a_)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        self.assertEqual(y_scalar, y_dpcpp_scalar.to(cpu_device))
        del modelJit

    # @pytest.mark.skip("oneDNN not implement yet") 
    # def test_linear_binary_eq_fusion(self, dtype=torch.float):
    #     x = torch.randn([2, 4], device=cpu_device)
    #     a = torch.randn([2, 4], device=cpu_device)
    #     model = LinearBinaryEqual(4, 4)
    #     y = model(x, a)
    #     print("raw: ", y)

    #     x = x.to("xpu")
    #     a = a.to("xpu")
    #     model.to("xpu")
    #     modelJit = torch.jit.trace(model, (x, a))

    #     with torch.no_grad():
    #         if print_graph:
    #             print(modelJit.graph_for(x, a))
    #         y_dpcpp = modelJit(x, a)
    #         print("fusion:", y_dpcpp.cpu())
    #     self.assertEqual(y, y_dpcpp.to(cpu_device).bool())
    #     del modelJit


    # @pytest.mark.skip("oneDNN not implement yet") 
    # def test_linear_binary_ne_fusion(self, dtype=torch.float):
    #     x = torch.randn([2, 4], device=cpu_device)
    #     a = torch.randn([2, 4], device=cpu_device)
    #     model = LinearBinaryNotEqual(4, 4)
    #     y = model(x, a)
    #     print("raw: ", y)

    #     x = x.to("xpu")
    #     a = a.to("xpu")
    #     model.to("xpu")
    #     modelJit = torch.jit.trace(model, (x, a))

    #     with torch.no_grad():
    #         if print_graph:
    #             print(modelJit.graph_for(x, a))
    #         y_dpcpp = modelJit(x, a)
    #         print("fusion:", y_dpcpp.cpu())
    #     self.assertEqual(y, y_dpcpp.to(cpu_device).bool())
    #     del modelJit


    def test_linear_binary_max_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        a = torch.randn([2, 4], device=cpu_device)
        model = LinearBinaryMax(4, 4)
        y = model(x, a)
        print("raw: ", y)

        x = x.to("xpu")
        a = a.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, (x, a))

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x, a))
            y_dpcpp = modelJit(x, a)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit


    def test_linear_binary_min_fusion(self, dtype=torch.float):
        x = torch.randn([2, 4], device=cpu_device)
        a = torch.randn([2, 4], device=cpu_device)
        model = LinearBinaryMin(4, 4)
        y = model(x, a)
        print("raw: ", y)

        x = x.to("xpu")
        a = a.to("xpu")
        model.to("xpu")
        modelJit = torch.jit.trace(model, (x, a))

        with torch.no_grad():
            if print_graph:
                print(modelJit.graph_for(x, a))
            y_dpcpp = modelJit(x, a)
            print("fusion:", y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit

    # @pytest.mark.skip("oneDNN not implement yet") 
    # def test_linear_binary_ge_fusion(self, dtype=torch.float):
    #     x = torch.randn([2, 4], device=cpu_device)
    #     a = torch.randn([2, 4], device=cpu_device)
    #     model = LinearBinaryGE(4, 4)
    #     y = model(x, a)
    #     print("raw: ", y)

    #     x = x.to("xpu")
    #     a = a.to("xpu")
    #     model.to("xpu")
    #     modelJit = torch.jit.script(model)

    #     with torch.no_grad():
    #         if print_graph:
    #             print(modelJit.graph_for(x, a))
    #         y_dpcpp = modelJit(x, a)
    #         print("fusion:", y_dpcpp.cpu())
    #     self.assertEqual(y, y_dpcpp.to(cpu_device).bool())
    #     del modelJit

    # @pytest.mark.skip("oneDNN not implement yet") 
    # def test_linear_binary_gt_fusion(self, dtype=torch.float):
    #     x = torch.randn([2, 4], device=cpu_device)
    #     a = torch.randn([2, 4], device=cpu_device)
    #     model = LinearBinaryGT(4, 4)
    #     y = model(x, a)
    #     print("raw: ", y)

    #     x = x.to("xpu")
    #     a = a.to("xpu")
    #     model.to("xpu")
    #     modelJit = torch.jit.trace(model, (x, a))

    #     with torch.no_grad():
    #         if print_graph:
    #             print(modelJit.graph_for(x, a))
    #         y_dpcpp = modelJit(x, a)
    #         print("fusion:", y_dpcpp.cpu())
    #     self.assertEqual(y, y_dpcpp.to(cpu_device).bool())
    #     del modelJit

    # @pytest.mark.skip("oneDNN not implement yet") 
    # def test_linear_binary_le_fusion(self, dtype=torch.float):
    #     x = torch.randn([2, 4], device=cpu_device)
    #     a = torch.randn([2, 4], device=cpu_device)
    #     model = LinearBinaryLE(4, 4)
    #     y = model(x, a)
    #     print("raw: ", y)

    #     x = x.to("xpu")
    #     a = a.to("xpu")
    #     model.to("xpu")
    #     modelJit = torch.jit.trace(model, (x, a))

    #     with torch.no_grad():
    #         if print_graph:
    #             print(modelJit.graph_for(x, a))
    #         y_dpcpp = modelJit(x, a)
    #         print("fusion:", y_dpcpp.cpu())
    #     self.assertEqual(y, y_dpcpp.to(cpu_device).bool())
    #     del modelJit

    # @pytest.mark.skip("oneDNN not implement yet") 
    # def test_linear_binary_lt_fusion(self, dtype=torch.float):
    #     x = torch.randn([2, 4], device=cpu_device)
    #     a = torch.randn([2, 4], device=cpu_device)
    #     model = LinearBinaryLT(4, 4)
    #     y = model(x, a)
    #     print("raw: ", y)

    #     x = x.to("xpu")
    #     a = a.to("xpu")
    #     model.to("xpu")
    #     modelJit = torch.jit.trace(model, (x, a))

    #     with torch.no_grad():
    #         if print_graph:
    #             print(modelJit.graph_for(x, a))
    #         y_dpcpp = modelJit(x, a)
    #         print("fusion:", y_dpcpp.cpu())
    #     self.assertEqual(y, y_dpcpp.to(cpu_device).bool())
    #     del modelJit
