import torch
import copy
from torch.testing._internal.common_utils import TestCase

import time
import intel_extension_for_pytorch  # noqa
import numpy as np

from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)
import pytest

checking_atol = 3e-2
checking_rtol = 3e-2


class ConvSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.Sigmoid()
        )
        self.block[0].weight.data.fill_(1)
        self.block[2].weight.data.fill_(1)

    def forward(self, x):
        x = self.block(x)
        return x


class ConvLeakyRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.LeakyReLU(0.01, inplace=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class ConvMish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = Mish()
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


class ConvMishAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = Mish()
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        h = x
        h = self.conv2(h)
        h = self.activation(h)
        x = x + h
        return x

class ConvBinaryBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ConvSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.SiLU(inplace=True)
        )
    def forward(self, x):
        x = self.block(x)
        return x

def impe_fp32_model(model, device, test_input):
    modelImpe = model.to(device)
    return modelImpe(test_input.clone().to(device))


def impe_int8_model(model, device, test_input):
    modelImpe = torch.quantization.QuantWrapper(model)
    modelImpe = modelImpe.to(device)
    modelImpe.eval()

    qconfig_u8 = torch.quantization.QConfig(
        activation=torch.quantization.observer.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
            dtype=torch.quint8
        ),
        weight=torch.quantization.default_weight_observer
    )
    modelImpe.qconfig = qconfig_u8

    torch.quantization.prepare(modelImpe, inplace=True)

    with torch.no_grad():
        calib = test_input.to(device)
        modelImpe(calib)

    torch.quantization.convert(modelImpe, inplace=True)

    return modelImpe(test_input.to(device))


def trace_int8_model(model, device, test_input):
    model = model.to(device)
    modelJit = torch.jit.trace(model, test_input.to(device))
    modelJit.eval()
    modelJit.to(device)
    print(modelJit)
    print("finish jit tracing...")

    print("start ", device, " calibration ...")
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
    test_input = test_input.to(device)
    with torch.no_grad():
        for i in range(1):
            calib_input = test_input
            modelJit(calib_input)
    print("start ", device, " convert...")
    modelJit = convert_jit(modelJit, True)
    # inference
    print("start ", device, " inference ...")
    with torch.no_grad():
        for i in range(1):
            start = time.time()
            output_cpu = modelJit(test_input)
            end = time.time()
            print("iter.{} ... {time:.3f}ms".format(i, time=(end - start) * 1000))
        print("print ", device, " jit graph ....")
        print(modelJit.graph_for(test_input))

        print("get ", device, " test input result....")
        output = modelJit(test_input)
        print("finish ", device, " testing.......")
    return output


class TestTorchMethod(TestCase):
    # @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    # @pytest.mark.skip("Temporary skip due to small diff after oneDNN 3.0 upgrade")
    def test_qConv2d_sigmoid(self, dtype=torch.float):
        model = ConvSigmoid()
        model1 = copy.deepcopy(model)
        test_input = torch.ones([1, 3, 8, 8])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input)
        impe_res = impe_int8_model(model1, "xpu", test_input)
        # imperatie path has an extra quantized&dequatnize pair, which includes more error
        np.testing.assert_almost_equal(xpu_res.cpu().numpy(), impe_res.cpu().numpy(), decimal=1)

        # cpu vs. xpu
        # For model that impe path is unreachable, we can compare int8 result
        # with CPU float module result + quantization counterpart.
        model = model.to("cpu")
        cpu_res = model(test_input)
        xpu_res = trace_int8_model(model, "xpu", test_input)
        cpu_res = torch.quantize_per_tensor(cpu_res, 1.0 / 255.0 *2.0, 0, torch.qint8)
        xpu_res = torch.quantize_per_tensor(xpu_res, 1.0 / 255.0 *2.0, 0, torch.qint8)
        # fbgemm and onednn use different scale here
        np.testing.assert_almost_equal(xpu_res.dequantize().cpu().numpy(), impe_res.dequantize().cpu().numpy(), decimal=1)

    def test_qConv2d_leakyrelu(self, dtype=torch.float):
        model = ConvLeakyRelu()
        model1 = copy.deepcopy(model)
        test_input = torch.rand([1, 3, 8, 8])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input.clone())
        xpu_ref = impe_fp32_model(model, "xpu", test_input.clone())
        self.assertEqual(xpu_res.cpu(), xpu_ref.cpu(), atol=checking_atol, rtol=checking_rtol)

    def test_qConv2d_mish(self, dtype=torch.float):
        model = ConvMish()
        model1 = copy.deepcopy(model)
        test_input = torch.rand([8, 8, 32, 32])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input.clone())
        xpu_ref = impe_fp32_model(model, "xpu", test_input.clone())
        self.assertEqual(xpu_res.cpu(), xpu_ref.cpu(), atol=checking_atol, rtol=checking_rtol)

    def test_qConv2d_mish_add(self, dtype=torch.float):
        model = ConvMishAdd()
        model1 = copy.deepcopy(model)
        test_input = torch.rand([8, 8, 32, 32])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input.clone())
        xpu_ref = impe_fp32_model(model, "xpu", test_input.clone())
        self.assertEqual(xpu_res.cpu(), xpu_ref.cpu(), atol=checking_atol, rtol=checking_rtol)

    def test_conv_binary_bias(self, dtype=torch.float):
        model = ConvBinaryBias()
        test_input = torch.rand([1, 256, 1, 1])
        xpu_res = trace_int8_model(model, "xpu", test_input.clone())

    def test_conv_silu(self, dtype=torch.float):
        model = ConvSiLU()
        model1 = copy.deepcopy(model)
        test_input = torch.rand([3, 3, 640, 640])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input.clone())
        xpu_ref = impe_fp32_model(model, "xpu", test_input.clone())
        self.assertEqual(xpu_res.cpu(), xpu_ref.cpu(), atol=checking_atol, rtol=checking_rtol)
