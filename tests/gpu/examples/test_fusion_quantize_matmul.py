import torch
import torch.nn as nn
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
import platform

torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_profiling_executor(True)

checking_atol = 3e-2
checking_rtol = 3e-2


class LinearActivation(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(50, 60),
            nn.Linear(60, 50),
        )
        self.act = act

    def forward(self, x):
        x = self.block(x)
        x = self.act(x)
        return x


class LinearSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )
        self.block[0].weight.data.fill_(1)
        self.block[2].weight.data.fill_(1)

    def forward(self, x):
        x = self.block(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


class LinearHardswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.Hardswish(),
            nn.Linear(128, 128),
            nn.Hardswish()
        )

    def forward(self, x):
        return self.block(x)


class LinearBinaryBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(
                512, 128
            ),
            nn.ReLU(),
            nn.Linear(
                128, 128
            ),
            nn.ReLU(),
            nn.Linear2d(
                128, 128
            ),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LinearBinaryAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(5, 5)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # y = self.block(x) + x
        y = self.block(x) + x
        return y


def impe_fp32_model(model, device, test_input):
    modelImpe = model.to(device)
    return modelImpe(test_input.clone().to(device))


def impe_int8_model(model, device, test_input, dtype=torch.quint8, symm=True):
    modelImpe = torch.quantization.QuantWrapper(model)
    modelImpe = modelImpe.to(device)
    modelImpe.eval()

    qscheme = torch.per_tensor_symmetric if symm else torch.per_tensor_affine
    qconfig_u8 = torch.quantization.QConfig(
        activation=torch.quantization.observer.MinMaxObserver.with_args(
            qscheme=qscheme, reduce_range=False, dtype=dtype
        ),
        weight=torch.quantization.default_weight_observer,
    )
    modelImpe.qconfig = qconfig_u8

    torch.quantization.prepare(modelImpe, inplace=True)

    with torch.no_grad():
        calib = test_input.to(device)
        modelImpe(calib)

    torch.quantization.convert(modelImpe, inplace=True)

    return modelImpe(test_input.to(device))


def trace_int8_model(model, device, test_input, dtype=torch.quint8, symm=True):
    model = model.to(device)
    modelJit = torch.jit.trace(model, test_input.to(device))
    modelJit.eval()
    modelJit.to(device)
    print(modelJit)
    print("finish jit tracing...")

    print("start ", device, " calibration ...")
    qscheme = torch.per_tensor_symmetric if symm else torch.per_tensor_affine
    qconfig_u8 = torch.quantization.QConfig(
        activation=torch.quantization.observer.MinMaxObserver.with_args(
            qscheme=qscheme, reduce_range=False, dtype=dtype
        ),
        weight=torch.quantization.default_weight_observer,
    )

    modelJit = prepare_jit(modelJit, {"": qconfig_u8}, True)

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
        print(modelJit.graph_for(test_input))
        for i in range(3):
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
    def qlinear_act(self, act):
        model = LinearActivation(act)
        test_input = torch.randn([8, 50, 50])
        cpu_ref = trace_int8_model(model, "cpu", test_input.clone())
        xpu_res = trace_int8_model(model, "xpu", test_input.clone())
        self.assertEqual(xpu_res.cpu(), cpu_ref, atol=checking_atol, rtol=checking_rtol)

    @pytest.mark.skipif(platform.system() == 'Windows', 
                        reason="Asymm quantization has undefined behaviour(hang, CL) on Windows current")
    def test_qlinear_sigmoid(self, dtype=torch.float):
        model = LinearSigmoid()
        model1 = copy.deepcopy(model)
        test_input = torch.ones([16, 16, 512])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input)
        impe_res = impe_int8_model(model1, "xpu", test_input)
        # imperatie path has an extra quantized&dequatnize pair, which includes more error
        np.testing.assert_almost_equal(
            xpu_res.cpu().numpy(), impe_res.cpu().numpy(), decimal=1
        )

        # cpu vs. xpu
        # For model that impe path is unreachable, we can compare int8 result
        # with CPU float module result + quantization counterpart.
        model = model.to("cpu")
        cpu_res = model(test_input)
        xpu_res = trace_int8_model(model, "xpu", test_input)
        cpu_res = torch.quantize_per_tensor(cpu_res, 1.0 / 256.0, -128, torch.qint8)
        xpu_res = torch.quantize_per_tensor(xpu_res, 1.0 / 256.0, -128, torch.qint8)
        # fbgemm and onednn use different scale here
        np.testing.assert_almost_equal(
            xpu_res.dequantize().cpu().numpy(),
            impe_res.dequantize().cpu().numpy(),
            decimal=1,
        )

    @pytest.mark.skipif(platform.system() == 'Windows', 
                        reason="Asymm quantization has undefined behaviour(hang, CL) on Windows current")
    def test_qlinear_abs(self):
        self.qlinear_act(torch.abs)

    @pytest.mark.skipif(platform.system() == 'Windows', 
                        reason="Asymm quantization has undefined behaviour(hang, CL) on Windows current")
    def test_qlinear_relu(self):
        act = torch.nn.ReLU()
        self.qlinear_act(act)

    def test_qlinear_silu(self):
        act = torch.nn.SiLU()
        self.qlinear_act(act)
