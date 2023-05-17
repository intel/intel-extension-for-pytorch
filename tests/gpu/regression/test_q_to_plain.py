import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa F401

from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)
import time


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
            qscheme=torch.per_tensor_symmetric, reduce_range=False, dtype=torch.quint8
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


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, 1, 1)
        self.instance_norm = nn.InstanceNorm2d(
            6, **{"eps": 1e-5, "affine": True, "momentum": 0.1}
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        return x


class TestQTensortoPlain(TestCase):
    def test_q_to_plain(self):
        mod = SimpleModule()
        test_input = torch.randn(3, 3, 16, 16)
        with torch.no_grad():
            with torch.xpu.onednn_layout():
                trace_int8_model(mod, "xpu", test_input)
