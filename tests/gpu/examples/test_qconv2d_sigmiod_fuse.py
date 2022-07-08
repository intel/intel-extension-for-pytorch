import torch
import copy
from torch.testing._internal.common_utils import TestCase

import time
import intel_extension_for_pytorch

from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)
from torch.quantization import default_qconfig

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        return x

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
    model = model.to("cpu")
    modelJit = torch.jit.trace(model, test_input.to("cpu"))
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
    def test_qConv2d_sigmoid(self, dtype=torch.float):
        model = M()
        model1 = copy.deepcopy(model)
        test_input = torch.rand([1, 3, 8, 8])
        # impe vs. jit
        # For model that JIT and impe path are both reachable.
        xpu_res = trace_int8_model(model, "xpu", test_input)
        impe_res = impe_int8_model(model1, "xpu", test_input)
        self.assertEqual(xpu_res.cpu(), impe_res.cpu())

        # cpu vs. xpu
        # For model that impe path is unreachable, we can compare int8 result
        # with CPU float module result + quantization counterpart.
        model = model.to("cpu")
        cpu_res = model(test_input)
        xpu_res = trace_int8_model(model, "xpu", test_input)
        cpu_res = torch.quantize_per_tensor(cpu_res, 1.0 / 255.0, 0, torch.quint8)
        xpu_res = torch.quantize_per_tensor(xpu_res, 1.0 / 255.0, 0, torch.quint8)
        self.assertEqual(cpu_res.int_repr().numpy(), xpu_res.to("cpu").int_repr().numpy())
