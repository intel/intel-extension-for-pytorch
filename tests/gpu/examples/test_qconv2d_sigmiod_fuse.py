import torch
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
            torch.nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        return x

class TestTorchMethod(TestCase):
    def test_qConv2d_sigmoid(self, dtype=torch.float):
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)
        model = M()

        # cpu int8
        modelJit = torch.jit.trace(model, torch.randn([1, 64, 128, 128]))
        modelJit.eval()
        print(modelJit)
        print("finish jit...")

        input = torch.rand([1, 64, 128, 128])
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
            calib_input = input
            modelJit(calib_input)
        print("start cpu convert")
        modelJit = convert_jit(modelJit, True)
        # inference
        print("start inference ...")
        for i in range(5):
            start = time.time()
            output_cpu = modelJit(input)
            end = time.time()
            print("iter.{} ... {time:.3f}ms".format(i, time=(end - start) * 1000))


        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # xpu
        print("-------start xpu path-------")
        print("start jit ...")
        model = model.to("xpu")
        model = torch.jit.trace(model, torch.randn([1, 64, 128, 128]).to("xpu"))
        modelJit = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
        modelJit.eval()
        print("finish jit ...")

        input = torch.rand([1, 64, 128, 128], device="xpu")
        input = input.to("xpu")
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
            calib_input = input
            print(calib_input.size())
            modelJit(calib_input)
        modelJit = convert_jit(modelJit, True)
        print(modelJit.graph_for(input))

        print("start inference ...")
        for i in range(5):
            start = time.time()

            output = modelJit(input)
            torch.xpu.synchronize()

            end = time.time()
            print("iter.{} ... {time:.3f}ms".format(i, time=(end - start) * 1000))
        self.assertEqual(output.cpu(), output)
