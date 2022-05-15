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

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
device = "xpu"

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class TestTorchMethod(TestCase):
    def test_qConv2d_LeakyRelu(self, dtype=torch.float):
        model = M()
        print("start jit ...")
        model = torch.jit.script(model)
        modelJit = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
        modelJit.eval()
        print("finish jit ...")

        input = torch.rand([64, 3, 416, 416])
        input = input.to(device)
        modelJit = modelJit.to(device)

        print("start calibration ...")
        # calibration
        # default with per_tensor quantization
        with torch.inference_mode():
            qconfig_s8 = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                    dtype=torch.qint8
                ),
                weight=torch.quantization.default_weight_observer
            )

            modelJit = prepare_jit(modelJit, {'': qconfig_s8}, True)
            modelJit = modelJit.to(device)

            # do calibration
            for i in range(1):
                calib_input = input
                modelJit(calib_input)
            modelJit = convert_jit(modelJit, True)
            print(modelJit.graph_for(input))
        # inference
        print("start inference ...")
        with torch.inference_mode():
            for i in range(5):
                start = time.time()

                output = modelJit(input)
                torch.xpu.synchronize()

                end = time.time()
                print("iter.{} ... {time:.3f}ms".format(i, time=(end - start) * 1000))
