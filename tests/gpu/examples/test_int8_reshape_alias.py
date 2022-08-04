import torch
from torch.testing._internal.common_utils import TestCase

import time
import intel_extension_for_pytorch # noqa

from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Conv2d(3, 48, kernel_size=3, padding=1, dilation=1)
        self.fc = torch.nn.Linear(48 * 224 * 224, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class TestInt8ReshapeAlias(TestCase):
    def test_int8_reshape_alias(self):
        model = Net()

        # jit model
        print("start jit ...")
        modelJit = torch.jit.script(model)
        modelJit = wrap_cpp_module(torch._C._jit_pass_fold_convbn(modelJit._c))
        modelJit.eval()
        print("finish jit ...")

        modelJit = modelJit.to("xpu")
        input = torch.randn([1, 3, 224, 224], device="xpu")
        print("start calibration ...")

        # calibration
        # default with per_tensor quantization
        with torch.no_grad():
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                    dtype=torch.qint8
                ),
                weight=torch.quantization.default_weight_observer
            )

            modelJit = prepare_jit(modelJit, {'': qconfig}, True)
            modelJit = modelJit.to("xpu")

            # do calibration
            for i in range(1):
                calib_input = input
                modelJit(calib_input)

            modelJit = convert_jit(modelJit, True)

        # inference
        print("start inference ...")
        with torch.no_grad():
            for i in range(5):
                start = time.time()

                modelJit(input)
                torch.xpu.synchronize()

                end = time.time()
                print("iter.{} ... {time:.3f}ms".format(i, time=(end - start) * 1000))
