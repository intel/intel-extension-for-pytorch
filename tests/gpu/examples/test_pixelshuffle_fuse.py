import torch
import torch.nn as nn
import torch_ipex
from torch.nn.modules.utils import _pair
import pytest

from torch.testing._internal.common_utils import TestCase
from torch.jit._recursive import wrap_cpp_module

from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
    script_qconfig,
)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

class PixelShuffle(torch.nn.Module):
  def __init__(self):
    super(PixelShuffle, self).__init__()
    self.conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
    self.pixel = nn.PixelShuffle(2)

  def forward(self, x):
    res = self.conv(x)
    res = self.pixel(res)
    return res

class  TestTorchMethod(TestCase):
    def test_dequant_pixelshuffle(self, dtype=torch.float):
        src_cpu = torch.randn(1, 64, 64, 64)
        src_xpu = src_cpu.to("xpu")

        data_type = torch.qint8
        tensor_scale = 0.3
        tensor_zero_point = 0

        model = PixelShuffle()

        dst_cpu = model(src_cpu)
        print("dst cpu ", dst_cpu)

        jit_model = torch.jit.script(model)
        qconfig_s8 = torch.quantization.QConfig(
            activation=torch.quantization.observer.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
                dtype=torch.qint8
            ),
            weight=torch.quantization.default_weight_observer
        )
        jit_model = prepare_jit(jit_model,
                {
                '':qconfig_s8,
                },
                True)

        jit_model = jit_model.to("xpu")
        modelJit = convert_jit(jit_model, True)

        y_dpcpp = modelJit(src_xpu)
        with torch.no_grad():
            print(modelJit.graph_for(src_xpu))
