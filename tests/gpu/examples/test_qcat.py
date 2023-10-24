import torch
import torch.nn as nn
import copy

from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
from test_fusion_quantize import trace_int8_model
import numpy as np
import platform
import pytest


class Conv_Cat(nn.Module):
    def __init__(self, with_relu):
        super(Conv_Cat, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)
        self.with_relu = with_relu
        if with_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        if self.with_relu:
            x1 = self.relu(x1)
            x2 = self.relu(x2)
            x3 = self.relu(x3)
        out = torch.cat([x1, x2, x3], dim=1)
        return out


class TestTorchMethod(TestCase):
    def test_cat_array_quint8(self, dtype=torch.float):
        zp_vec = [0] if platform.system() == "Windows" else [0, 2]
        for dtype in [torch.quint8, torch.qint8]:
            for zp in zp_vec:
                zp_out = 0 if platform.system() == "Windows" else 4
                input1 = torch.randn(1, 1, 5, 5)
                input2 = torch.randn(1, 1, 5, 5)
                input3 = torch.randn(1, 1, 5, 5)

                input1_gpu = input1.to("xpu")
                input2_gpu = input2.to("xpu")
                input3_gpu = input3.to("xpu")

                q_input1 = torch.quantize_per_tensor(input1, 0.4, zp, dtype)
                q_input2 = torch.quantize_per_tensor(input2, 0.5, zp, dtype)
                q_input3 = torch.quantize_per_tensor(input3, 0.6, zp, dtype)

                output_int8 = torch.ops.quantized.cat(
                    [q_input1, q_input2, q_input3], dim=1, scale=0.02, zero_point=zp_out
                )

                q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.4, zp, dtype)
                q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.5, zp, dtype)
                q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.6, zp, dtype)

                output_gpu_int8 = torch.ops.quantized.cat(
                    [q_input1_gpu, q_input2_gpu, q_input3_gpu],
                    dim=1,
                    scale=0.02,
                    zero_point=zp_out,
                )

                print("output_input\n", output_int8.dequantize()[0][1])
                print("output_input\n", output_gpu_int8.dequantize()[0][1])

                self.assertEqual(output_int8, output_gpu_int8)

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Asymm quantization has undefined behaviour(hang, CL) on Windows current",
    )
    def test_conv_cat(self):
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)
        for with_relu in [True, False]:
            with torch.no_grad():
                dtype = torch.quint8
                model = Conv_Cat(with_relu)
                model1 = copy.deepcopy(model)
                test_input = torch.randn(1, 3, 15, 15)
                # cpu_res = trace_int8_model(model1, "cpu", test_input)
                xpu_res = trace_int8_model(model, "xpu", test_input)
                ref_res = model1(test_input)
                np.testing.assert_almost_equal(
                    xpu_res.cpu().numpy(), ref_res.cpu().numpy(), decimal=1
                )
