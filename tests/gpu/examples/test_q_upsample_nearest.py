import torch
from torch.testing._internal.common_utils import TestCase
import torch.nn.functional as F
import intel_extension_for_pytorch # noqa

from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)

class ConvUpsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = torch.nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        y = F.interpolate(x, size=(x.size()[2], x.size()[3]), scale_factor=None, mode='nearest')
        return torch.cat((y, x), dim=1)

def trace_int8_model(model, device, test_input):
    model = model.to(device)
    modelJit = torch.jit.trace(model, test_input.to(device), check_trace=False)
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
            output_cpu = modelJit(test_input)
        output = modelJit(test_input)
    return output

class TestNNMethod(TestCase):
    def test_q_upsamle_nearest(self, dtype=torch.float):
        x_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=torch.device("cpu"))
        x_gpu = x_cpu.to("xpu")
        scales = [6, 8]
        rsf = False

        dtype_inputs = torch.qint8
        q_scale = 0.04
        q_cpu = torch.quantize_per_tensor(x_cpu, q_scale, 0, dtype_inputs)
        q_gpu = torch.quantize_per_tensor(x_gpu, q_scale, 0, dtype_inputs)

        output_cpu = torch.nn.functional.interpolate(
            q_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_gpu = torch.nn.functional.interpolate(
            q_gpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)

        self.assertEqual(output_cpu, output_gpu)

    def test_q_upsample_nearest2(self, dtype=torch.float):
        M = ConvUpsample()
        x_input = torch.randn([8, 8, 1, 1]).to("xpu")

        y_int8 = trace_int8_model(M, "xpu", x_input)
        y_fp32 = M(x_input)

        self.assertEqual(y_int8, y_fp32, atol=3e-2, rtol=3e-2)
