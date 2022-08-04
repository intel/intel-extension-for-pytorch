import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa


class TestNNMethod(TestCase):
    def test_instance_norm(self):
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 0
        dims = (1, 4, 8, 1, 1)

        float_cls = torch.nn.InstanceNorm3d

        q_cls = torch.nn.quantized.InstanceNorm3d

        X = (torch.rand(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.qint8)
        dqX = qX.dequantize()

        float_mod = float_cls(dims[1]).float()
        float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
        float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(dqY_ref, y_scale, y_zero_point, dtype=torch.qint8)  # for assert equal
        print("--------float ends--------")

        quant_mod = q_cls(dims[1], float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        qY = quant_mod(qX)
        print("--------quantized cpu ends--------")

        quant_mod_xpu = quant_mod.to("xpu")
        qX_gpu = X.to("xpu")
        qX_xpu = torch.quantize_per_tensor(qX_gpu, x_scale, x_zero_point, dtype=torch.qint8)
        qY_xpu = quant_mod_xpu(qX_xpu)

        qY_dequantized = torch.dequantize(qY)
        qY_xpu_dequantized = torch.dequantize(qY_xpu).cpu().contiguous()
        self.assertEqual(qY_dequantized, qY_xpu_dequantized)


if __name__ == "__main__":
    test_case = TestNNMethod()
    test_case.test_instance_norm()
