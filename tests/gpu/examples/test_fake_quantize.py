import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_fake_quantize_per_channel_affine(self, dtype=torch.float):
        src_cpu = torch.randn([1, 3, 2, 2], requires_grad=True, dtype=torch.float)
        src_xpu = src_cpu.clone().detach().to("xpu")
        src_xpu.requires_grad = True

        data_type = torch.quint8
        channel_scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point_cpu = torch.tensor([0, 0, 0], dtype=torch.int)
        channel_scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        channel_zero_point_xpu = torch.tensor([0, 0, 0], dtype=torch.int).to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max

        dst_cpu = torch.fake_quantize_per_channel_affine(src_cpu, channel_scale_cpu,
                                                         channel_zero_point_cpu, 1, quant_min, quant_max)
        dst_xpu = torch.fake_quantize_per_channel_affine(src_xpu, channel_scale_xpu,
                                                         channel_zero_point_xpu, 1, quant_min, quant_max)

        self.assertEqual(dst_cpu, dst_xpu.cpu())

        dst_cpu = torch.sum(dst_cpu)
        dst_xpu = torch.sum(dst_xpu)
        dst_cpu.backward()
        dst_xpu.backward()

        self.assertEqual(src_cpu.grad, src_xpu.grad.cpu())

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_fake_quantize_learnable_per_channel_affine(self, dtype=torch.float):
        src_cpu = torch.randn([1, 3, 2, 2])
        src_xpu = src_cpu.clone().to("xpu")

        data_type = torch.quint8
        channel_scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point_cpu = torch.tensor([0, 0, 0], dtype=torch.float)
        channel_scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        channel_zero_point_xpu = torch.tensor([0, 0, 0], dtype=torch.float).to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max
        grad_factor = 1.5
        dst_cpu = torch._fake_quantize_learnable_per_channel_affine(src_cpu, channel_scale_cpu,
                                                                    channel_zero_point_cpu, 1, quant_min, quant_max, grad_factor)
        dst_xpu = torch._fake_quantize_learnable_per_channel_affine(src_xpu, channel_scale_xpu,
                                                                    channel_zero_point_xpu, 1, quant_min, quant_max, grad_factor)
        self.assertEqual(dst_cpu, dst_xpu.cpu())

    def test_fake_quantize_learnable_per_channel_affine_backward(self, dtype=torch.float):
        x_cpu = torch.randn([1, 3, 2, 2], requires_grad=True)
        x_xpu = x_cpu.clone().detach().to("xpu")
        x_xpu.requires_grad = True

        data_type = torch.quint8
        channel_scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point_cpu = torch.tensor([0, 0, 0], dtype=torch.float)
        channel_scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        channel_zero_point_xpu = torch.tensor([0, 0, 0], dtype=torch.float).to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max
        grad_factor = 1.5
        y_cpu = torch._fake_quantize_learnable_per_channel_affine(x_cpu, channel_scale_cpu,
                                                                  channel_zero_point_cpu, 1, quant_min, quant_max, grad_factor)
        y_xpu = torch._fake_quantize_learnable_per_channel_affine(x_xpu, channel_scale_xpu,
                                                                  channel_zero_point_xpu, 1, quant_min, quant_max, grad_factor)

        linear = torch.nn.Linear(2, 10)
        activation = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=0)

        y_cpu = linear(y_cpu)
        y_cpu = activation(y_cpu)
        y_cpu = softmax(y_cpu)
        y_cpu = torch.sum(y_cpu)
        y_cpu.backward()

        y_xpu = linear.to("xpu")(y_xpu)
        y_xpu = activation.to("xpu")(y_xpu)
        y_xpu = softmax.to("xpu")(y_xpu)
        y_xpu = torch.sum(y_xpu)
        y_xpu.backward()

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())

    def test_fake_quantize_per_tensor_affine(self, dtype=torch.float):
        src_cpu = torch.randn([1, 3, 2, 2], requires_grad=True)
        src_xpu = src_cpu.clone().detach().to("xpu")
        src_xpu.requires_grad = True

        data_type = torch.quint8
        scale_cpu = torch.tensor(1.5)
        zero_point_cpu = torch.tensor(1, dtype=torch.int)
        scale_xpu = scale_cpu.clone().to("xpu")
        zero_point_xpu = zero_point_cpu.clone().to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max

        dst_cpu = torch.fake_quantize_per_tensor_affine(src_cpu, scale_cpu,
                                                        zero_point_cpu, quant_min, quant_max)
        dst_xpu = torch.fake_quantize_per_tensor_affine(src_xpu, scale_xpu,
                                                        zero_point_xpu, quant_min, quant_max)

        self.assertEqual(dst_cpu, dst_xpu.cpu())

        dst_cpu = torch.sum(dst_cpu)
        dst_xpu = torch.sum(dst_xpu)
        dst_cpu.backward()
        dst_xpu.backward()

        self.assertEqual(src_cpu.grad, src_xpu.grad.cpu())

    def test_fake_quantize_learnable_per_tensor_affine(self, dtype=torch.float):
        src_cpu = torch.randn([1, 3, 2, 2])
        src_xpu = src_cpu.clone().to("xpu")

        data_type = torch.quint8
        scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        zero_point_cpu = torch.tensor([0, 0, 0], dtype=torch.float)
        scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        zero_point_xpu = torch.tensor([0, 0, 0], dtype=torch.float).to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max
        grad_factor = 1.5
        dst_cpu = torch._fake_quantize_learnable_per_tensor_affine(src_cpu, scale_cpu,
                                                                   zero_point_cpu, quant_min, quant_max, grad_factor)
        dst_xpu = torch._fake_quantize_learnable_per_tensor_affine(src_xpu, scale_xpu,
                                                                   zero_point_xpu, quant_min, quant_max, grad_factor)
        self.assertEqual(dst_cpu, dst_xpu.cpu())

    def test_fake_quantize_learnable_per_tensor_affine_backward(self, dtype=torch.float):
        x_cpu = torch.randn([1, 3, 2, 2], requires_grad=True)
        x_xpu = x_cpu.clone().detach().to("xpu")
        x_xpu.requires_grad = True

        data_type = torch.quint8
        scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        zero_point_cpu = torch.tensor([0, 0, 0], dtype=torch.float)
        scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        zero_point_xpu = torch.tensor([0, 0, 0], dtype=torch.float).to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max
        grad_factor = 1.5
        y_cpu = torch._fake_quantize_learnable_per_tensor_affine(x_cpu, scale_cpu,
                                                                 zero_point_cpu, quant_min, quant_max, grad_factor)
        y_xpu = torch._fake_quantize_learnable_per_tensor_affine(x_xpu, scale_xpu,
                                                                 zero_point_xpu, quant_min, quant_max, grad_factor)

        linear = torch.nn.Linear(2, 10)
        activation = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=0)

        y_cpu = linear(y_cpu)
        y_cpu = activation(y_cpu)
        y_cpu = softmax(y_cpu)
        y_cpu = torch.sum(y_cpu)
        y_cpu.backward()

        y_xpu = linear.to("xpu")(y_xpu)
        y_xpu = activation.to("xpu")(y_xpu)
        y_xpu = softmax.to("xpu")(y_xpu)
        y_xpu = torch.sum(y_xpu)
        y_xpu.backward()

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())

    def test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(self, dtype=torch.float):
        src_cpu = torch.randn([1, 3, 2, 2], requires_grad=True, dtype=torch.float)
        src_xpu = src_cpu.clone().detach().to("xpu")
        src_xpu.requires_grad = True

        data_type = torch.quint8
        scale_cpu = torch.tensor(1.5)
        zero_point_cpu = torch.tensor(1, dtype=torch.int)
        scale_xpu = scale_cpu.clone().to("xpu")
        zero_point_xpu = zero_point_cpu.clone().to("xpu")
        fake_quant_cpu_enabled = torch.tensor(1)
        fake_quant_xpu_enabled = fake_quant_cpu_enabled.clone().to("xpu")
        quant_min = torch.iinfo(data_type).min
        quant_max = torch.iinfo(data_type).max

        dst_cpu = torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            src_cpu, scale_cpu, zero_point_cpu, fake_quant_cpu_enabled, quant_min, quant_max)
        dst_xpu = torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            src_xpu, scale_xpu, zero_point_xpu, fake_quant_xpu_enabled, quant_min, quant_max)

        self.assertEqual(dst_cpu[0], dst_xpu[0].cpu())
