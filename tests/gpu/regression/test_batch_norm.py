import torch

from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):

    def test_batch_norm_with_none_running_stats(self, dtype=torch.float):
        x = torch.randn(4, 2, 3, 3, dtype=torch.float)
        grad_x = torch.randn(4, 2, 3, 3, dtype=torch.float)
        x = Variable(x, requires_grad=True)
        bn = torch.nn.BatchNorm2d(2, track_running_stats=False)

        ref_cf = bn(x)
        ref_cf.backward(grad_x)

        x_xpu = x.to("xpu")
        grad_xpu = grad_x.to("xpu")
        x_xpu = Variable(x_xpu, requires_grad=True)
        bn.to("xpu")
        ref_cf_xpu = bn(x_xpu)
        ref_cf_xpu.backward(grad_xpu)


        self.assertEqual(ref_cf, ref_cf_xpu.to("cpu"))
        self.assertEqual(x.grad, x_xpu.grad.to("cpu"))

    def test_instance_norm_with_none_weight_inputs(self):
        i = torch.randn(2, 64, 64, 64)
        i = Variable(i, requires_grad=True)
        grad_i = torch.randn(2, 64, 64, 64)

        weight = None
        bias = None
        running_mean = None
        running_var = None
        use_input_stats = True
        momentum = 0.1
        eps = 1e-5


        # instance norm is implemented by batch norm
        y_cpu = torch.instance_norm(i, weight, bias, running_mean, running_var, use_input_stats,
                                    momentum, eps, torch.backends.cudnn.enabled)
        y_cpu.backward(grad_i)

        i_xpu = i.to("xpu")
        i_xpu = Variable(i_xpu, requires_grad=True)
        grad_xpu = grad_i.to("xpu")
        y_xpu = torch.instance_norm(i_xpu, weight, bias, running_mean, running_var, use_input_stats,
                                    momentum, eps, torch.backends.cudnn.enabled)
        y_xpu.backward(grad_xpu)
        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(i.grad, i_xpu.grad.cpu())
