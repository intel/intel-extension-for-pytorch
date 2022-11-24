import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

class TestFill(TestCase):
    def test_logdet_backward(self):
        det = torch.rand([4,4])
        det_xpu = det.to("xpu")
        det.requires_grad_(True)
        det_xpu.requires_grad_(True)
        grad = torch.tensor(2.0)
        grad_xpu = grad.to("xpu")
        y = torch.logdet(det)
        y.backward(grad)

        y_xpu = torch.logdet(det_xpu)
        y_xpu.backward(grad_xpu)

        self.assertEqual(det_xpu.grad.to("cpu"), det.grad)