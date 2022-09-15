import torch
import intel_extension_for_pytorch
import copy
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_pstl_sort(self):
        es = torch.nn.EmbeddingBag(5, 2, mode='sum')
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=torch.int)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=torch.int)
        grad = torch.randn([5, 2])
        output = es(input, offsets)
        output.backward(grad)
        grad_weight_cpu = copy.deepcopy(es.weight.grad.data)

        input_xpu = input.to("xpu")
        offsets_xpu = offsets.to("xpu")
        es.to("xpu")
        grad_xpu = grad.to("xpu")
        output_xpu = es(input_xpu, offsets_xpu)
        es.zero_grad()
        output_xpu.backward(grad_xpu)
        grad_weight_xpu = copy.deepcopy(es.weight.grad.data)
        self.assertEqual(grad_weight_cpu, grad_weight_xpu.cpu())
        





