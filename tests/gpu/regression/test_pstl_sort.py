import torch
import intel_extension_for_pytorch  # noqa
import copy
from torch.testing._internal.common_utils import TestCase
from intel_extension_for_pytorch.nn.functional._embeddingbag import torch_embedding_bag

class TestTorchMethod(TestCase):
    def test_pstl_sort(self):
        es = torch.nn.EmbeddingBag(5, 2, mode='sum')
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=torch.int)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=torch.int)
        grad = torch.randn([5, 2])

        # FIXME: here using torch_embedding_bag to call original torch embeddingbag forward
        # because merge cpu frontend code will globally replace the torch embedding bag in 
        # intel_extension_for_pytorch/nn/functional/_embeddingbag.py and it only support int64
        # input and indices. It is miss aligned with torch(torch supports int32 and int64
        # input and indices). Thus here use torch_embedding_bag.
        output_tuple = torch_embedding_bag(es.weight, input, offsets)
        output_tuple[0].backward(grad)
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
