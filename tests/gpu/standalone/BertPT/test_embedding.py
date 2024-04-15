import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
 
import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_embedding_1(self, dtype=torch.float):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(2, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_bfloat16_1(self, dtype=torch.bfloat16):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(2, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_float16_1(self, dtype=torch.float16):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(2, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_2(self, dtype=torch.float):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(512, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_bfloat16_2(self, dtype=torch.bfloat16):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(512, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_float16_2(self, dtype=torch.float16):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(512, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_3(self, dtype=torch.float):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(30522, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_bfloat16_3(self, dtype=torch.bfloat16):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(30522, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_embedding_float16_3(self, dtype=torch.float16):
        input = torch.ones(1, 512, dtype=dtype).long()
        weight = torch.randn(30522, 1024, dtype=dtype)
        input_xpu = input.xpu()
        weight_xpu = weight.xpu()
        output = F.embedding(input, weight)
        output_xpu = F.embedding(input_xpu, weight_xpu)
        self.assertEqual(output, output_xpu.cpu())
