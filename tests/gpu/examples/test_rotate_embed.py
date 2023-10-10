import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_interleave(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return rotate_every_two(tensor) * sin + tensor * cos

def apply_rotary_pos_emb_half(tensor, sin, cos):
    return tensor * cos + rotate_half(tensor) * sin

class TestNNMethod(TestCase):
    def test_rotary_embedding_interleave(self):
        test_tensor_size = [(1, 1, 1, 16),
                            (64, 32, 1, 16),
                            (64, 32, 1, 32),
                            (64, 32, 1, 130),
                            (64, 32, 20, 116),
                            (64, 32, 1, 1028),
                            (64, 32, 1, 2048),
                            (1024, 1024, 1, 16)]
        for size in test_tensor_size:
            tensor = torch.randn(size).float().to("xpu")
            tensor1 = torch.randn(size).float().to("xpu")
            sin = torch.randn(size).float().to("xpu")
            cos = torch.randn(size).float().to("xpu")

            ref = apply_rotary_pos_emb_interleave(tensor, sin, cos)
            ref1 = apply_rotary_pos_emb_interleave(tensor1, sin, cos)
            out = torch.empty_like(tensor)
            out1 = torch.empty_like(tensor1)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_two(tensor, sin, cos, out)
            self.assertEqual(out, ref)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_two_qk(tensor, tensor1, sin, cos, out, out1)
            self.assertEqual(out, ref)
            self.assertEqual(out1, ref1)

    def test_rotary_embedding_half(self):
        test_tensor_size = [(1, 1, 1, 16),
                            (64, 32, 1, 16),
                            (64, 32, 1, 32),
                            (64, 32, 1, 130),
                            (64, 32, 20, 116),
                            (64, 32, 1, 1028),
                            (64, 32, 1, 2048),
                            (1024, 1024, 1, 16)
                            ]
        for size in test_tensor_size:
            tensor = torch.randn(size).float().to("xpu")
            tensor1 = torch.randn(size).float().to("xpu")
            sin = torch.randn(size).float().to("xpu")
            cos = torch.randn(size).float().to("xpu")

            ref = apply_rotary_pos_emb_half(tensor, sin, cos)
            ref1 = apply_rotary_pos_emb_half(tensor1, sin, cos)
            out = torch.empty_like(tensor)
            out1 = torch.empty_like(tensor1)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_half(tensor, sin, cos, out)
            self.assertEqual(out, ref)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_half_qk(tensor, tensor1, sin, cos, out, out1)
            self.assertEqual(out, ref)
            self.assertEqual(out1, ref1)
