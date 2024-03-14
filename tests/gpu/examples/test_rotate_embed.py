import torch
from torch.testing._internal.common_utils import TestCase
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.transformer_modules.RoPE import (
    LlamaRotaryEmbedding,
    GPTJRotaryEmbedding,
)
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules._transformer_configuration import (
    IPEXTransformerConfig,
)
import intel_extension_for_pytorch as ipex  # noqa

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


def apply_rotary_pos_emb_interleave(
    tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    return rotate_every_two(tensor) * sin + tensor * cos


def apply_rotary_pos_emb_half(tensor, sin, cos):
    return tensor * cos + rotate_half(tensor) * sin


class TestNNMethod(TestCase):
    def test_rotary_embedding_interleave(self):
        test_tensor_size = [
            (1, 1, 1, 16),
            (64, 32, 1, 16),
            (64, 32, 1, 32),
            (64, 32, 1, 130),
            (64, 32, 20, 116),
            (64, 32, 1, 1028),
            (64, 32, 1, 2048),
            (1024, 1024, 1, 16),
        ]
        for size in test_tensor_size:
            tensor = torch.randn(size).float().to("xpu")
            tensor1 = torch.randn(size).float().to("xpu")
            sin = torch.randn(size).float().to("xpu")
            cos = torch.randn(size).float().to("xpu")

            ref = apply_rotary_pos_emb_interleave(tensor, sin, cos)
            ref1 = apply_rotary_pos_emb_interleave(tensor1, sin, cos)
            out = torch.empty_like(tensor)
            out1 = torch.empty_like(tensor1)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_two(
                tensor, sin, cos, out
            )
            self.assertEqual(out, ref)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
                tensor, tensor1, sin, cos, out, out1
            )
            ipex.llm.modules.RotaryEmbedding.apply(
                tensor, tensor1, sin, cos, tensor.size(-1), False
            )
            self.assertEqual(out, ref)
            self.assertEqual(out1, ref1)
            self.assertEqual(tensor, ref)
            self.assertEqual(tensor1, ref1)

    def test_rotary_embedding_half(self):
        test_tensor_size = [
            (1, 1, 1, 16),
            (64, 32, 1, 16),
            (64, 32, 1, 32),
            (64, 32, 1, 130),
            (64, 32, 20, 116),
            (64, 32, 1, 1028),
            (64, 32, 1, 2048),
            (1024, 1024, 1, 16),
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
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_half(
                tensor, sin, cos, out
            )
            self.assertEqual(out, ref)
            kernel_out = torch.ops.torch_ipex.apply_rotary_embedding_half_qk(
                tensor, tensor1, sin, cos, out, out1
            )
            ipex.llm.modules.RotaryEmbedding.apply(
                tensor, tensor1, sin, cos, tensor.size(-1), True
            )
            self.assertEqual(out, ref)
            self.assertEqual(out1, ref1)
            self.assertEqual(tensor, ref)
            self.assertEqual(tensor1, ref1)

    def test_rope_module_rotate_half(self):
        bs = 10
        seqlen = 128
        num_head = 32
        num_kv_head = 8
        head_dim = 128
        query_size = [bs, seqlen, num_head * head_dim]
        key_size = [bs, seqlen, num_kv_head * head_dim]
        q = torch.randn(query_size, dtype=torch.half, device="xpu")
        k = torch.randn(key_size, dtype=torch.half, device="xpu")
        q_ref = q.clone().view(bs, seqlen, num_head, head_dim)
        k_ref = k.clone().view(bs, seqlen, num_kv_head, head_dim)
        pos_ids = (
            torch.arange(seqlen, dtype=torch.long, device="xpu")
            .view(1, seqlen)
            .repeat(bs, 1)
        )
        transformer_config = IPEXTransformerConfig()
        transformer_config.embedding_dim = num_head * head_dim
        transformer_config.num_attention_head = num_head

        # test rotate half
        half_embed = LlamaRotaryEmbedding(transformer_config, torch.half)
        xpu_rope_module = ipex.llm.modules.RotaryEmbedding(2048, head_dim)
        half_embed(q_ref, k_ref, pos_ids, 0, 4, seqlen)
        q = xpu_rope_module(
            q, pos_ids, num_head, head_dim, head_dim // 2, head_dim, seqlen
        )
        k = xpu_rope_module(
            k, pos_ids, num_head, head_dim, head_dim // 2, head_dim, seqlen
        )
        self.assertEqual(q, q_ref)
        self.assertEqual(k, k_ref)

    def test_rope_module_rotate_interleave(self):
        bs = 10
        seqlen = 128
        num_head = 32
        head_dim = 128
        query_size = [bs, seqlen, num_head * head_dim]
        key_size = [bs, seqlen, num_head * head_dim]
        q = torch.randn(query_size, dtype=torch.half, device="xpu")
        k = torch.randn(key_size, dtype=torch.half, device="xpu")
        q_ref = q.clone().view(bs, seqlen, num_head, head_dim)
        k_ref = k.clone().view(bs, seqlen, num_head, head_dim)
        pos_ids = (
            torch.arange(seqlen, dtype=torch.long, device="xpu")
            .view(1, seqlen)
            .repeat(bs, 1)
        )
        transformer_config = IPEXTransformerConfig()
        transformer_config.embedding_dim = num_head * head_dim
        transformer_config.num_attention_head = num_head
        transformer_config.rotary_dim = head_dim

        # test rotate half
        interleave_embed = GPTJRotaryEmbedding(transformer_config, torch.half)
        xpu_rope_module = ipex.llm.modules.RotaryEmbedding(2048, head_dim)
        interleave_embed(q_ref, k_ref, pos_ids, 0, 4, seqlen)
        q = xpu_rope_module(q, pos_ids, num_head, head_dim, 1, head_dim, seqlen)
        k = xpu_rope_module(k, pos_ids, num_head, head_dim, 1, head_dim, seqlen)
        self.assertEqual(q, q_ref)
        self.assertEqual(k, k_ref)

    def test_rope_module_pack_qk_rotate_half(self):
        bs = 10
        seqlen = 128
        num_head = 32
        num_kv_head = 8
        head_dim = 128
        query_size = [bs, seqlen, num_head * head_dim]
        key_size = [bs, seqlen, num_kv_head * head_dim]
        q = torch.randn(query_size, dtype=torch.half, device="xpu")
        k = torch.randn(key_size, dtype=torch.half, device="xpu")
        q_ref = q.clone().view(bs, seqlen, num_head, head_dim)
        k_ref = k.clone().view(bs, seqlen, num_kv_head, head_dim)
        pos_ids = (
            torch.arange(seqlen, dtype=torch.long, device="xpu")
            .view(1, seqlen)
            .repeat(bs, 1)
        )
        transformer_config = IPEXTransformerConfig()
        transformer_config.embedding_dim = num_head * head_dim
        transformer_config.num_attention_head = num_head

        # test rotate half
        half_embed = LlamaRotaryEmbedding(transformer_config, torch.half)
        xpu_rope_module = ipex.llm.modules.RotaryEmbedding(2048, head_dim)
        half_embed(q_ref, k_ref, pos_ids, 0, 4, seqlen)
        qk = torch.cat([q, k], dim=-1)
        q, k = xpu_rope_module(
            qk, pos_ids, num_head, head_dim, head_dim // 2, head_dim, seqlen, 2
        )
        self.assertEqual(q, q_ref)
        self.assertEqual(k, k_ref)
