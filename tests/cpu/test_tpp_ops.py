import unittest
import torch
import random
import copy
import numpy
import intel_extension_for_pytorch as ipex


try:
    import transformers
except ImportError:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.31.0"]
    )
    import transformers
from common_utils import TestCase
import intel_extension_for_pytorch._C as torch_ipex_cpp


class Config:
    def __init__(self):
        self.attention_probs_dropout_prob = 0
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.layer_norm_eps = 1e-12
        self.max_position_embeddings = 512
        self.model_type = "bert"
        self.num_attention_heads = 16
        self.num_hidden_layers = 24
        self.pad_token_id = 0
        self.type_vocab_size = 2
        self.vocab_size = 30522
        self.is_decoder = False
        seed = 12345
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch_ipex_cpp.xsmm_manual_seed(seed)


class TPPOPsTester(TestCase):
    def setUp(self):
        self.config = Config()
        self.batch = 4
        self.max_seq_len = 384
        self.attention_mask = torch.zeros(self.batch, self.max_seq_len)
        self.attention_mask[0][128:] += -10000.0
        self.attention_mask[1][128:] += -10000.0
        self.attention_mask[2][78:] += -10000.0
        self.attention_mask[3][34:] += -10000.0
        self.attention_mask = self.attention_mask.unsqueeze(dim=1).unsqueeze(dim=1)
        return super().setUp()

    def _unblock_grad(self, b_param):
        return b_param.blocking_manager.unblock(b_param.grad.data).to(
            b_param.unblocked_dtype
        )

    def create_sinusoidal_positions(self, num_pos: int, dim: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
        ).float()
        return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

    def _test_backward(self, hf_res, tpp_res, hf_model, tpp_model, prec=0.0001):
        # UT for backward
        hf_loss = hf_res.sum()
        tpp_loss = tpp_res.sum()
        hf_loss.backward()
        tpp_loss.backward()
        for param_hf, param_tpp in zip(hf_model.parameters(), tpp_model.parameters()):
            if param_tpp.is_blocked():
                self.assertEqual(
                    param_hf.grad, self._unblock_grad(param_tpp), prec=prec
                )
            else:
                self.assertEqual(param_hf.grad, param_tpp.grad, prec=0.005)

    def test_tpp_bert_embeddings(self):
        hf_embs = transformers.models.bert.modeling_bert.BertEmbeddings(self.config)
        tpp_embs = ipex.cpu.tpp.fused_bert.BertEmbeddings(self.config)
        tpp_embs.load_state_dict(hf_embs.state_dict())
        for i, j in zip(tpp_embs.state_dict(), tpp_embs.state_dict()):
            assert i == j
        input_ids = torch.randint(100, 3000, (4, 384)).to(torch.long)
        input_ids[0][128:] = 0
        input_ids[1][128:] = 0
        input_ids[2][78:] = 0
        input_ids[3][34:] = 0

        token_type_ids = torch.ones(4, 384).to(torch.long)
        token_type_ids[0][128:] = 0
        token_type_ids[1][128:] = 0
        token_type_ids[2][78:] = 0
        token_type_ids[3][34:] = 0
        hf_res = hf_embs(input_ids, token_type_ids)

        tpp_res = tpp_embs(input_ids, token_type_ids).unblocked_tensor()
        self.assertEqual(hf_res, tpp_res)

    def test_tpp_bert_self_attention(self):
        ipex.cpu.tpp.fused_bert.unpad = False
        hf_self_att = transformers.models.bert.modeling_bert.BertSelfAttention(
            self.config
        )
        tpp_self_att = ipex.cpu.tpp.fused_bert.BertSelfAttention(self.config)
        tpp_self_att.load_state_dict(hf_self_att.state_dict())
        hidden_states = torch.randn(
            self.batch, self.max_seq_len, self.config.hidden_size
        )

        hf_res = hf_self_att(hidden_states, self.attention_mask)[0]
        (
            self.msk,
            self.tpp_att_mask,
            self.seq_offsets,
            self.seq_spr_offsets,
        ) = ipex.cpu.tpp.fused_bert.generate_mask(self.attention_mask)
        tpp_res = (
            tpp_self_att(
                hidden_states.view(
                    self.batch * self.max_seq_len, self.config.hidden_size
                ),
                self.tpp_att_mask,
                seq_offsets=self.seq_offsets,
                seq_sqr_offsets=self.seq_spr_offsets,
            )[0]
            .unblocked_tensor()
            .view(self.batch, self.max_seq_len, -1)
        )
        self.assertEqual(hf_res, tpp_res, prec=0.0002)
        self._test_backward(hf_res, tpp_res, hf_self_att, tpp_self_att, prec=0.005)

    def test_tpp_bert_output(self):
        hf_self_out = transformers.models.bert.modeling_bert.BertSelfOutput(self.config)
        tpp_self_out = ipex.cpu.tpp.fused_bert.BertSelfOutput(self.config)
        tpp_self_out.load_state_dict(hf_self_out.state_dict())
        hidden_states = torch.randn(
            self.batch, self.max_seq_len, self.config.hidden_size
        )
        input_tensor = torch.randn(
            self.batch, self.max_seq_len, self.config.hidden_size
        )
        hf_res = hf_self_out(hidden_states, input_tensor)
        tpp_res = (
            tpp_self_out(
                hidden_states.view(
                    self.batch * self.max_seq_len, self.config.hidden_size
                ),
                input_tensor.view(
                    self.batch * self.max_seq_len, self.config.hidden_size
                ),
            )
            .unblocked_tensor()
            .view(self.batch, self.max_seq_len, -1)
        )
        self.assertEqual(hf_res, tpp_res, prec=0.001)
        self._test_backward(hf_res, tpp_res, hf_self_out, tpp_self_out)

    def test_tpp_bert_intermediate(self):
        hf_intermediate = transformers.models.bert.modeling_bert.BertIntermediate(
            self.config
        )
        tpp_intermediate = ipex.cpu.tpp.fused_bert.BertIntermediate(self.config)
        tpp_intermediate.load_state_dict(hf_intermediate.state_dict())
        hidden_states = torch.randn(
            self.batch, self.max_seq_len, self.config.hidden_size
        )
        hf_res = hf_intermediate(hidden_states)
        tpp_res = (
            tpp_intermediate(
                hidden_states.view(
                    self.batch * self.max_seq_len, self.config.hidden_size
                )
            )
            .unblocked_tensor()
            .view(self.batch, self.max_seq_len, -1)
        )
        self.assertEqual(hf_res, tpp_res, prec=0.001)
        self._test_backward(
            hf_res, tpp_res, hf_intermediate, tpp_intermediate, prec=0.01
        )

    def test_tpp_gptj_attention_rope(self):
        def _get_embed_positions(embed_positions, position_ids):
            if embed_positions.device != position_ids.device:
                embed_positions = embed_positions.to(position_ids.device)
                self.embed_positions = embed_positions
            return embed_positions.repeat(position_ids.shape[0], 1, 1)

        def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(
                -2
            )  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

        def apply_rotary_pos_emb(
            tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
        ) -> torch.Tensor:
            sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
            cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
            return (tensor * cos) + (rotate_every_two(tensor) * sin)

        def hf_forward(query, key, position_ids, embed_positions, rotary_dim=None):
            embed_positions = _get_embed_positions(embed_positions, position_ids)
            repeated_position_ids = position_ids.unsqueeze(-1).repeat(
                1, 1, embed_positions.shape[-1]
            )
            sincos = torch.gather(embed_positions, 1, repeated_position_ids)
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

            if rotary_dim is not None:
                k_rot = key[:, :, :, :rotary_dim]
                k_pass = key[:, :, :, rotary_dim:]

                q_rot = query[:, :, :, :rotary_dim]
                q_pass = query[:, :, :, rotary_dim:]

                k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
                q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

                key = torch.cat([k_rot, k_pass], dim=-1)
                query = torch.cat([q_rot, q_pass], dim=-1)
            else:
                key = apply_rotary_pos_emb(key, sin, cos)
                query = apply_rotary_pos_emb(query, sin, cos)
            return query, key

        for rotary_dim in [64, None]:
            query = torch.rand(
                1, 32, 16, 256
            )  # (batch, head, seq_length, head_features)
            key = torch.rand(1, 32, 16, 256)
            query_tpp = copy.deepcopy(query)
            key_tpp = copy.deepcopy(key)
            position_ids = torch.arange(32).unsqueeze(0)

            pos_embd_dim = rotary_dim or 256
            embed_positions = self.create_sinusoidal_positions(2048, pos_embd_dim)
            query_hf, key_hf = hf_forward(
                query, key, position_ids, embed_positions, rotary_dim
            )
            torch.ops.torch_ipex.rotary_position_embedding(
                key_tpp, embed_positions, position_ids, 16, 256, 1, 64
            )
            torch.ops.torch_ipex.rotary_position_embedding(
                query_tpp, embed_positions, position_ids, 16, 256, 1, 64
            )

            self.assertEqual(query_hf, query_tpp)
            self.assertEqual(key_hf, key_tpp)

    def test_tpp_gptj_attention_rope_torchcompile(self):
        def func(input, embed_positions, position_ids):
            return torch.ops.torch_ipex.rotary_position_embedding(
                input, embed_positions, position_ids, 16, 256, 1, 64
            )

        for rotary_dim in [64, None]:
            query = torch.rand(
                1, 32, 16, 256
            )  # (batch, head, seq_length, head_features)
            key = torch.rand(1, 32, 16, 256)
            query_compile = copy.deepcopy(query)
            key_compile = copy.deepcopy(key)
            position_ids = torch.arange(32).unsqueeze(0)

            pos_embd_dim = rotary_dim or 256
            embed_positions = self.create_sinusoidal_positions(2048, pos_embd_dim)
            func(query, embed_positions, position_ids)
            func(key, embed_positions, position_ids)

            # torch compile with IPEX backend.
            torch._dynamo.reset()
            ipex._set_compiler_backend("inductor")
            func_compile = torch.compile(func, backend="ipex")

            func_compile(query_compile, embed_positions, position_ids)
            func_compile(key_compile, embed_positions, position_ids)

            self.assertEqual(query_compile, query)
            self.assertEqual(key_compile, key)

    def test_tpp_chatglm_attention_rope(self):
        def apply_rotary_pos_emb(
            x: torch.Tensor, rope_cache: torch.Tensor
        ) -> torch.Tensor:
            sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
            rot_dim = rope_cache.shape[-2] * 2
            x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
            rope_cache = rope_cache[:sq]
            xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
            rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
            x_out2 = torch.stack(
                [
                    xshaped[..., 0] * rope_cache[..., 0]
                    - xshaped[..., 1] * rope_cache[..., 1],
                    xshaped[..., 1] * rope_cache[..., 0]
                    + xshaped[..., 0] * rope_cache[..., 1],
                ],
                -1,
            )
            x_out2 = x_out2.flatten(3)
            return torch.cat((x_out2, x_pass), dim=-1)

        class RotaryEmbedding(torch.nn.Module):
            def __init__(self, dim, dtype=torch.float32):
                super().__init__()
                inv_freq = 1.0 / (
                    10000 ** (torch.arange(0, dim, 2).to(dtype=dtype) / dim)
                )
                self.register_buffer("inv_freq", inv_freq)
                self.dim = dim

            def forward_impl(
                self,
                seq_len: int,
                n_elem: int,
                dtype: torch.dtype,
                device: torch.device,
                base: int = 10000,
            ):
                theta = 1.0 / (
                    base
                    ** (
                        torch.arange(0, n_elem, 2, dtype=torch.float, device=device)
                        / n_elem
                    )
                )
                seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)
                idx_theta = torch.outer(seq_idx, theta).float()
                cache = torch.stack(
                    [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1
                )
                if dtype in (torch.float16, torch.bfloat16, torch.int8):
                    cache = (
                        cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
                    )
                return cache

            def forward(self, max_seq_len, offset=0):
                return self.forward_impl(
                    max_seq_len,
                    self.dim,
                    dtype=self.inv_freq.dtype,
                    device=self.inv_freq.device,
                )

        def hf_forward(query, key, position_ids, seq_length):
            rotary_emb = RotaryEmbedding(64)
            rotary_pos_emb = rotary_emb(seq_length)
            if position_ids is not None:
                rotary_pos_emb = rotary_pos_emb[position_ids]
            else:
                rotary_pos_emb = rotary_pos_emb[None, :seq_length]
            rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
            query = apply_rotary_pos_emb(query, rotary_pos_emb)
            key = apply_rotary_pos_emb(key, rotary_pos_emb)
            return query, key

        query = torch.rand(32, 1, 32, 128)
        key = torch.rand(32, 1, 2, 128)
        query_tpp = copy.deepcopy(query).transpose(0, 1)
        key_tpp = copy.deepcopy(key).transpose(0, 1)
        position_ids = torch.arange(32).unsqueeze(0)

        embed_positions = self.create_sinusoidal_positions(2048, 64)
        query_hf, key_hf = hf_forward(query, key, position_ids, 32)
        past_len = 0
        torch.ops.torch_ipex.rotary_position_embedding(
            key_tpp,
            embed_positions,
            torch.tensor(past_len),
            key_tpp.size(-2),
            key_tpp.size(-1),
            1,
            64,
        )
        torch.ops.torch_ipex.rotary_position_embedding(
            query_tpp,
            embed_positions,
            torch.tensor(past_len),
            query_tpp.size(-2),
            query_tpp.size(-1),
            1,
            64,
        )

        self.assertEqual(query_hf, query_tpp.transpose(0, 1))
        self.assertEqual(key_hf, key_tpp.transpose(0, 1))

    def test_tpp_chatglm_attention_rope_torchcompile(self):
        def func(input, embed_positions, position_ids):
            return torch.ops.torch_ipex.rotary_position_embedding(
                input,
                embed_positions,
                position_ids,
                input.size(-2),
                input.size(-1),
                1,
                64,
            )

        query = torch.rand(1, 32, 32, 128)
        key = torch.rand(1, 32, 2, 128)
        query_compile = copy.deepcopy(query)
        key_compile = copy.deepcopy(key)
        past_len = 0
        position_ids = torch.tensor(past_len)

        embed_positions = self.create_sinusoidal_positions(2048, 64)
        func(query, embed_positions, position_ids)
        func(key, embed_positions, position_ids)

        # torch compile with IPEX backend.
        torch._dynamo.reset()
        ipex._set_compiler_backend("inductor")
        func_compile = torch.compile(func, backend="ipex")

        func_compile(query_compile, embed_positions, position_ids)
        func_compile(key_compile, embed_positions, position_ids)

        self.assertEqual(query_compile, query)
        self.assertEqual(key_compile, key)


if __name__ == "__main__":
    test = unittest.main()
