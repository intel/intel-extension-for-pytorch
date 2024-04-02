import unittest
import torch
import random
import numpy
import intel_extension_for_pytorch as ipex


try:
    import transformers
except ImportError:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.38.1"]
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


if __name__ == "__main__":
    test = unittest.main()
