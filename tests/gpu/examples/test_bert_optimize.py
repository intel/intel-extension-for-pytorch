import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch as ipex


class TestBertOptimize(TestCase):
    def test_bert_self_attention_replace(self, dtype=torch.half):
        from transformers import BertConfig, AutoModelForPreTraining

        config = BertConfig(
            attention_probs_dropout_prob=0,
            hidden_act="gelu",
            hidden_dropout_prob=0,
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=4096,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            num_attention_heads=16,
            num_hidden_layers=24,
            pad_token_id=0,
            position_embedding_type="absolute",
            type_vocab_size=2,
            use_cache=True,
            vocab_size=30522,
        )
        hidden_states = torch.randn([16, 512, 1024], dtype=dtype)
        attention_mask = torch.zeros([16, 1, 1, 512], dtype=dtype)

        raw_model = AutoModelForPreTraining.from_config(
            config,
            attn_implementation="eager",
        ).to(dtype)
        raw_model.train()
        expect_output = raw_model.bert.encoder.layer[0].attention.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        optimized_model, _ = ipex.optimize_transformers(
            raw_model.to("xpu"), dtype=dtype, device="xpu"
        )
        actual_output = optimized_model.bert.encoder.layer[0].attention.self(
            hidden_states=hidden_states.to("xpu"),
            attention_mask=attention_mask.to("xpu"),
        )
        self.assertEqual(
            expect_output,
            actual_output,
            atol=1e-2,
            rtol=1e-4,
        )
