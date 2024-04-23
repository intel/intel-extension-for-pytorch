import unittest
import torch
import intel_extension_for_pytorch as ipex
import sys
import subprocess
import os
import copy
from intel_extension_for_pytorch.transformers import (
    shard_mha_weights,
    shard_mlp_weights,
    shard_lm_head_weights,
    update_heads_info,
    TensorParallelRowLinear,
    TensorParallelLMhead,
    TensorParallelConv2d,
)
from intel_extension_for_pytorch.cpu import comm as ipex_comm

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.35.2"]
    )
    import transformers
    from transformers import AutoConfig

from common_utils import TestCase

torch.manual_seed(128)

curpath = os.path.abspath(os.path.dirname(__file__))

from hf_configs.yuan.yuan_hf_model import YuanForCausalLM
from hf_configs.phi.modeling_phi import PhiForCausalLM


class TensorParallelTester(TestCase):
    def _shard_model(self, model):
        rank = ipex_comm.get_rank()
        world_size = ipex_comm.get_world_size()
        supported_mha_classes = [
            transformers.models.llama.modeling_llama.LlamaAttention,
            transformers.models.gptj.modeling_gptj.GPTJAttention,
        ]
        supported_mlp_classes = [
            transformers.models.llama.modeling_llama.LlamaMLP,
            transformers.models.gptj.modeling_gptj.GPTJMLP,
        ]
        supported_model_classes = [
            transformers.models.llama.modeling_llama.LlamaForCausalLM,
            transformers.models.gptj.modeling_gptj.GPTJForCausalLM,
        ]
        yuan_attention = None
        if isinstance(model, YuanForCausalLM):
            yuan_attention = type(model.model.layers[0].self_attn)
        if isinstance(model, YuanForCausalLM) or isinstance(model, PhiForCausalLM):
            supported_mha_classes.append(type(model.model.layers[0].self_attn))
            supported_mlp_classes.append(type(model.model.layers[0].mlp))
            supported_model_classes.append(type(model))
        for supported_mha_class in supported_mha_classes:
            num_heads = model.config.num_attention_heads
            num_kv_heads = num_heads
            for name in ["num_key_value_heads"]:
                if hasattr(model.config, name):
                    num_kv_heads = getattr(model.config, name)
            head_dim = model.config.hidden_size // num_heads
            value_with_share_qk = supported_mha_class == yuan_attention
            shard_local_filtering = supported_mha_class == yuan_attention
            shard_mha_weights(
                model,
                supported_mha_class,
                num_heads,
                num_kv_heads,
                head_dim,
                rank,
                world_size,
                value_with_share_qk,
                shard_local_filtering,
            )
        for supported_mlp_class in supported_mlp_classes:
            shard_mlp_weights(
                model,
                supported_mlp_class,
                num_heads,
                num_kv_heads,
                head_dim,
                rank,
                world_size,
            )
        for supported_model_class in supported_model_classes:
            if isinstance(model, supported_model_class):
                shard_lm_head_weights(
                    model,
                    supported_model_class,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rank,
                    world_size,
                )
                update_heads_info(model, rank, world_size)
        return model

    def tensor_parallel_with_optimize_transformers(self, model):
        input_ids = torch.ones(10).to(torch.long)
        attention_mask = torch.ones(len(input_ids))
        position_ids = torch.arange(len(input_ids))
        input_dict = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "use_cache": True,
        }
        input_dict["position_ids"] = position_ids.unsqueeze(0)
        ref_m = copy.deepcopy(model)
        for dtype in [torch.float32, torch.bfloat16]:
            ipex_model = ipex.optimize_transformers(model, dtype=dtype)
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if dtype is torch.bfloat16 else False
            ):
                key_hf = ref_m(**input_dict)
                key_ipex = ipex_model(**input_dict)

            self.assertEqual(key_hf[0], key_ipex[0], prec=0.1)

    def test_tensor_parallel_replace_check_gptj(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        model = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        tp_model = self._shard_model(copy.deepcopy(model))
        self.assertTrue(
            isinstance(tp_model.transformer.h[0].attn.out_proj, TensorParallelRowLinear)
        )
        self.assertTrue(
            isinstance(tp_model.transformer.h[0].mlp.fc_out, TensorParallelRowLinear)
        )
        self.assertTrue(isinstance(tp_model.lm_head, TensorParallelLMhead))
        self.tensor_parallel_with_optimize_transformers(model)

    def test_tensor_parallel_replace_check_llama(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        tp_model = self._shard_model(copy.deepcopy(model))
        self.assertTrue(
            isinstance(
                tp_model.model.layers[0].self_attn.o_proj, TensorParallelRowLinear
            )
        )
        self.assertTrue(
            isinstance(tp_model.model.layers[0].mlp.down_proj, TensorParallelRowLinear)
        )
        self.assertTrue(isinstance(tp_model.lm_head, TensorParallelLMhead))
        self.assertTrue(tp_model.lm_head, TensorParallelLMhead)
        self.tensor_parallel_with_optimize_transformers(model)

    def test_tensor_parallel_replace_check_yuan(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/yuan", return_dict=False, trust_remote_code=True
        )
        model = YuanForCausalLM(config).eval()
        tp_model = self._shard_model(copy.deepcopy(model))
        self.assertTrue(
            isinstance(
                tp_model.model.layers[0].self_attn.o_proj, TensorParallelRowLinear
            )
        )
        self.assertTrue(
            isinstance(tp_model.model.layers[0].mlp.down_proj, TensorParallelRowLinear)
        )
        self.assertTrue(
            isinstance(
                tp_model.model.layers[0].self_attn.lf_gate.conv1, TensorParallelConv2d
            )
        )
        self.assertTrue(
            isinstance(
                tp_model.model.layers[0].self_attn.lf_gate.conv2, TensorParallelConv2d
            )
        )
        self.assertTrue(isinstance(tp_model.lm_head, TensorParallelLMhead))
        self.assertTrue(tp_model.lm_head, TensorParallelLMhead)
        self.tensor_parallel_with_optimize_transformers(model)

    def test_tensor_parallel_replace_check_phi(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/phi", return_dict=False, trust_remote_code=True
        )
        model = PhiForCausalLM(config).eval()
        tp_model = self._shard_model(copy.deepcopy(model))
        self.assertTrue(
            isinstance(
                tp_model.model.layers[0].self_attn.dense, TensorParallelRowLinear
            )
        )
        self.assertTrue(
            isinstance(tp_model.model.layers[0].mlp.fc2, TensorParallelRowLinear)
        )
        self.assertTrue(isinstance(tp_model.lm_head, TensorParallelLMhead))
        self.assertTrue(tp_model.lm_head, TensorParallelLMhead)
        self.tensor_parallel_with_optimize_transformers(model)


if __name__ == "__main__":
    test = unittest.main()
