import tempfile
import torch

import copy
import os
import unittest
import transformers
from transformers import AutoConfig

from common_utils import TestCase

import intel_extension_for_pytorch as ipex

from torch.testing._internal.common_utils import run_tests
from intel_extension_for_pytorch.quantization import (
    WoqWeightDtype,
    WoqLowpMode,
)


class GPTQLLMTester(TestCase):
    def test_gptq_quantize(self):
        class GPTQLLMDataLoader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    yield torch.ones([1, 512], dtype=torch.long)

        def _get_gptj_example_inputs():
            input_ids = torch.ones(8).to(torch.long)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            past_key_values = tuple(
                [
                    (
                        torch.zeros(1, 1, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros(1, 4, dtype=torch.long),
                    )
                    for i in range(1)
                ]
            )
            return (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                past_key_values,
                position_ids.unsqueeze(0),
            )

        dataloader = GPTQLLMDataLoader()
        curpath = os.path.abspath(os.path.dirname(__file__))
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        gptj = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as work_dir:
            for act_order in [False, True]:
                model = copy.deepcopy(gptj)
                model.eval()
                compressed_model = ipex.quantization.gptq(
                    model,
                    dataloader=dataloader,
                    wbits=4,
                    group_size=128,
                    act_order=act_order,
                    use_max_length=True,
                    pad_max_length=512,
                    scale_dtype=torch.float16,
                    save_dir=work_dir,
                )
                self.assertTrue(isinstance(compressed_model, torch.nn.Module))
                input = torch.ones([1, 512], dtype=torch.long)
                out0 = model(input)
                out1 = compressed_model(input)
                self.assertTrue(torch.allclose(out0[0], out1[0], atol=1e-05))

                low_precision_checkpoint = torch.load(
                    work_dir + "/gptq_checkpoint_g128.pt"
                )
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=WoqWeightDtype.INT4,
                    lowp_mode=WoqLowpMode.INT8,
                )
                model = copy.deepcopy(gptj)
                model.eval()
                model = ipex.llm.optimize(
                    model,
                    dtype=torch.float,
                    quantization_config=qconfig,
                    inplace=True,
                    low_precision_checkpoint=low_precision_checkpoint,
                    deployment_mode=False,
                )
                _IPEXAttentionCPU = (
                    ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                )
                _IPEXDecoderLayerCPU = (
                    ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                )
                WeightOnlyQuantizedLinear = ipex.nn.modules.WeightOnlyQuantizedLinear
                assert model.transformer.h[0].attn.__class__ is _IPEXAttentionCPU
                assert model.transformer.h[0].__class__ is _IPEXDecoderLayerCPU
                layers_to_check = [
                    model.transformer.h[0].attn.out_proj,
                    model.transformer.h[0].linear_add_add.linear,
                    model.transformer.h[0].linear_gelu.linear,
                ]
                # concat linear is unsupported with act_order
                if not act_order:
                    layers_to_check.append(
                        model.transformer.h[0].attn.concat_qkv.concat_linear
                    )
                assert all(
                    mod.__class__ is WeightOnlyQuantizedLinear
                    for mod in layers_to_check
                )

                # Ensure model can run without errors
                with torch.no_grad():
                    example_inputs = _get_gptj_example_inputs()
                    # the optimized model is ipex_m.trace_graph
                    model(*example_inputs)


if __name__ == "__main__":
    test = unittest.main()
    run_tests()
