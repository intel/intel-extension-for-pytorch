from __future__ import print_function
import os
import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import unittest

try:
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
skipIfNoTransformers = unittest.skipIf(not HAS_TRANSFORMERS, "no transformers")

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
curpath = os.path.abspath(os.path.dirname(__file__))


class TestNNMethod(TestCase):
    @skipIfNoTransformers
    def test_BartEmbedding(self, dtype=torch.float16):
        from transformers.models import bart
        from intel_extension_for_pytorch.nn.utils import _parameter_wrapper

        config = transformers.AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/bart-large",
        )
        model = bart.modeling_bart.BartForCausalLM(config)
        model.to(dpcpp_device)
        params_attr = {}
        _parameter_wrapper.get_shared_parameter_status(model, params_attr)
        for name, param in model.named_parameters():
            if name == "model.decoder.embed_positions.weight":
                self.assertTrue(
                    bart.modeling_bart.BartLearnedPositionalEmbedding
                    in params_attr[param].modules_cls
                )
                self.assertEqual(param.dtype, torch.float32)
                self.assertTrue(
                    params_attr[param].can_cast_inference(torch.float16, "xpu")
                )
                params_attr[param].cast_for_inference(torch.float16)
                self.assertEqual(param.dtype, torch.float16)
                break
