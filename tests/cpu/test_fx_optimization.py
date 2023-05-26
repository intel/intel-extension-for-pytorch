import unittest

import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    _IPEXLinear as _IPEXLinear,
)
from torch.testing._internal.common_utils import TestCase
from typing import List
import random
import copy
import itertools
import os

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
except RuntimeError:
    HAS_TRANSFORMERS = False
skipIfNoTRANSFORMERS = unittest.skipIf(not HAS_TRANSFORMERS, "no transformers")

try:
    import diffusers

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
except RuntimeError:
    HAS_DIFFUSERS = False
skipIfNoDIFFUSERS = unittest.skipIf(not HAS_DIFFUSERS, "no diffusers")


class MultipleLinear(torch.nn.Module):
    def __init__(
        self, out_fs: List[int], in_fs: List[int], bias: bool, dtype: torch.dtype
    ):
        super(MultipleLinear, self).__init__()
        self.l0 = torch.nn.Linear(in_fs[0], out_fs[0], bias=bias, dtype=dtype)
        self.l1 = torch.nn.Linear(in_fs[1], out_fs[1], bias=bias, dtype=dtype)
        self.l2 = torch.nn.Linear(in_fs[2], out_fs[2], bias=bias, dtype=dtype)

    def forward(self, x):
        out0 = self.l0(x)
        out1 = self.l1(x)
        out2 = self.l2(x)
        return out0, out1, out2


class FxTester(TestCase):
    def _check_concat(self, model_before_concat, model_after_concat):
        def is_linear(m):
            return isinstance(child, torch.nn.Linear) or isinstance(child, _IPEXLinear)

        # checkout whether all linears on model_before_concat
        # is concated on model_after_concat
        total_out_f = 0
        for _, child in model_before_concat.named_modules():
            if is_linear(child):
                total_out_f += child.out_features

        found_linear = False
        for _, child in model_after_concat.named_modules():
            if is_linear(child):
                self.assertFalse(found_linear)
                self.assertEqual(child.out_features, total_out_f)
                found_linear = True
        self.assertTrue(found_linear)

    def test_concat_linear(self):
        _bias = [True, False]
        _inplace = [True, False]
        _in_feature = [16, 129]
        _dtype = [torch.float, torch.bfloat16]
        options = itertools.product(_bias, _inplace, _in_feature, _dtype)
        for bias, inplace, in_feature, dtype in options:
            x = torch.randn(100, in_feature, dtype=dtype)
            out_fs = random.sample(range(128), 3)
            in_fs = [in_feature] * 3
            m = MultipleLinear(out_fs, in_fs, bias, dtype)
            y1 = m(x)
            gm = torch.fx.symbolic_trace(m)
            y2 = gm(x)
            self.assertEqual(y1, y2)
            concat_gm = ipex.fx.concat_linear.concat_linear(
                copy.deepcopy(gm), inplace=inplace
            )
            y3 = concat_gm(x)
            self.assertEqual(y1, y3)
            # checkout success concat
            self._check_concat(gm, concat_gm)

    @skipIfNoTRANSFORMERS
    def test_concat_linear_hf_bert(self):
        from transformers import AutoModelForCausalLM, AutoConfig
        from transformers.utils.fx import symbolic_trace as hf_symbolic_trace

        loc = os.path.dirname(os.path.abspath(__file__))
        config = AutoConfig.from_pretrained(loc + "/bert-base-config.json")
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        inputs = torch.load(loc + "/bert-inputs.pt")
        gm = hf_symbolic_trace(model, input_names=list(inputs.keys()))
        ref_out = gm(**inputs)
        concat_gm = ipex.fx.concat_linear.concat_linear(copy.deepcopy(gm), inplace=True)
        out = concat_gm(**inputs)
        self.assertEqual(out, ref_out)
        # checkout success concat
        for layer_id in range(12):
            self_att = getattr(gm.bert.encoder.layer, str(layer_id)).attention.self
            concat_self_att = getattr(
                concat_gm.bert.encoder.layer, str(layer_id)
            ).attention.self
            self._check_concat(self_att, concat_self_att)

    @skipIfNoTRANSFORMERS
    def test_automatically_apply_concat_linear_with_ipex_optimize(self):
        from transformers import AutoModelForCausalLM, AutoConfig

        loc = os.path.dirname(os.path.abspath(__file__))
        config = AutoConfig.from_pretrained(loc + "/bert-base-config.json")
        base_model = AutoModelForCausalLM.from_config(config).eval()
        inputs = torch.load(loc + "/bert-inputs.pt")
        for dtype in [torch.float, torch.bfloat16]:
            for inplace in [True, False]:
                model = copy.deepcopy(base_model)
                auto_cast = dtype == torch.bfloat16
                with torch.cpu.amp.autocast(auto_cast, dtype=torch.bfloat16):
                    ref_out = model(**inputs)
                ipex_model = ipex.optimize(
                    model, dtype=dtype, inplace=inplace, concat_linear=True
                )
                with torch.cpu.amp.autocast(auto_cast, dtype=torch.bfloat16):
                    out = ipex_model(**inputs)
                if dtype == torch.bfloat16:
                    self.assertEqual(out, ref_out, rtol=5e-2, atol=5e-2)
                else:
                    self.assertEqual(out, ref_out)
                for layer_id in range(12):
                    self_att = getattr(
                        model.bert.encoder.layer, str(layer_id)
                    ).attention.self
                    concat_self_att = getattr(
                        ipex_model.bert.encoder.layer, str(layer_id)
                    ).attention.self
                    self._check_concat(self_att, concat_self_att)

    @skipIfNoDIFFUSERS
    def test_stable_diffusion(self):
        def check_unet_concated(model):
            for child in model.children():
                check_unet_concated(child)
            if isinstance(model, diffusers.models.attention.BasicTransformerBlock):
                attn1 = model.attn1
                attn2 = model.attn2
                original_out_f = attn2.L__self___to_q.out_features
                self.assertTrue(hasattr(attn1, "L__self___to_q"))
                self.assertFalse(hasattr(attn1, "L__self___to_k"))
                self.assertFalse(hasattr(attn1, "L__self___to_v"))
                self.assertEqual(original_out_f * 3, attn1.L__self___to_q.out_features)
                self.assertTrue(hasattr(attn2, "L__self___to_q"))
                self.assertTrue(hasattr(attn2, "L__self___to_k"))
                self.assertFalse(hasattr(attn2, "L__self___to_v"))
                self.assertEqual(original_out_f * 2, attn2.L__self___to_k.out_features)

        models_list = [
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
        ]
        loc = os.path.dirname(os.path.abspath(__file__))
        for model_id in models_list:
            config_dir = loc + "/stable-difusion-config/" + model_id
            unet_cls = diffusers.models.unet_2d_condition.UNet2DConditionModel
            base_model = unet_cls.from_config(config_dir).eval()

            if model_id == "stabilityai/stable-diffusion-2-1":
                input = (
                    torch.randn(4, 4, 64, 64),
                    torch.tensor(921),
                    torch.randn(4, 77, 1024),
                )
            else:
                input = (
                    torch.randn(2, 4, 64, 64),
                    torch.tensor(921),
                    torch.randn(2, 77, 768),
                )
            for dtype in [torch.float, torch.bfloat16]:
                for inplace in [True, False]:
                    model1 = copy.deepcopy(base_model)
                    model2 = copy.deepcopy(base_model)
                    auto_cast = dtype == torch.bfloat16
                    ipex_model1 = ipex.optimize(
                        model1, dtype=dtype, inplace=inplace, concat_linear=True
                    )
                    check_unet_concated(ipex_model1)
                    ipex_model2 = ipex.optimize(
                        model2, dtype=dtype, inplace=inplace, concat_linear=False
                    )
                    with torch.cpu.amp.autocast(auto_cast, dtype=torch.bfloat16):
                        out1 = ipex_model1(*input)
                        out2 = ipex_model2(*input)
                    self.assertEqual(out1, out2)


if __name__ == "__main__":
    test = unittest.main()
