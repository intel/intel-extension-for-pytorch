import sys
import os
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _IPEXLinear,
    _IPEXLinearAllreduce,
)
from intel_extension_for_pytorch.quantization import prepare, convert
from intel_extension_for_pytorch.quantization._quantize import (
    DynamicQuantizedLinearLayer,
    DynamicQuantizedLinearAllreduce,
)

from test_weight_prepack import module_found


class MyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # For deepspeed support, please do not change the name of the attribute.
        self.q_proj = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 2)

    def forward(self, x):
        x = self.q_proj(x)
        z = self.out_proj(x)
        return z


class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MyAttention()

    def forward(self, x):
        z = self.attn(x)
        return z


# For deepspeed support, please do not change the name of the class.
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # For deepspeed support, please do not change the ModuleList structure of the class.
        self.linears = nn.ModuleList([MyBlock()])

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x


# The class DeepSpeedTestM is written for deepspeed to recognize the modules and to be functional.
# Please do not change it.
class DeepSpeedTestM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = MyModel()

    def forward(self, x):
        z = self.linear(x)
        return z


class DeepspeedTester(TestCase):
    def _get_ds_model(self, m_linear):
        import deepspeed

        ds_world_size = int(os.getenv("WORLD_SIZE", "1"))
        assert (
            ds_world_size > 1
        ), "expect ds_world_size > 1, you could try launching the script with: \
            deepspeed --num_gpus 2 --bind_cores_to_rank tests/cpu/test_deepspeed.py"
        engine = deepspeed.init_inference(
            model=m_linear,
            mp_size=ds_world_size,
            dtype=torch.float32,
            replace_method="auto",
        )
        ds_model = engine.module
        return ds_model

    def test_ipex_optimize(self):
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            with torch.no_grad():
                LinearAllreduce, LinearLayer = deepspeed_modules
                x = torch.randn(2, 4)
                m_linear = DeepSpeedTestM().eval()
                y = m_linear(x)

                ds_model = self._get_ds_model(m_linear)
                self.assertTrue(module_found(ds_model, LinearLayer))
                self.assertTrue(module_found(ds_model, LinearAllreduce))

                optimized = ipex.optimize(ds_model.eval(), inplace=True)
                jit_optimized = torch.jit.trace(optimized, x)
                jit_optimized = torch.jit.freeze(jit_optimized)
                self.assertTrue(module_found(optimized, _IPEXLinear))
                self.assertTrue(module_found(optimized, _IPEXLinearAllreduce))

                optimized = optimized(x)
                jit_res = jit_optimized(x)
                self.assertEqual(y, jit_res)
                self.assertEqual(y, optimized)

    def test_dynamic_quantization(self):
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            LinearAllreduce, LinearLayer = deepspeed_modules
            x = torch.randn(2, 4)
            m_linear = DeepSpeedTestM().eval()
            y = m_linear(x)

            ds_model = self._get_ds_model(m_linear)
            self.assertTrue(module_found(ds_model, LinearLayer))
            self.assertTrue(module_found(ds_model, LinearAllreduce))

            dynamic_qconfig = ipex.quantization.default_dynamic_qconfig
            prepared_model = prepare(
                ds_model,
                dynamic_qconfig,
                example_inputs=(x),
                inplace=True,
                bn_folding=False,
            )
            converted = convert(prepared_model, inplace=True)
            self.assertTrue(module_found(converted, DynamicQuantizedLinearLayer))
            self.assertTrue(module_found(converted, DynamicQuantizedLinearAllreduce))
            quantized = converted(x)
            self.assertEqual(y, quantized, atol=0.005, rtol=1.3e-6)


if __name__ == "__main__":
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        # when launching with deepspeed, the cmd will be python -u tests/cpu/test_deepspeed.py --local_rank=xx
        # Need to handle the --local_rank before unittest.main()
        if len(sys.argv) > 1:
            local_rank = sys.argv.pop()

        test = unittest.main()
