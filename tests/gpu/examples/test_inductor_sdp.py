import torch
import intel_extension_for_pytorch as ipex  # noqa
import math
import pytest

from torch.testing._internal.common_utils import TestCase

torch.set_printoptions(profile="full")


# torch bench llama
def fun(query, key, value):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    scores = torch.softmax(scores.float(), dim=-1).type_as(query)
    output = torch.matmul(scores, value)
    return output


b = 2
n = 4
q = 8
kv = 8
h = 256


class TestTorchMethod(TestCase):
    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="ipex build without xetla"
    )
    def test_llama_sdp_fusion(self, dtype=torch.float16):
        query_states = torch.randn(
            (b, q, n, h), device=torch.device("xpu"), dtype=dtype, requires_grad=False
        )
        key_states = torch.randn(
            (b, kv, n, h), device=torch.device("xpu"), dtype=dtype, requires_grad=False
        )
        value_states = torch.randn(
            (b, kv, n, h), device=torch.device("xpu"), dtype=dtype, requires_grad=False
        )
        args_xpu = [query_states, key_states, value_states]

        ref1 = fun(*args_xpu)
        fun_in = torch.compile(fun)
        actual = fun_in(*args_xpu)
        with torch.inference_mode():
            actual = fun_in(*args_xpu)

        # print(ref1.cpu())
        # print(ref2.cpu())
        self.assertEqual(ref1.cpu(), actual.cpu(), atol=1e-3, rtol=1e-3)
