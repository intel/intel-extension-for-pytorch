import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import itertools

device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_clamp_promotion(self):
        test_dtypes = [
            [torch.bool, torch.bool, torch.float],
            [torch.bool, torch.float, torch.bool],
        ]

        def test_one_clamp_promotion(dtypes):
            dtype0, dtype1, dtype2 = dtypes
            S = 4

            def make_tensor(size, dtype):
                if dtype == torch.bool:
                    return torch.randint(2, size, dtype=dtype, device=device)
                elif dtype == torch.int:
                    return torch.randint(10, size, dtype=dtype, device=device)
                else:
                    return torch.randn(size, dtype=dtype, device=device)

            min_t = make_tensor((S,), dtype1)
            max_t = make_tensor((S,), dtype2)
            mins = (min_t, min_t[0], min_t[0].item())
            maxs = (max_t, max_t[0], max_t[0].item())
            inp = make_tensor((S,), dtype0)
            for min_v, max_v in itertools.product(mins, maxs):
                if type(max_v) != type(min_v):
                    continue
                if (
                    isinstance(min_v, torch.Tensor)
                    and min_v.ndim == 0
                    and max_v.ndim == 0
                ):
                    continue  # 0d tensors go to scalar overload, and it's tested separately

                def expected_type(inp, max, min):
                    arg1, arg2 = max, min
                    if isinstance(max, torch.Tensor) and max.ndim == 0:
                        # first do a maybe dimensional boundary
                        arg1, arg2 = min, max
                    exp_type = torch.result_type(inp, arg1)
                    inp_new = torch.empty_like(inp, dtype=exp_type)
                    return torch.result_type(inp_new, arg2)

                exp_type = expected_type(inp, min_v, max_v)
                if exp_type != torch.bool:
                    actual = torch.clamp(inp, min_v, max_v)
                    inps = [
                        x.to(exp_type) if isinstance(x, torch.Tensor) else x
                        for x in (inp, min_v, max_v)
                    ]
                    expected = torch.clamp(inps[0], inps[1], inps[2])
                    self.assertEqual(actual, expected)
            for val in mins:

                def expected_type(inp, val):
                    return torch.result_type(inp, val)

                exp_type = expected_type(inp, val)
                if exp_type != torch.bool:
                    actual = torch.clamp_min(inp, val)
                    inps = [
                        x.to(exp_type) if isinstance(x, torch.Tensor) else x
                        for x in (inp, val)
                    ]
                    expected = torch.clamp_min(inps[0], inps[1])
                    self.assertEqual(actual.dtype, exp_type)
                    self.assertEqual(actual, expected)
                    if inp.dtype == exp_type:
                        actual = torch.clamp_min_(inp, val)
                        self.assertEqual(actual, expected)
                    actual = torch.clamp_max(inp, val)
                    expected = torch.clamp_max(inps[0], inps[1])
                    self.assertEqual(actual, expected)

        for dtypes in test_dtypes:
            test_one_clamp_promotion(dtypes)
