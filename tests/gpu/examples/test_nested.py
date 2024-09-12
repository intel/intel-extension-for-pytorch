import torch
import pytest  # noqa
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


def random_nt(device, dtype, num_tensors, max_dims, min_dims=None):
    if min_dims is None:
        min_dims = tuple([0] * len(max_dims))
    ts1 = []
    for _ in range(num_tensors):
        tensor_dims = tuple(
            [
                torch.randint(low=min_dim, high=max_dim, size=(1,)).item()
                for (min_dim, max_dim) in zip(min_dims, max_dims)
            ]
        )
        t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
        ts1.append(t1)
    return torch.nested.nested_tensor(ts1, device=device, dtype=dtype)


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_transform_bias_rescale_qkv_nested(self, device="xpu", dtype=torch.float32):
        tests = [
            (64, 4, 16, 8),
            (24, 2, 4, 2),
            (2, 2, 2, 2),
            (24, 4, 4, 2),
            (48, 4, 16, 8),
        ]
        for embed_dim, num_heads, bs, sl in tests:
            dense_x = x = torch.randn(
                bs, sl, 3 * embed_dim, device="cpu", dtype=torch.float32
            )
            xs = list(torch.unbind(x))
            x = torch.nested.nested_tensor(xs, device="cpu", dtype=torch.float32)
            x_xpu = x.to("xpu")

            qkv = torch.nn.Linear(
                embed_dim, 3 * embed_dim, device="cpu", dtype=torch.float32
            )
            bias = qkv.bias
            bias_xpu = bias.to("xpu")

            self.assertEqual(x.to(cpu_device), x_xpu.to(cpu_device))
            self.assertEqual(bias.to(cpu_device), bias_xpu.to(cpu_device))

            (q, k, v) = torch._transform_bias_rescale_qkv(x, bias, num_heads=num_heads)
            (q_xpu, k_xpu, v_xpu) = torch._transform_bias_rescale_qkv(
                x_xpu, bias_xpu, num_heads=num_heads
            )

            self.assertEqual(q.to(cpu_device), q_xpu.to(cpu_device))
            self.assertEqual(k.to(cpu_device), k_xpu.to(cpu_device))
            self.assertEqual(v.to(cpu_device), v_xpu.to(cpu_device))

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_transform_bias_rescale_qkv_non_nested(
        self, device="xpu", dtype=torch.float32
    ):
        tests = [
            (64, 4, 16, 8),
            (24, 2, 1, 1),
            (2, 2, 2, 2),
            (24, 4, 4, 2),
            (48, 4, 16, 8),
        ]
        for embed_dim, num_heads, bs, seq_len in tests:
            x = torch.randn(bs, seq_len, 3 * embed_dim, device="cpu", dtype=dtype)
            x_xpu = x.to("xpu")

            qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device="cpu", dtype=dtype)
            bias = qkv.bias
            xpu_bias = bias.to("xpu")

            self.assertEqual(x.to(cpu_device), x_xpu.to(cpu_device))
            self.assertEqual(bias.to(cpu_device), xpu_bias.to(cpu_device))

            (q, k, v) = torch._transform_bias_rescale_qkv(x, bias, num_heads=num_heads)
            (q_xpu, k_xpu, v_xpu) = torch._transform_bias_rescale_qkv(
                x_xpu, xpu_bias, num_heads=num_heads
            )

            self.assertEqual(q.to(cpu_device), q_xpu.to(cpu_device))
            self.assertEqual(k.to(cpu_device), k_xpu.to(cpu_device))
            self.assertEqual(v.to(cpu_device), v_xpu.to(cpu_device))

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_softmax(self):
        def _test_softmax(device, dtype):
            # normal nested tensor
            ntensors = 4
            nt = random_nt(device, dtype, ntensors, (4, 4))
            # error case: softmax across nested dimension
            self.assertRaisesRegex(
                RuntimeError,
                "Cannot apply softmax across nested dimension 0",
                lambda: torch.nn.functional.softmax(nt, 0),
            )
            self.assertRaisesRegex(
                RuntimeError,
                "Cannot apply softmax across nested dimension 0",
                lambda: torch.nn.functional.softmax(nt, -3),
            )
            # error case: dimension out of range
            self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, 3))
            self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, -4))
            # normal case: should equal to padding -inf
            softmaxer = torch.nn.Softmax(1)
            y0 = softmaxer(nt)
            y1 = torch.nn.functional.softmax(nt, 1)
            self.assertEqual(y0, y1)
            pt = torch.nested.to_padded_tensor(nt, float("-inf"))
            # if an entire slice is padded, then softmax will return 0.0 / 0.0 = nan
            # however, physically speaking that should be 0.0
            expect = torch.nn.functional.softmax(pt, 1).nan_to_num_(0.0)
            self.assertEqual(torch.nested.to_padded_tensor(y0, 0.0), expect)
            # edge case: empty nested tensor
            nt0 = torch.nested.nested_tensor([])
            y = torch.nn.functional.softmax(nt0, 1)
            self.assertEqual(nt0, y)
            # edge case: nesting scalars
            nt1 = torch.nested.nested_tensor([torch.tensor(0.0), torch.tensor(1.0)])
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.softmax(nt1, 0))
            self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt1, 1))

        _test_softmax(dpcpp_device, torch.float32)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_bmm(self):
        def _test_bmm(device, dtype):
            # error case: not 3D tensors
            nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
            nt1 = torch.nested.nested_tensor(
                [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
            )
            nt2 = torch.nested.nested_tensor(
                [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
            )
            self.assertRaisesRegex(
                RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt0)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt1)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt2)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt0)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt1)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt2)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch2 must be a 3D tensor", lambda: nt2.bmm(nt0)
            )
            self.assertRaisesRegex(
                RuntimeError, "batch2 must be a 3D tensor", lambda: nt2.bmm(nt1)
            )
            # error case: incompatible batch size
            nt0 = torch.nested.nested_tensor(
                [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
            )
            nt1 = torch.nested.nested_tensor(
                [torch.randn((4, 6)), torch.randn((4, 5)), torch.randn((4, 7))],
                device=device,
                dtype=dtype,
            )
            self.assertRaisesRegex(
                RuntimeError,
                "Expected size for the 1st dimension of batch2 tensor to be: 2 but got: 3.",
                lambda: nt0.bmm(nt1),
            )
            self.assertRaisesRegex(
                RuntimeError,
                "Expected size for the 1st dimension of batch2 tensor to be: 3 but got: 2.",
                lambda: nt1.bmm(nt0),
            )
            # error case: underlying matrices cannot be multiplied
            nt0 = torch.nested.nested_tensor(
                [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
            )
            self.assertRaisesRegex(
                RuntimeError,
                r"0-th nested matrices in batch cannot be multiplied \(2x4 and 2x4\)",
                lambda: nt0.bmm(nt0),
            )
            # normal nested tensor
            nt0_0 = torch.randn((4, 4))
            nt0_1 = torch.randn((4, 4))
            nt1_0 = torch.randn((4, 10))
            nt1_1 = torch.randn((4, 10))
            nt0_cpu = torch.nested.nested_tensor(
                [nt0_0, nt0_1], device=cpu_device, dtype=dtype
            )
            nt1_cpu = torch.nested.nested_tensor(
                [nt1_0, nt1_1], device=cpu_device, dtype=dtype
            )
            nt0_xpu = torch.nested.nested_tensor(
                [nt0_0, nt0_1], device=dpcpp_device, dtype=dtype
            )
            nt1_xpu = torch.nested.nested_tensor(
                [nt1_0, nt1_1], device=dpcpp_device, dtype=dtype
            )
            actual = nt0_xpu.bmm(nt1_xpu)
            expect = nt0_cpu.bmm(nt1_cpu)
            if dtype == torch.float16:
                self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
            else:
                self.assertEqual(actual, expect)

        _test_bmm(dpcpp_device, torch.float32)
        _test_bmm(dpcpp_device, torch.double)
        _test_bmm(dpcpp_device, torch.bfloat16)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_alias(self):
        def _test_alias(device, dtype):
            # error case: one is nested but the other is not
            nt = torch.nested.nested_tensor(
                [torch.randn(2), torch.randn(3)], device=cpu_device, dtype=dtype
            )
            aten = torch.ops.aten
            expect = aten.alias(nt)
            actual = aten.alias(nt.to(device)).to(cpu_device)
            self.assertEqual(actual, expect)

        _test_alias(dpcpp_device, torch.float32)
        _test_alias(dpcpp_device, torch.double)
        _test_alias(dpcpp_device, torch.bfloat16)
