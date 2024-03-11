import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_bmm(self):
        def _test_bmm(device, dtype):
            # error case: one is nested but the other is not
            nt = torch.nested.nested_tensor(
                [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
            )
            t = torch.randn(4, device=device, dtype=dtype)
            self.assertRaisesRegex(
                RuntimeError,
                "Expected both to be nested, but got a nested self and non-nested other",
                lambda: nt.bmm(t),
            )
            self.assertRaisesRegex(
                RuntimeError,
                "Expected both to be nested, but got a non-nested self and nested other",
                lambda: t.bmm(nt),
            )
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
