import torch
import torch.nn as nn
import intel_extension_for_pytorch  # noqa
import copy
import pytest
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_clip_grad_norm_with_cpu(self):
        def run_test(norm_type=0.5, foreach=True, dtype=torch.float32):
            l_cpu = nn.Linear(20, 100).to("cpu").to(dtype)
            l_xpu = copy.deepcopy(l_cpu).to("xpu").to(dtype)

            rand_value = torch.rand(20, 100, device="cpu").to(dtype)

            grads_cpu = rand_value, torch.ones(100, device="cpu").div(1000).to(dtype)
            grads_xpu = copy.deepcopy(rand_value).to("xpu"), torch.ones(
                100, device="xpu"
            ).div(1000).to(dtype)

            for p, g in zip(l_cpu.parameters(), grads_cpu):
                p._grad = g.clone().view_as(p.data).to(dtype)

            for p, g in zip(l_xpu.parameters(), grads_xpu):
                p._grad = g.clone().view_as(p.data).to(dtype)

            max_norm = 2
            cpu_result = torch.nn.utils.clip_grad_norm_(
                l_cpu.parameters(), max_norm, norm_type=norm_type, foreach=foreach
            )
            xpu_result = torch.xpu.utils.clip_grad_norm_(
                l_xpu.parameters(), max_norm, norm_type=norm_type, foreach=foreach
            )

            print("python cpu result:", cpu_result)
            print("python xpu result:", xpu_result)
            checking_atol = 1e-5
            checking_rtol = 1.3e-6
            if dtype == torch.bfloat16:
                checking_atol = 1e-3
                checking_rtol = 1.6e-2
            self.assertEqual(
                cpu_result, xpu_result, atol=checking_atol, rtol=checking_rtol
            )

        run_test(norm_type="inf", foreach=False, dtype=torch.float32)
        run_test(norm_type=0, foreach=False, dtype=torch.float32)
        run_test(norm_type=1, foreach=False, dtype=torch.float32)
        run_test(norm_type=2, foreach=False, dtype=torch.float32)
        run_test(norm_type="inf", foreach=False, dtype=torch.bfloat16)
        run_test(norm_type=0, foreach=False, dtype=torch.bfloat16)
        run_test(norm_type=1, foreach=False, dtype=torch.bfloat16)
        run_test(norm_type=2, foreach=False, dtype=torch.bfloat16)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_clip_grad_norm_multitensor(self):
        def run_test(norm_type=0.5, foreach=True, dtype=torch.float32):
            l_cpu = nn.Linear(10, 10).to("cpu").to(dtype)
            l_xpu = copy.deepcopy(l_cpu).to("xpu").to(dtype)

            rand_value = torch.rand(10, 10, device="cpu").to(dtype)

            grads_cpu = 513 * [
                rand_value,
                torch.ones(10, device="cpu").div(1000).to(dtype),
            ]
            grads_xpu = 513 * [
                rand_value.to("xpu"),
                torch.ones(10, device="xpu").div(1000).to(dtype),
            ]

            for p, g in zip(l_cpu.parameters(), grads_cpu):
                p._grad = g.clone().view_as(p.data).to(dtype)

            for p, g in zip(l_xpu.parameters(), grads_xpu):
                p._grad = g.clone().view_as(p.data).to(dtype)

            max_norm = 2
            cpu_result = torch.nn.utils.clip_grad_norm_(
                l_cpu.parameters(), max_norm, norm_type=norm_type, foreach=foreach
            )
            xpu_result = torch.xpu.utils.clip_grad_norm_(
                l_xpu.parameters(), max_norm, norm_type=norm_type, foreach=foreach
            )

            print("python cpu result:", cpu_result)
            print("python xpu result:", xpu_result)
            checking_atol = 1e-5
            checking_rtol = 1.3e-6
            if dtype == torch.bfloat16:
                checking_atol = 1e-3
                checking_rtol = 1.6e-2
            self.assertEqual(
                cpu_result, xpu_result, atol=checking_atol, rtol=checking_rtol
            )

        run_test(norm_type="inf", foreach=False, dtype=torch.bfloat16)
        run_test(norm_type=0, foreach=False, dtype=torch.bfloat16)
        run_test(norm_type=1, foreach=False, dtype=torch.bfloat16)
        run_test(norm_type=2, foreach=False, dtype=torch.bfloat16)
        run_test(norm_type="inf", foreach=False, dtype=torch.float32)
        run_test(norm_type=0, foreach=False, dtype=torch.float32)
        run_test(norm_type=1, foreach=False, dtype=torch.float32)
        run_test(norm_type=2, foreach=False, dtype=torch.float32)
