import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex  # noqa


class RMSNormRef(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class TestNNMethod(TestCase):
    def test_rms_norm(self):
        def test_rms_norm_fw_xpu(dtype):
            print("test_rms_norm_fw_xpu", dtype)
            modelb = RMSNormRef(64)
            model0 = RMSNormRef(768)
            model1 = RMSNormRef(2048)
            model2 = RMSNormRef(4096)
            model3 = RMSNormRef(16384)
            model4 = RMSNormRef(16384 * 4 + 123)
            hszs = [64, 768, 2048, 4096, 16384, 16384 * 4 + 123]
            ls = [modelb, model0, model1, model2, model3, model4]
            for i, model in enumerate(ls):
                model = model.to(dtype)
                hsz = hszs[i]
                input_case = torch.rand(4, 1024, hsz).to(dtype)
                output_ref = model(input_case)
                input_case = input_case.xpu()
                w = model.weight.xpu()
                output = torch.ops.torch_ipex.rms_norm(input_case, [hsz], w, 1e-5)
                output1 = ipex.llm.modules.RMSNorm.apply(input_case, w, 1e-5)
                output2 = torch.xpu.IpexRmsNorm(input_case, [hsz], w, 1e-5)
                # diff = (output.cpu() - output_ref).abs().max().item()
                # print('diff', diff)
                # assert diff < 1e-2
                self.assertEqual(output[0].cpu(), output_ref, atol=1e-2, rtol=1e-2)
                self.assertEqual(output1.cpu(), output_ref, atol=1e-2, rtol=1e-2)
                self.assertEqual(output2.cpu(), output_ref, atol=1e-2, rtol=1e-2)

        test_rms_norm_fw_xpu(torch.float)
        test_rms_norm_fw_xpu(torch.bfloat16)

    def test_rms_norm_bw(self):
        def test_rms_norm_fwd_bwd(dtype):
            print("test_rms_norm_fw_bw", dtype)
            torch.manual_seed(13)
            modelb = RMSNormRef(64)
            model0 = RMSNormRef(768)
            model1 = RMSNormRef(2048)
            model2 = RMSNormRef(4096)
            model3 = RMSNormRef(16384)
            model4 = RMSNormRef(16384 * 4 + 123)
            hszs = [64, 768, 2048, 4096, 16384, 16384 * 4 + 123]
            ls = [modelb, model0, model1, model2, model3, model4]
            for i, model in enumerate(ls):
                model = model.to(dtype)
                hsz = hszs[i]
                input_case = torch.rand(4, 1024, hsz).to(dtype)
                input_case.requires_grad_(True)
                grad = torch.rand(4, 1024, hsz).to(dtype)
                output_ref = model(input_case)
                output_ref.backward(grad)
                grad_wei = model.weight.grad.clone()
                input_grad_cpu = input_case.grad.clone()
                w = model.weight.clone()

                input_case_xpu = input_case.clone().xpu()
                input_case_xpu.retain_grad()
                input_case_xpu.requires_grad_(True)
                grad_xpu = grad.xpu()
                w = w.xpu()
                w.retain_grad()
                w.requires_grad_(True)
                output1 = torch.xpu.IpexRmsNorm(input_case_xpu, [hsz], w, 1e-5)
                output1.backward(grad_xpu)
                grad_wei_xpu = w.grad

                self.assertEqual(grad_wei_xpu.cpu(), grad_wei, atol=10e-2, rtol=10e-2)
                self.assertEqual(
                    input_case_xpu.grad.cpu(), input_grad_cpu, atol=10e-2, rtol=10e-2
                )

        test_rms_norm_fwd_bwd(torch.bfloat16)
        test_rms_norm_fwd_bwd(torch.float)
