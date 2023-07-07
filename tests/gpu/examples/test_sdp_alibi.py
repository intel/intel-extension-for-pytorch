import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F

from torch.testing._internal.common_utils import TestCase

checking_atol = 1e-3 
checking_rtol = 1e-3

class TestTorchMethod(TestCase):
    def test_fsdp_atten_mask_alibi_stride(self, dtype=torch.float16):
        query = torch.permute(torch.reshape(torch.rand(1,1,4096),(1,1,16,256)),(0,2,1,3)).half()
        key = torch.permute(torch.reshape(torch.rand(1,2048,4096),(1,2048,16,256)),(0,2,1,3))[:,:,:33,:].half()
        value = torch.permute(torch.reshape(torch.rand(1,2048,4096),(1,2048,16,256)),(0,2,1,3))[:,:,:33,:].half()
        alibi = torch.empty(1).xpu()
        atten_mask = torch.empty(1).xpu()
        head_mask = None
        alpha = 1.0
        beta = 1.0
        dropout = 0.0
        is_causal = False

        out = torch.xpu.IpexSDP(query.xpu(), key.xpu(), value.xpu(), atten_mask, alibi, head_mask, alpha, beta, dropout, is_causal)

        print('out = ', out)

