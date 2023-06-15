import torch
import intel_extension_for_pytorch
import torch.nn.functional as F

from torch.testing._internal.common_utils import TestCase

class TestTorchMethod(TestCase):
    def test_sdp_mem_effi_half(self, dtype=torch.float16):
        head_dim = 256 
        seq_lenth = 1
        k_seq_lenth = 33
        v_seq_lenth = 33
        query = torch.rand(1, 16, seq_lenth, head_dim, dtype=dtype)
        key = torch.rand(1, 16, k_seq_lenth, head_dim, dtype=dtype)
        value = torch.rand(1, 16, v_seq_lenth, head_dim, dtype=dtype)

        out_cpu = F.scaled_dot_product_attention(query.float(),key.float(),value.float())
        out_xpu = F.scaled_dot_product_attention(query.xpu(),key.xpu(),value.xpu())

        self.assertEqual(out_cpu, out_xpu.cpu().float())

#    def test_sdp_mem_effi_bf16(self, dtype=torch.bfloat16):
#        head_dim = 256
#        seq_lenth = 1
#        k_seq_lenth = 32
#        v_seq_lenth = 32
#        query = torch.rand(1, 16, seq_lenth, head_dim, dtype=dtype)
#        key = torch.rand(1, 16, k_seq_lenth, head_dim, dtype=dtype)
#        value = torch.rand(1, 16, v_seq_lenth, head_dim, dtype=dtype)
#
#        out_cpu = F.scaled_dot_product_attention(query,key,value)
#        out_xpu = F.scaled_dot_product_attention(query.xpu(),key.xpu(),value.xpu())
#
#        self.assertEqual(out_cpu, out_xpu.cpu())
