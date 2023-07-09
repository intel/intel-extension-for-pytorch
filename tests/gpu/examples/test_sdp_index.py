import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F

from torch.testing._internal.common_utils import TestCase

checking_atol = 1e-3 
checking_rtol = 1e-3

class TestTorchMethod(TestCase):
    def test_fsdp_atten_mask_alibi_stride(self, dtype=torch.float16):
        query = torch.empty([1, 4, 16, 128], dtype=dtype).xpu() 
        key = torch.empty([32, 4, 16, 128], dtype=dtype).xpu() 
        value = torch.empty([32, 4, 16, 128], dtype=dtype).xpu() 
        key_cache = torch.empty([10, 4, 16, 128], dtype=dtype).xpu()
        value_cache = torch.empty([10, 4, 16, 128], dtype=dtype).xpu()
        index = torch.empty([4, 128]).xpu()
        timestamp = 10
        atten_mask = None
        dropout = 0.0
        is_causal = False

        out = torch.xpu.IpexSDP_Index(query, key, value, key_cache, value_cache, index, timestamp, atten_mask, dropout, is_causal)

        print("out_shape = ", out.shape)
        print('out = ', out)

