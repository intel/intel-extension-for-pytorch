import torch
from torch._C import parse_ir
import intel_extension_for_pytorch
import torch.nn.functional as F
import time
from torch.testing._internal.common_utils import TestCase
torch.set_printoptions(profile="full")

class TestFusionMethod(TestCase):
    def test_interaction_fusion(self):
        # Currently, the Interaction op only supports DLRM inference
        # in forward path on Terabyte Dataset
        # the inputs are in shapes of [1, 82768, 128]
        # and [26, 32768, 128] with Half datatype
        batch_size = 32768
        li_cpu = torch.tensor([i for i in range(27) for j in range(i)])
        lj_cpu = torch.tensor([j for i in range(27) for j in range(i)])

        x_mlp_cpu = torch.rand(1, batch_size, 128)
        x_emb_cpu = torch.rand(26, batch_size, 128)
        x_cpu = torch.cat([x_mlp_cpu, x_emb_cpu], dim=0)
        x_cpu = x_cpu.transpose(0, 1).contiguous()
        b = x_cpu.clone().transpose(1, 2).contiguous()
        out_bmm = torch.bmm(x_cpu, b)
        out_index = out_bmm[:, li_cpu, lj_cpu]
        out_cpu = torch.cat([x_mlp_cpu[0], out_index], dim=1)

        x_mlp_xpu = x_mlp_cpu.clone().half().xpu()
        x_emb_xpu = x_emb_cpu.clone().half().xpu()
        out_xpu = torch.xpu.intrinsic.Interaction(x_mlp_xpu, x_emb_xpu).float()

        self.assertEqual(out_cpu, out_xpu.cpu(), atol=5 * 1e-2, rtol=0)
