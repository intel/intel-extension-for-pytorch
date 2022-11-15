import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_gather(self):
        seqlen = torch.tensor(4096)
        total_hashes = 8

        buckets = torch.randn(8, 32768).to(torch.int64)
        buckets_xpu = buckets.to(xpu_device)

        ticker = torch.arange(total_hashes * seqlen, device=cpu_device).unsqueeze(0).expand_as(buckets)
        ticker_xpu = torch.arange(total_hashes * seqlen, device=xpu_device).unsqueeze(0).expand_as(buckets_xpu)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t_xpu = seqlen * buckets_xpu + (ticker_xpu % seqlen)
        buckets_and_t = buckets_and_t.detach()
        buckets_and_t_xpu = buckets_and_t_xpu.detach()

        def sort_key_val(t1, t2, dim=-1):
            values, indices = t1.sort(dim=dim, stable=True)
            t2 = t2.expand_as(t1)
            return values, t2.gather(dim, indices)

        def sort_key_val_half(t1, t2, dim=-1):
            values, indices = t1.sort(dim=dim, stable=True)
            t2 = t2.expand_as(t1)
            t2 = t2.half()
            return values, t2.gather(dim, indices)

        output1, output2 = sort_key_val(buckets_and_t, ticker)
        output1_xpu, output2_xpu = sort_key_val(buckets_and_t_xpu, ticker_xpu)
        output1_h, output2_h = sort_key_val_half(buckets_and_t, ticker)
        output1_h_xpu, output2_h_xpu = sort_key_val_half(buckets_and_t_xpu, ticker_xpu)
        self.assertEqual(buckets_and_t, buckets_and_t_xpu.cpu())
        self.assertEqual(ticker, ticker_xpu.cpu())
        self.assertEqual(output1_h, output1_h_xpu.cpu())
        self.assertEqual(output2_h, output2_h_xpu.cpu())
        self.assertEqual(output1, output1_xpu.cpu())
        self.assertEqual(output2, output2_xpu.cpu())
