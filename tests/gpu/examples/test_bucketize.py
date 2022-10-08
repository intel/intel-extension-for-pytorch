import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

class TestNNMethod(TestCase):
    def test_searchsorted(self):
        dtypes = [torch.float32, torch.float64, torch.bfloat16]
        for dtype in dtypes:
            sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
            values = torch.tensor([[3, 6, 9], [3, 6, 9]])
            searchsorted_result_cpu = torch.searchsorted(sorted_sequence, values)
            searchsorted_result_xpu = torch.searchsorted(sorted_sequence.to('xpu').to(dtype), values.to('xpu').to(dtype))
            self.assertEqual(searchsorted_result_cpu, searchsorted_result_xpu)
            print('searchsorted_result_cpu', searchsorted_result_cpu)
            print('searchsorted_result_xpu', searchsorted_result_xpu)

            boundaries = torch.tensor([1, 3, 5, 7, 9])
            v = torch.tensor([[3, 6, 9], [3, 6, 9]])
            bucketize_result_cpu = torch.bucketize(v, boundaries)
            bucketize_result_xpu = torch.bucketize(v.to('xpu').to(dtype), boundaries.to('xpu').to(dtype))
            self.assertEqual(bucketize_result_cpu, bucketize_result_xpu)
            print('bucketize_result_cpu', bucketize_result_cpu)
            print('bucketize_result_xpu', bucketize_result_xpu)

            # scalar type
            x = torch.tensor([1.5, 2.5, 3.5])
            y = torch.tensor(2)
            searchsorted_result_cpu = torch.searchsorted(x, y)
            searchsorted_result_xpu = torch.searchsorted(x.to('xpu').to(dtype), y.to('xpu').to(dtype))
            self.assertEqual(searchsorted_result_cpu, searchsorted_result_xpu)
            print('searchsorted_result_cpu', searchsorted_result_cpu)
            print('searchsorted_result_xpu', searchsorted_result_xpu)

            bucketize_result_cpu = torch.bucketize(y, x)
            bucketize_result_xpu = torch.bucketize(y.to('xpu').to(dtype), x.to('xpu').to(dtype))
            print('bucketize_result_cpu', bucketize_result_cpu)
            print('bucketize_result_xpu', bucketize_result_xpu)
