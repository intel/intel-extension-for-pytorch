import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_segment_reduce_scatter_cases_and_backward(self, dtype=torch.float):       
        val_dtype = dtype
        length_dtype = int
        tests = [
            {
                'src': [1, 2, 3, 4, 5, 6],
                'index': [0, 0, 1, 1, 1, 3],
                'indptr': [0, 2, 5, 5, 6],
                'sum': [3, 12, 0, 6],
                'prod': [2, 60, 1, 6],
                'mean': [1.5, 4, float('nan'), 6],
                'min': [1, 3, float('inf'), 6],
                'max': [2, 5, -float('inf'), 6],
            },
            {
                'src': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                'index': [0, 0, 1, 1, 1, 3],
                'indptr': [0, 2, 5, 5, 6],
                'sum': [[4, 6], [21, 24], [0, 0], [11, 12]],
                'prod': [[3, 8], [315, 480], [1, 1], [11, 12]],
                'mean': [[2, 3], [7, 8], [float('nan'), float('nan')], [11, 12]],
                'min': [[1, 2], [5, 6], [float('inf'), float('inf')], [11, 12]],
                'max': [[3, 4], [9, 10], [-float('inf'), -float('inf')], [11, 12]],
            },
            {
                'src': [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
                'index': [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
                'indptr': [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
                'sum': [[4, 21, 0, 11], [12, 18, 12, 0]],
                'prod': [[3, 315, 1, 11], [48, 80, 12, 1]],
                'mean': [[2, 7, float('nan'), 11], [4, 9, 12, float('nan')]],
                'min': [[1, 5, float('inf'), 11], [2, 8, 12, float('inf')]],
                'max': [[3, 9, -float('inf'), 11], [6, 10, 12, -float('inf')]],
            },
            {
                'src': [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
                'index': [[0, 0, 1], [0, 2, 2]],
                'indptr': [[0, 2, 3, 3], [0, 1, 1, 3]],
                'sum': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
                'prod': [[[3, 8], [5, 6], [1, 1]], [[7, 9], [1, 1], [120, 143]]],
                'mean': [[[2, 3], [5, 6], [float('nan'), float('nan')]],
                         [[7, 9], [float('nan'), float('nan')], [11, 12]]],
                'min': [[[1, 2], [5, 6], [float('inf'), float('inf')]],
                        [[7, 9], [float('inf'), float('inf')], [10, 11]]],
                'max': [[[3, 4], [5, 6], [-float('inf'), -float('inf')]],
                        [[7, 9], [-float('inf'), -float('inf')], [12, 13]]],
            },
            {
                'src': [[1, 3], [2, 4]],
                'index': [[0, 0], [0, 0]],
                'indptr': [[0, 2], [0, 2]],
                'sum': [[4], [6]],
                'prod': [[3], [8]],
                'mean': [[2], [3]],
                'min': [[1], [2]],
                'max': [[3], [4]],
            },
            {
                'src': [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
                'index': [[0, 0], [0, 0]],
                'indptr': [[0, 2], [0, 2]],
                'sum': [[[4, 4]], [[6, 6]]],
                'prod': [[[3, 3]], [[8, 8]]],
                'mean': [[[2, 2]], [[3, 3]]],
                'min': [[[1, 1]], [[2, 2]]],
                'max': [[[3, 3]], [[4, 4]]],
            },
        ]
        for test in tests:
            reduces = ['sum', 'prod', 'min', 'max', 'mean']
            for reduce in reduces:
                data = torch.tensor(test['src'], dtype=val_dtype, device=dpcpp_device, requires_grad=True)
                data_cpu = torch.tensor(test['src'], dtype=val_dtype, device=cpu_device, requires_grad=True)
                indptr = torch.tensor(test['indptr'], dtype=length_dtype, device=dpcpp_device)
                indptr_cpu = indptr.cpu()
                dim = indptr.ndim - 1
                lengths = torch.diff(indptr, dim=dim)
                lengths_cpu = lengths.cpu()
                result = torch.segment_reduce(
                                data=data,
                                reduce=reduce,
                                lengths=lengths,
                                axis=dim,
                                unsafe=True,
                            )
                result_cpu = torch.segment_reduce(
                                data=data_cpu,
                                reduce=reduce,
                                lengths=lengths_cpu,
                                axis=dim,
                                unsafe=True,
                            )
                self.assertEqual(result.cpu(), result_cpu)
                # Test backward
                result.sum().backward()
                result_cpu.sum().backward()
                # print(data.grad)
                # print(data_cpu.grad)
                self.assertEqual(data.grad, data_cpu.grad)
