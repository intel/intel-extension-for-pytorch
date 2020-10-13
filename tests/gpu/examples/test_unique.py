import torch
import torch_ipex
from torch.testing._internal.common_utils import TestCase
import pytest

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

class  TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch_ipex._usm_pstl_is_enabled()")
    def test_activation(self):
        output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
        output_dpcpp = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long, device=sycl_device))
        print(output_dpcpp)
        #tensor([ 2,  3,  1])
        self.assertEqual(output, output_dpcpp.cpu())

        output, inverse_indices = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
        output_dpcpp, inverse_indices_dpcpp = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long, device=sycl_device), sorted=True, return_inverse=True)
        print(output_dpcpp)
        #tensor([ 1,  2,  3])
        print(inverse_indices_dpcpp)
        #tensor([ 0,  2,  1,  2])
        self.assertEqual(output, output_dpcpp.cpu())
        self.assertEqual(inverse_indices, inverse_indices_dpcpp.cpu())

        output, inverse_indices, counts = torch.unique(torch.tensor([[5, 6], [2, 3]], dtype=torch.long), dim=0, sorted=True, return_inverse=True, return_counts=True)
        output_dpcpp, inverse_indices_dpcpp, counts_dpcpp = torch.unique(torch.tensor([[5, 6], [2, 3]], dtype=torch.long, device=sycl_device), dim=0, sorted=True, return_inverse=True, return_counts=True)
        self.assertEqual(output, output_dpcpp.cpu())
        self.assertEqual(inverse_indices, inverse_indices_dpcpp.cpu())
        self.assertEqual(counts, counts_dpcpp.cpu())
        
        output, inverse_indices, counts = torch.unique_consecutive(torch.tensor([[1, 3], [2, 3]], dtype=torch.long), return_inverse=True, return_counts=True)
        output_dpcpp, inverse_indices_dpcpp, counts_dpcpp = torch.unique_consecutive(torch.tensor([[1, 3], [2, 3]], dtype=torch.long, device=sycl_device), return_inverse=True, return_counts=True)
        print(output_dpcpp)
        #tensor([ 1,  2,  3])
        print(inverse_indices_dpcpp)
        #tensor([[ 0,  2], [ 1,  2]])
        self.assertEqual(output, output_dpcpp.cpu())
        self.assertEqual(inverse_indices, inverse_indices_dpcpp.cpu())
        self.assertEqual(counts, counts_dpcpp.cpu())

