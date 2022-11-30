import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_dtype import complex_types
devices = (torch.device('cpu'), torch.device('xpu:0'))
from common.pytorch_test_base import TestCase, dtypesIfXPU, TEST_XPU, TEST_MULTIGPU, largeTensorTest

class TestComplexTensor(TestCase):

    @dtypes(*complex_types())
    def test_to_list(self, device, dtype):
        self.assertEqual(torch.zeros((2, 2), device=device, dtype=dtype).tolist(), [[0j, 0j], [0j, 0j]])

    @dtypes(torch.float32, torch.float64)
    def test_dtype_inference(self, device, dtype):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        x = torch.tensor([3.0, 3.0 + 5j], device=device)
        torch.set_default_dtype(default_dtype)
        self.assertEqual(x.dtype, torch.cdouble if dtype == torch.float64 else torch.cfloat)
instantiate_device_type_tests(TestComplexTensor, globals())
if __name__ == '__main__':
    run_tests()