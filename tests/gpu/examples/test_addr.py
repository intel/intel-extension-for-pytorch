import torch
import intel_extension_for_pytorch # noqa
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_dtype import get_all_fp_dtypes, get_all_complex_dtypes
import numpy as np

class TestTorchMethod(TestCase):

    def test_addr_vs_cpu(self, device="xpu", dtype=torch.float32, beta=1, alpha=1):
        cpu_device = "cpu"
        m = make_tensor((50, 50), device=cpu_device, dtype=dtype, low=-2, high=2)
        a = make_tensor((50,), device=cpu_device, dtype=dtype, low=-2, high=2)
        b = make_tensor((50,), device=cpu_device, dtype=dtype, low=-2, high=2)
        res_cpu = torch.addr(m, a, b, beta=1, alpha=1)

        m_xpu = m.to("xpu")
        a_xpu = a.to("xpu")
        b_xpu = b.to("xpu")
        res_xpu = torch.addr(m_xpu, a_xpu, b_xpu, beta=1, alpha=1)
        self.assertEqual(res_cpu, res_xpu.to("cpu"))


    def test_addr_vs_numpy(self, device="xpu", dtype=torch.float32, beta=1, alpha=1):
        def check(m, a, b, beta, alpha):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()
                exact_dtype = True
#            print("outer:", np.outer(a_np,b_np))
            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)

            res = torch.addr(m, a, b, beta=beta, alpha=alpha)
            self.assertEqual(res, expected, exact_dtype=exact_dtype)

            # Test out variant
            out = torch.empty_like(res)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected, exact_dtype=exact_dtype)

        m = make_tensor((50, 50), device=device, dtype=dtype, low=-2, high=2)
        a = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)
        b = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)

        check(m, a, b, beta, alpha)

        # test 0 strided tensor
        zero_strided = make_tensor((1,), device=device, dtype=dtype, low=-2, high=2).expand(50)
        check(m, zero_strided, b, beta, alpha)

        # test scalar
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        check(m_scalar, a, b, beta, alpha)

        # test nans and infs are not propagated to the output when beta == 0
        float_and_complex_dtypes = get_all_fp_dtypes() + get_all_complex_dtypes()
        if beta == 0 and dtype in float_and_complex_dtypes:
            m[0][10] = m[10][10] = m[20][20] = float('inf')
            m[1][10] = m[11][10] = m[21][20] = float('nan')
        check(m, a, b, 0, alpha)

    def test_addr_type_promotion(self, device="xpu", dtypes=torch.float32):
        dtypes = [torch.float64, torch.float32, torch.bfloat16]
        device = "xpu"
        a = make_tensor((50,), device=device, dtype=dtypes[0], low=-2, high=2)
        b = make_tensor((50,), device=device, dtype=dtypes[1], low=-2, high=2)
        m = make_tensor((50, 50), device=device, dtype=dtypes[2], low=-2, high=2)

        desired_dtype = torch.promote_types(torch.promote_types(dtypes[0], dtypes[1]),
                                            dtypes[2])
        for op in (torch.addr, torch.Tensor.addr):
            print(op)
            print("desired type:", desired_dtype)
            result = op(m, a, b)
            print("result type:", result.dtype)
            self.assertEqual(result.dtype, desired_dtype)
