import torch
import intel_extension_for_pytorch # noqa
import scipy
from functools import partial
from torch.testing._internal.common_utils import TestCase
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_int_dtypes

class TestTorchMethod(TestCase):
    def test_xlogy_xlog1py(self, device="xpu", dtypes=(torch.float32, torch.float32)):
        x_dtype, y_dtype = dtypes

        def out_variant_helper(torch_fn, x, y):
            expected = torch_fn(x, y)
            out = torch.empty_like(expected)
            torch_fn(x, y, out=out)
            self.assertEqual(expected, out)

        def xlogy_inplace_variant_helper(x, y):
            if x.dtype in get_all_int_dtypes() + [torch.bool]:
                with self.assertRaisesRegex(RuntimeError,
                                            "can't be cast to the desired output type"):
                    x.clone().xlogy_(y)
            else:
                expected = torch.empty_like(x)
                torch.xlogy(x, y, out=expected)
                inplace_out = x.clone().xlogy_(y)
                self.assertEqual(expected, inplace_out)

        def test_helper(torch_fn, reference_fn, inputs, scalar=None):
            x, y, z = inputs
            torch_fn_partial = partial(torch_fn, x)
            reference_fn_partial = partial(reference_fn, x.cpu().numpy())
            self.compare_with_numpy(torch_fn_partial, reference_fn_partial, x, exact_dtype=False)
            self.compare_with_numpy(torch_fn_partial, reference_fn_partial, y, exact_dtype=False)
            self.compare_with_numpy(torch_fn_partial, reference_fn_partial, z, exact_dtype=False)

            val = scalar if scalar is not None else x
            out_variant_helper(torch_fn, val, x)
            out_variant_helper(torch_fn, val, y)
            out_variant_helper(torch_fn, val, z)

        # Tensor-Tensor Test (tensor of same and different shape)
        x = make_tensor((3, 2, 4, 5), device=device, dtype=x_dtype, low=0.5, high=1000)
        y = make_tensor((3, 2, 4, 5), device=device, dtype=y_dtype, low=0.5, high=1000)
        z = make_tensor((4, 5), device=device, dtype=y_dtype, low=0.5, high=1000)

        x_1p = make_tensor((3, 2, 4, 5), device=device, dtype=x_dtype, low=-0.5, high=1000)
        y_1p = make_tensor((3, 2, 4, 5), device=device, dtype=y_dtype, low=-0.5, high=1000)
        z_1p = make_tensor((4, 5), device=device, dtype=y_dtype, low=-0.5, high=1000)

        xlogy_fns = torch.xlogy, scipy.special.xlogy
        xlog1py_fns = torch.special.xlog1py, scipy.special.xlog1py
#
        test_helper(*xlogy_fns, (x, y, z))
        xlogy_inplace_variant_helper(x, x)
        xlogy_inplace_variant_helper(x, y)
        xlogy_inplace_variant_helper(x, z)
        test_helper(*xlog1py_fns, (x_1p, y_1p, z_1p))

        # Scalar-Tensor Test
        test_helper(*xlogy_fns, (x, y, z), 3.14)
        test_helper(*xlog1py_fns, (x_1p, y_1p, z_1p), 3.14)

        # Special Values Tensor-Tensor
        t = torch.tensor([-1., 0., 1., 2., float('inf'), -float('inf'), float('nan')], device=device)
        zeros = torch.zeros(7, dtype=y_dtype, device=device)

        def test_zeros_special_helper(torch_fn, reference_fn, scalar=False):
            zeros_t = 0 if scalar else zeros
            zeros_np = 0 if scalar else zeros.cpu().numpy()
            torch_fn_partial = partial(torch_fn, zeros_t)
            reference_fn_partial = partial(reference_fn, zeros_np)
            self.compare_with_numpy(torch_fn_partial, reference_fn_partial, t, exact_dtype=False)
            out_variant_helper(torch_fn, zeros_t, t)

        test_zeros_special_helper(*xlogy_fns)
        xlogy_inplace_variant_helper(zeros, t)
        test_zeros_special_helper(*xlog1py_fns)

        # Special Values Scalar-Tensor
        test_zeros_special_helper(*xlogy_fns, scalar=True)
        test_zeros_special_helper(*xlog1py_fns, scalar=True)
