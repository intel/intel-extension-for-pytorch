import torch
import intel_extension_for_pytorch  # noqa
import random

from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase

dpcpp_device = torch.device("xpu")


class TestScatterGather(TestCase):
    # Fills an index tensor with valid indices
    def _fill_indices(
        self, idx, dim, dim_size, elems_per_row, m, n, o, unique_indices=True
    ):
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    if unique_indices:
                        idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]
                    else:
                        idx[tuple(ii)] = torch.randint(dim_size, (elems_per_row,))

    def _test_scatter_base(
        self,
        fn,
        *,
        device,
        dtype,
        is_scalar,
        reduction,
        unique_indices=True,
        include_self=True
    ):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.empty(tuple(idx_size), device=device, dtype=torch.long)
        self._fill_indices(
            idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o, unique_indices
        )

        if is_scalar:
            src = random.random()
        else:
            src_size = [random.randint(1, 5) + s for s in idx_size]
            src = make_tensor(tuple(src_size), device=device, dtype=dtype)

        base = make_tensor((m, n, o), device=device, dtype=dtype)
        if reduction is not None:
            if fn is torch.Tensor.scatter_reduce_:
                actual = fn(
                    base.clone(),
                    dim,
                    idx,
                    src,
                    reduce=reduction,
                    include_self=include_self,
                )
            else:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction)
        else:
            actual = fn(base.clone(), dim, idx, src)

        expected = base.clone()
        counts = torch.zeros(base.shape, dtype=torch.long, device=device) + include_self
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    if fn is torch.Tensor.scatter_add_:
                        expected[tuple(ii)] += src[i, j, k]
                    else:
                        # method may be 'scatter_', 'scatter', 'scatter_reduce'
                        # or 'scatter_reduce_', the former two might have a reduction argument
                        # while the latter two always do
                        value = src if is_scalar else src[i, j, k]

                        if (not include_self) and counts[tuple(ii)] == 0:
                            expected[tuple(ii)] = value
                        else:
                            if reduction == "add" or reduction == "sum":
                                expected[tuple(ii)] += value
                            elif reduction == "multiply" or reduction == "prod":
                                expected[tuple(ii)] *= value
                            elif reduction == "amax":
                                expected[tuple(ii)] = max(expected[tuple(ii)], value)
                            elif reduction == "amin":
                                expected[tuple(ii)] = min(expected[tuple(ii)], value)
                            elif reduction == "mean":
                                expected[tuple(ii)] += value
                            else:
                                expected[tuple(ii)] = value

                        counts[tuple(ii)] += 1

        if reduction == "mean":
            counts.masked_fill_(counts == 0, 1)
            if dtype.is_floating_point or dtype.is_complex:
                expected /= counts
            else:
                expected.div_(counts, rounding_mode="floor")

        self.assertEqual(actual, expected, atol=0, rtol=0)

        # Tests empty index
        dst = make_tensor((2, 2), device=device, dtype=dtype)
        idx = torch.tensor((), device=device, dtype=torch.long)
        src = make_tensor((2, 2), device=device, dtype=dtype)
        if reduction is not None:
            actual = fn(dst, 0, idx, src, reduce=reduction)
        else:
            actual = fn(dst, 0, idx, src)
        self.assertEqual(actual, dst, atol=0, rtol=0)

    # reduction = sum
    def test_scatter_reduce_sum(self, device="xpu", dtype=torch.float32):
        for include_self in (True, False):
            self._test_scatter_base(
                torch.Tensor.scatter_reduce_,
                device=device,
                dtype=dtype,
                is_scalar=False,
                reduction="sum",
                unique_indices=False,
                include_self=include_self,
            )

    # reduction = prod
    def test_scatter_reduce_prod(self, device="xpu", dtype=torch.float32):
        for include_self in (True, False):
            self._test_scatter_base(
                torch.Tensor.scatter_reduce_,
                device=device,
                dtype=dtype,
                is_scalar=False,
                reduction="prod",
                unique_indices=False,
                include_self=include_self,
            )

    # reduction = mean
    def test_scatter_reduce_mean(self, device="xpu", dtype=torch.float32):
        for include_self in (True, False):
            self._test_scatter_base(
                torch.Tensor.scatter_reduce_,
                device=device,
                dtype=dtype,
                is_scalar=False,
                reduction="mean",
                unique_indices=False,
                include_self=include_self,
            )

    # reduction = max
    def test_scatter_reduce_amax(self, device="xpu", dtype=torch.float32):
        for include_self in (True, False):
            self._test_scatter_base(
                torch.Tensor.scatter_reduce_,
                device=device,
                dtype=dtype,
                is_scalar=False,
                reduction="amax",
                unique_indices=False,
                include_self=include_self,
            )
            # simple test for nan/inf propagation
            if dtype.is_floating_point:
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor(
                    [1, float("nan"), -float("inf"), -float("inf"), 2, float("inf")],
                    device=device,
                    dtype=dtype,
                )
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, "amax", include_self=include_self)
                expected_result = torch.tensor(
                    [float("nan"), -float("inf"), float("inf")],
                    device=device,
                    dtype=dtype,
                )
                if include_self:
                    expected_result[1] = 0
                self.assertEqual(input, expected_result)

    # reduction = min
    def test_scatter_reduce_amin(self, device="xpu", dtype=torch.float32):
        for include_self in (True, False):
            self._test_scatter_base(
                torch.Tensor.scatter_reduce_,
                device=device,
                dtype=dtype,
                is_scalar=False,
                reduction="amin",
                unique_indices=False,
                include_self=include_self,
            )
            # simple test for nan/inf propagation
            if dtype.is_floating_point:
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor(
                    [1, float("nan"), -2, -float("inf"), float("inf"), float("inf")],
                    device=device,
                    dtype=dtype,
                )
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, "amin", include_self=include_self)
                expected_result = torch.tensor(
                    [float("nan"), -float("inf"), float("inf")],
                    device=device,
                    dtype=dtype,
                )
                if include_self:
                    expected_result[2] = 0
                self.assertEqual(input, expected_result)
