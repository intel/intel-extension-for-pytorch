import torch
from torch.testing._internal.common_utils import TestCase
import random
from itertools import permutations, product

import intel_extension_for_pytorch  # noqa

SIZE = 100


class dtypes:
    def __init__(self, *args, device='xpu'):
        self.args = args
        self.device = device

    def __call__(self, fn):
        def new_fn(*args, **kwargs):
            for dtype in self.args:
                kwargs['dtype'] = dtype
                kwargs['device'] = self.device
                fn(*args, **kwargs)
        return new_fn


class TestNNMethod(TestCase):
    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = x.size(1)
        if order == 'descending':
            def check_order(a, b):
                # `a != a` because we put NaNs
                # at the end of ascending sorted lists,
                # and the beginning of descending ones.
                return ((a != a) | (a >= b)).all().item()
        elif order == 'ascending':
            def check_order(a, b):
                # see above
                return ((b != b) | (a <= b)).all().item()
        else:
            error('unknown order "{}", must be "ascending" or "descending"'.format(order))

        are_ordered = True
        for k in range(1, SIZE):
            self.assertTrue(check_order(mxx[:, k - 1], mxx[:, k]),
                            'torch.sort ({}) values unordered for {}'.format(order, task))

        seen = set()
        indicesCorrect = True
        size0 = x.size(0)
        size = x.size(x.dim() - 1)
        x = x.tolist()
        mxx = mxx.tolist()
        ixx = ixx.tolist()
        for k in range(size0):
            seen.clear()
            for j in range(size):
                self.assertEqual(x[k][ixx[k][j]], mxx[k][j],
                                 msg='torch.sort ({}) indices wrong for {}'.format(order, task))
                seen.add(ixx[k][j])
            self.assertEqual(len(seen), size)

    def test_sort_base(self, device='xpu'):
        for SIZE in (4, 2049):
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x)

            # Test inplace
            y = x.clone()
            y_inds = torch.tensor((), dtype=torch.int64, device=device)
            torch.sort(y, out=(y, y_inds))
            x_vals, x_inds = torch.sort(x)
            self.assertEqual(x_vals, y)
            self.assertEqual(x_inds, y_inds)

            # Test use of result tensor
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)

            # Test sorting of random numbers
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random')

            # Test simple sort
            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0, rtol=0
            )

            # Test that we still have proper sorting with duplicate keys
            x = torch.floor(torch.rand(4, SIZE, device=device) * 10)
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with duplicate keys')

            # DESCENDING SORT
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x, x.dim() - 1, True)

            # Test use of result tensor
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x, x.dim() - 1, True), res1ind)
            self.assertEqual(x.argsort(x.dim() - 1, True), res1ind)

            # Test sorting of random numbers
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random')

            # Test simple sort task
            self.assertEqual(
                torch.sort(torch.tensor((10, 20, 30, 40, 50), device=device), 0, True)[0],
                torch.tensor((50, 40, 30, 20, 10), device=device),
                atol=0, rtol=0
            )

            # Test that we still have proper sorting with duplicate keys
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random with duplicate keys')

            # Test argument sorting with and without stable
            x = torch.tensor([1, 10, 2, 2, 3, 7, 7, 8, 9, 9] * 3)
            self.assertEqual(torch.argsort(x, stable=True), torch.sort(x, stable=True).indices)
            self.assertEqual(torch.argsort(x, stable=False), torch.sort(x, stable=False).indices)
            self.assertEqual(torch.argsort(x), torch.sort(x).indices)

            # Test sorting with NaNs
            x = torch.rand(4, SIZE, device=device)
            x[1][2] = float('NaN')
            x[3][0] = float('NaN')
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered('ascending', x, res2val, res2ind,
                                 'random with NaNs')
            torch.sort(x, out=(res2val, res2ind), descending=True)
            self.assertIsOrdered('descending', x, res2val, res2ind,
                                 'random with NaNs')

    def test_sort_simple(self):
        for SIZE in (4, 2049):
            device = 'xpu'
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x)

            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)

            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0, rtol=0
            )

    def test_sort_large_slice(self, device='xpu'):
        # tests direct cub path
        x = torch.randn(4, 1024000, device=device)
        res1val, res1ind = torch.sort(x, stable=True)
        torch.xpu.synchronize()
        # assertIsOrdered is too slow, so just compare to cpu
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())
        res1val, res1ind = torch.sort(x, descending=True, stable=True)
        torch.xpu.synchronize()
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), descending=True, stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())

    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double, torch.int, torch.long, torch.int8, torch.uint8)
    def test_stable_sort(self, device, dtype):
        sizes = (100, 1000, 10000)
        for ncopies in sizes:
            x = torch.tensor([0, 1] * ncopies, dtype=dtype, device=device)
            _, idx = x.sort(stable=True)
            self.assertEqual(
                idx[:ncopies],
                torch.arange(start=0, end=2 * ncopies, step=2, device=device)
            )
            self.assertEqual(
                idx[ncopies:],
                torch.arange(start=1, end=2 * ncopies, step=2, device=device)
            )

    @dtypes(torch.float32)
    def test_sort_restride(self, device, dtype):
        # Input: non-contiguous (stride: 5) 3-element array
        tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]
        # Outputs: 0-dim tensors
        # They will need to be resized, which means they will also be
        # restrided with the input tensor's strides as base.
        values = torch.tensor(0, dtype=dtype, device=device)
        indices = torch.tensor(0, dtype=torch.long, device=device)
        torch.sort(tensor, out=(values, indices))
        # Check: outputs were restrided to dense strides
        self.assertEqual(values.stride(), (1,))
        self.assertEqual(indices.stride(), (1,))
        # Check: 'tensor'  indexed by 'indices' is equal to 'values'
        self.assertEqual(tensor[indices], values)

    @dtypes(torch.float32)
    def test_sort_discontiguous(self, device, dtype):
        sizes = (5, 7, 2049)
        for shape in permutations(sizes):
            for perm in permutations((0, 1, 2)):
                for dim in range(3):
                    t = torch.randn(shape, device=device, dtype=dtype).permute(perm)
                    r1 = t.sort(dim=dim)
                    r2 = t.contiguous().sort(dim=dim)
                    self.assertEqual(r1, r2)
                    n = t.size(dim)

                    # assert ordered
                    self.assertTrue((r1.values.narrow(dim, 1, n - 1) >= r1.values.narrow(dim, 0, n - 1)).all())

                    # assert that different segments does not mix, which can easily happen
                    # if the stride is not handled correctly
                    self.assertTrue((t.unsqueeze(-1).transpose(dim, -1) ==
                                    r1.values.unsqueeze(-1)).any(dim=dim).any(dim=-1).all())

                    self.assertEqual(r1.values.stride(), t.stride())
                    self.assertEqual(r1.indices.stride(), t.stride())

    @dtypes(torch.float32, torch.half)
    def test_sort_1d_output_discontiguous(self, device, dtype):
        tensor = torch.randn(12, device=device, dtype=dtype)[:6]
        values = torch.empty_like(tensor)[::2]
        indices = torch.empty(18, device=device, dtype=torch.long)[::3]
        torch.sort(tensor, out=(values, indices))
        values_cont, indices_cont = tensor.sort()
        self.assertEqual(indices, indices_cont)
        self.assertEqual(values, values_cont)

    @dtypes(torch.float32)
    def test_topk_1d_output_discontiguous(self, device, dtype):
        tensor = torch.randn(12, device=device, dtype=dtype)
        values = torch.empty_like(tensor)[::2]
        indices = torch.empty(18, device=device, dtype=torch.long)[::3]
        for sorted in (True, False):
            # outputs of `sorted=False` test are not guaranteed to be the same,
            # but with current implementation they are
            torch.topk(tensor, 6, sorted=sorted, out=(values, indices))
            values_cont, indices_cont = tensor.topk(6, sorted=sorted)
            self.assertEqual(indices, indices_cont)
            self.assertEqual(values, values_cont)

    @dtypes(torch.float)
    def test_sort_expanded_tensor(self, device, dtype):
        # https://github.com/pytorch/pytorch/issues/91420
        data = torch.scalar_tensor(True, device=device, dtype=dtype)
        data = data.expand([1, 1, 1])
        ref = torch.Tensor([[[True]]])
        out = torch.sort(data, stable=True, dim=1, descending=True)
        expected = torch.sort(ref, stable=True, dim=1, descending=True)
        self.assertEqual(out, expected)

        data = torch.randn(4, 1, 10, device=device, dtype=dtype)
        data = data.expand([4, 8, 10])
        ref = data.contiguous()
        out = torch.sort(data, stable=True, dim=1, descending=True)
        expected = torch.sort(ref, stable=True, dim=1, descending=True)
        self.assertEqual(out, expected)

    def test_topk_simple(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('xpu')
            for largest_val in [True, False]:
                if (type(shape) == tuple):
                    for curr_dim in range(0, len(shape)):
                        dim_size = shape[curr_dim]
                        for k in range(0, dim_size + 1):
                            # print(x.shape, k, curr_dim, largest_val)
                            topk_values, topk_indices = torch.topk(x, k, dim=curr_dim, largest=largest_val, sorted=True)
                            # print(topk_values)
                            topk_values_cpu, topk_indices_cpu = torch.topk(
                                cpu_x, k, dim=curr_dim, largest=largest_val, sorted=True)
                            # print(topk_values_cpu)
                            self.assertEqual(topk_values, topk_values_cpu)
                            self.assertEqual(topk_indices, topk_indices_cpu)
                else:
                    for k in range(1, shape):
                        topk_values, topk_indices = torch.topk(x, k, dim=0, largest=largest_val)
                        topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=0, largest=largest_val)
                        self.assertEqual(topk_values, topk_values_cpu)
                        self.assertEqual(topk_indices, topk_indices_cpu)
        helper(2)
        helper((5, 1))
        helper((1, 5))
        helper((2, 3, 4, 5))
        helper((5, 9, 7, 4))
        helper((50, 20, 7, 4))

    def test_topk_base(self, device='xpu'):
        def topKViaSort(t, k, dim, dir):
            sorted, indices = t.sort(dim, dir)
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # Values should be exactly equivalent
            self.assertEqual(res1, res2, atol=0, rtol=0)

            # Indices might differ based on the implementation, since there is
            # no guarantee of the relative order of selection
            if not ind1.eq(ind2).all():
                # To verify that the indices represent equivalent elements,
                # gather from the input using the topk indices and compare against
                # the sort indices
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, atol=0, rtol=0)

        def compare(t, k, dim, dir):
            topKVal, topKInd = t.topk(k, dim, dir, True)
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        t = torch.rand(random.randint(1, SIZE),
                       random.randint(1, SIZE),
                       random.randint(1, SIZE), device=device)

        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

        t = torch.randn((2, 100000), device=device)
        compare(t, 2000, 1, True)
        compare(t, 2000, 1, False)

        t = torch.randn((2, 10000), device=device)
        compare(t, 2000, 1, True)
        compare(t, 2000, 1, False)

    def test_topk_arguments(self, device='xpu'):
        q = torch.randn(10, 2, 10, device=device)
        # Make sure True isn't mistakenly taken as the 2nd dimension (interpreted as 1)
        self.assertRaises(TypeError, lambda: q.topk(4, True))

    def test_topk_noncontiguous_gpu(self, device='xpu'):
        single_block_t = torch.randn(20, device=device)[::2]
        multi_block_t = torch.randn(20000, device=device)[::2]
        sort_t = torch.randn(200000, device=device)[::2]
        for t in (single_block_t, multi_block_t, sort_t):
            for k in (5, 2000, 10000):
                if k >= t.shape[0]:
                    continue
                top1, idx1 = t.topk(k)
                top2, idx2 = t.contiguous().topk(k)
                self.assertEqual(top1, top2)
                self.assertEqual(idx1, idx2)

    def _test_topk_dtype(self, device, dtype, integral, size):
        if integral:
            a = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max,
                              size=(size,), dtype=dtype, device=device)
        else:
            a = torch.randn(size=(size,), dtype=dtype, device=device)

        sort_topk = a.sort()[0][-(size // 2):].flip(0)
        topk = a.topk(size // 2)
        self.assertEqual(sort_topk, topk[0])      # check values
        self.assertEqual(sort_topk, a[topk[1]])   # check indices

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_topk_integral(self, device, dtype):
        small = 10
        large = 4096
        verylarge = 8192  # multi group
        for curr_size in (small, large, verylarge):
            self._test_topk_dtype(device, dtype, True, curr_size)

    @dtypes(torch.bfloat16)
    def test_topk_bfloat16(self, device, dtype):
        small = 10
        large = 4096
        verylarge = 8192  # multi group
        for curr_size in (small, large, verylarge):
            self._test_topk_dtype(device, dtype, False, curr_size)

    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_topk_nonfinite(self, device, dtype):
        x = torch.tensor([float('nan'), float('inf'), 1e4, 0, -1e4, -float('inf')], device=device, dtype=dtype)
        val, idx = x.topk(4)
        expect = torch.tensor([float('nan'), float('inf'), 1e4, 0], device=device, dtype=dtype)
        self.assertEqual(val, expect)
        self.assertEqual(idx, [0, 1, 2, 3])

        val, idx = x.topk(4, largest=False)
        expect = torch.tensor([-float('inf'), -1e4, 0, 1e4], device=device, dtype=dtype)
        self.assertEqual(val, expect)
        self.assertEqual(idx, [5, 4, 3, 2])

    def test_topk_4d(self, device='xpu'):
        small = 128
        large = 8192
        for size in (small, large):
            x = torch.ones(2, size, 2, 2, device=device)
            x[:, 1, :, :] *= 2.
            x[:, 10, :, :] *= 1.5
            val, ind = torch.topk(x, k=2, dim=1)
            expected_ind = torch.ones(2, 2, 2, 2, dtype=torch.long, device=device)
            expected_ind[:, 1, :, :] = 10
            expected_val = torch.ones(2, 2, 2, 2, device=device)
            expected_val[:, 0, :, :] *= 2.
            expected_val[:, 1, :, :] *= 1.5
            self.assertEqual(val, expected_val, atol=0, rtol=0)
            self.assertEqual(ind, expected_ind, atol=0, rtol=0)

    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double, torch.int, torch.long, torch.int8, torch.uint8)
    def test_topk_zero(self, device, dtype):
        # https://github.com/pytorch/pytorch/issues/49205
        t = torch.rand(2, 2, device=device).to(dtype=dtype)
        val, idx = torch.topk(t, k=0, largest=False)
        self.assertEqual(val.size(), torch.Size([2, 0]))
        self.assertEqual(idx.size(), torch.Size([2, 0]))
