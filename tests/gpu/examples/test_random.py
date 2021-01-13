import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_random(self):
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        torch.set_default_dtype(torch.double)
        t = torch.empty(200, device=sycl_device)
        lb = 1
        ub = 4

        t = t.fill_(-1)
        t = t.random_(lb, ub)
        #self.assertEqual(t.min(), lb)
        #self.assertEqual(t.max(), ub - 1)

        t = t.fill_(-1)
        t = t.random_(ub)
        #self.assertEqual(t.min(), 0)
        #self.assertEqual(t.max(), ub - 1)

    def test_random_bool(self):
        size = 2000
        t = torch.empty(size, dtype=torch.bool, device=sycl_device)

        t.fill_(False)
        t.random_()
        #self.assertEqual(t.min(), False)
        #self.assertEqual(t.max(), True)
        #self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

        t.fill_(True)
        t.random_()
        #self.assertEqual(t.min(), False)
        #self.assertEqual(t.max(), True)
        #self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

    def test_random_from_to_bool(self):
        size = 2000

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        min_val = 0
        max_val = 1

        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=torch.bool, device=sycl_device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val) or not (min_val <= (to_ - 1) <= max_val):
                        if not (min_val <= from_ <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "from is out of bounds"
                            )
                        if not (min_val <= (to_ - 1) <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "to - 1 is out of bounds"
                            )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        #self.assertTrue(from_ <= t.to(torch.int).min() < (from_ + delta))
                        #self.assertTrue((to_ - delta) <= t.to(torch.int).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    def test_random_full_range(self):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        torch.set_default_dtype(torch.double)

        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        t = torch.empty(size, device=sycl_device)

        if dtype in [torch.float, torch.double, torch.half]:
            from_ = int(max(torch.finfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.finfo(dtype).max, int64_max_val))
        elif dtype == torch.bfloat16:
            from_ = int(max(-3.389531389251535e+38, int64_min_val))
            to_inc_ = int(min(3.389531389251535e+38, int64_max_val))
        else:
            from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        t.random_(from_, None)
        delta = max(1, alpha * range_)
        #self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
        #self.assertTrue((to_inc_ - delta) < t.to(torch.double).max() <= to_inc_)

    def test_random_from_to(self):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        torch.set_default_dtype(torch.double)

        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, device=sycl_device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val) or not (min_val <= (to_ - 1) <= max_val):
                        if not (min_val <= from_ <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "from is out of bounds"
                            )
                        if not (min_val <= (to_ - 1) <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "to - 1 is out of bounds"
                            )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = max(1, alpha * range_)
                        if dtype == torch.bfloat16:
                            # Less strict checks because of rounding errors
                            # TODO investigate rounding errors
                            #self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                            #self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                        else:
                            #self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                            #self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)

    def test_random_to(self):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        torch.set_default_dtype(torch.double)

        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        from_ = 0
        for to_ in tos:
            t = torch.empty(size, device=sycl_device)
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertWarnsRegex(
                        lambda: t.random_(to_),
                        "to - 1 is out of bounds"
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    if dtype == torch.bfloat16:
                        # Less strict checks because of rounding errors
                        # TODO investigate rounding errors
                        #self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        #self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                    else:
                        #self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        #self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    def test_random_default(self):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        torch.set_default_dtype(torch.double)

        size = 2000
        alpha = 0.1

        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        elif dtype == torch.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, device=sycl_device)
        t.random_()
        #self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        #self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)

