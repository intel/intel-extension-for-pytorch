import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cat_1d_array(self, dtype=torch.float):
        shapes = [(8, 7, 2), (4, 4, 4), (4, 4, 1), (4, 1, 4),
                  (4, 1, 1), (1, 4, 4), (1, 4, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, W = shape[0], shape[1], shape[2]
            user_cpu1 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)
            user_cpu2 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)
            user_cpu3 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.channels_last_1d)
            user_cpu2 = user_cpu2.to(memory_format=torch.channels_last_1d)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last_1d)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last_1d))

            user_xpu1 = user_cpu1.to(dpcpp_device)
            user_xpu2 = user_cpu2.to(dpcpp_device)
            user_xpu3 = user_cpu3.to(dpcpp_device)

            print("\n-------------GPU Result:--------------")
            res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_dpcpp.cpu().shape)
            print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d))
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            if 1 == res_dpcpp.shape[1] or 1 == res_dpcpp.shape[2] or \
               (1 == res_dpcpp.shape[1] and 1 == res_dpcpp.shape[2]):
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(res_dpcpp.is_contiguous(), False)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d), True)

            user_cpu1 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)
            user_cpu2 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)
            user_cpu3 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.channels_last_1d)
            user_cpu2 = user_cpu2.to(memory_format=torch.contiguous_format)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last_1d)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last_1d))

            user_xpu1 = user_cpu1.to(dpcpp_device)
            user_xpu2 = user_cpu2.to(dpcpp_device)
            user_xpu3 = user_cpu3.to(dpcpp_device)

            print("\n-------------GPU Result:--------------")
            res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_dpcpp.cpu().shape)
            print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d))
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            if 1 == res_dpcpp.shape[1] or 1 == res_dpcpp.shape[2] or \
               (1 == res_dpcpp.shape[1] and 1 == res_dpcpp.shape[2]):
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(res_dpcpp.is_contiguous(), False)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d), True)

            user_cpu1 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)
            user_cpu2 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)
            user_cpu3 = torch.randn([N, C, W], device=cpu_device, dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.contiguous_format)
            user_cpu2 = user_cpu2.to(memory_format=torch.channels_last_1d)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last_1d)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last_1d))

            user_xpu1 = user_cpu1.to(dpcpp_device)
            user_xpu2 = user_cpu2.to(dpcpp_device)
            user_xpu3 = user_cpu3.to(dpcpp_device)

            print("\n-------------GPU Result:--------------")
            res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_dpcpp.cpu().shape)
            print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d))
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            if 1 == res_dpcpp.shape[1] or 1 == res_dpcpp.shape[2] or \
               (1 == res_dpcpp.shape[1] and 1 == res_dpcpp.shape[2]):
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last_1d), False)

    def test_cat_array(self, dtype=torch.float):
        shapes = [(8, 7, 3, 2), (4, 4, 4, 4), (4, 4, 1, 1), (4, 1, 4, 4),
                  (4, 1, 4, 1), (4, 1, 1, 4), (1, 4, 1, 4), (1, 4, 4, 1),
                  (4, 1, 1, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            user_cpu1 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)
            user_cpu2 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)
            user_cpu3 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.channels_last)
            user_cpu2 = user_cpu2.to(memory_format=torch.channels_last)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last))

            user_xpu1 = user_cpu1.to(dpcpp_device)
            user_xpu2 = user_cpu2.to(dpcpp_device)
            user_xpu3 = user_cpu3.to(dpcpp_device)

            print("\n-------------GPU Result:--------------")
            res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_dpcpp.cpu().shape)
            print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            if 1 == res_dpcpp.shape[1] or (1 == res_dpcpp.shape[2] and 1 == res_dpcpp.shape[3]) or \
               (1 == res_dpcpp.shape[1] and 1 == res_dpcpp.shape[2] and 1 == res_dpcpp.shape[3]):
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(res_dpcpp.is_contiguous(), False)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last), True)

            user_cpu1 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)
            user_cpu2 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)
            user_cpu3 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.channels_last)
            user_cpu2 = user_cpu2.to(memory_format=torch.contiguous_format)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last))

            user_xpu1 = user_cpu1.to(dpcpp_device)
            user_xpu2 = user_cpu2.to(dpcpp_device)
            user_xpu3 = user_cpu3.to(dpcpp_device)

            print("\n-------------GPU Result:--------------")
            res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_dpcpp.cpu().shape)
            print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            if 1 == res_dpcpp.shape[1] or (1 == res_dpcpp.shape[2] and 1 == res_dpcpp.shape[3]) or \
               (1 == res_dpcpp.shape[1] and 1 == res_dpcpp.shape[2] and 1 == res_dpcpp.shape[3]):
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(res_dpcpp.is_contiguous(), False)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last), True)

            user_cpu1 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)
            user_cpu2 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)
            user_cpu3 = torch.randn([N, C, H, W], device=cpu_device, dtype=dtype)

            user_cpu1 = user_cpu1.to(memory_format=torch.contiguous_format)
            user_cpu2 = user_cpu2.to(memory_format=torch.channels_last)
            user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

            dim_idx = 1
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            print("\n-------------CPU Result:--------------")
            print(res_cpu.shape)
            print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last))

            user_xpu1 = user_cpu1.to(dpcpp_device)
            user_xpu2 = user_cpu2.to(dpcpp_device)
            user_xpu3 = user_cpu3.to(dpcpp_device)

            print("\n-------------GPU Result:--------------")
            res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
            print("SYCL Result:")
            print(res_dpcpp.cpu().shape)
            print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            if 1 == res_dpcpp.shape[1] or (1 == res_dpcpp.shape[2] and 1 == res_dpcpp.shape[3]) or \
               (1 == res_dpcpp.shape[1] and 1 == res_dpcpp.shape[2] and 1 == res_dpcpp.shape[3]):
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(res_dpcpp.is_contiguous(), True)
                self.assertEqual(res_dpcpp.is_contiguous(memory_format=torch.channels_last), False)
