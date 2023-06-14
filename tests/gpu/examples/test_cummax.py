import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
torch.set_printoptions(precision=4)

test_1d_shapes = [4, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65]
test_2d_shapes = [
    (4, 31),
    (1, 7),
    (7, 1),
    (2, 7),
    (2, 8),
    (2, 9),
    (7, 1),
    (7, 2),
    (8, 2),
    (9, 2),
    (3, 15),
    (3, 16),
    (3, 17),
    (15, 3),
    (16, 3),
    (17, 3),
    (4, 31),
    (4, 32),
    (4, 33),
    (31, 4),
    (32, 4),
    (33, 4),
    (65, 65),
    (8193, 65),
]
test_3d_shapes = [
    (7, 1, 7),
    (1, 7, 7),
    (7, 7, 1),
    (7, 1, 7),
    (2, 7, 7),
    (2, 8, 8),
    (2, 9, 9),
    (7, 7, 1),
    (7, 7, 2),
    (8, 8, 2),
    (9, 9, 2),
    (3, 15, 15),
    (15, 3, 15),
    (3, 16, 16),
    (3, 17, 17),
    (15, 15, 3),
    (16, 16, 3),
    (17, 17, 3),
    (4, 31, 31),
    (31, 4, 31),
    (4, 32, 32),
    (4, 33, 33),
    (31, 31, 4),
    (32, 32, 4),
    (33, 33, 4),
    (33, 65, 65),
    (87, 7983, 45),
    (2048, 64, 3),
]


class TestTorchMethod(TestCase):

    def test_cummax_1d(self):
        print(
            "\n-----------------------1D---------------------------------------------"
        )
        dtypes = [torch.float, torch.bfloat16]
        for dtype in dtypes:
            for shape in test_1d_shapes:
                x2 = torch.randn(shape, device=cpu_device, dtype=dtype)
                x2_dpcpp = x2.to(dpcpp_device)
                if x2.size(0) > 100:
                    atol = 1
                else:
                    atol = 1e-1
                for dim_ in range(1):
                    print("1D, dtype:", dtype, ", shape:", shape, ", dim:", dim_)
                    re_cpu = torch.cummax(x2, dim=dim_)
                    re_xpu = torch.cummax(x2_dpcpp, dim=dim_)
                    self.assertEqual(
                        re_cpu[0],
                        re_xpu[0].to(cpu_device),
                        rtol=1.0e-4,
                        atol=atol,
                    )
                    self.assertEqual(
                        re_cpu[1],
                        re_xpu[1].to(cpu_device),
                        rtol=10e-4,
                        atol=atol,
                    )

    def test_cummax_2d(self):
        print(
            "\n-----------------------2D---------------------------------------------"
        )
        dtypes = [torch.float, torch.bfloat16]
        for dtype in dtypes:
            for shape in test_2d_shapes:
                x2 = torch.randn(shape, device=cpu_device, dtype=dtype)
                x2_dpcpp = x2.to(dpcpp_device)
                if x2.size(0) > 100:
                    atol = 1
                else:
                    atol = 1e-1
                for dim_ in range(2):
                    re_cpu = torch.cummax(x2, dim=dim_)
                    re_xpu = torch.cummax(x2_dpcpp, dim=dim_)
                    self.assertEqual(
                        re_cpu[0],
                        re_xpu[0].to(cpu_device),
                        rtol=1.0e-4,
                        atol=atol,
                    )
                    self.assertEqual(
                        re_cpu[1],
                        re_xpu[1].to(cpu_device),
                        rtol=10e-4,
                        atol=atol,
                    )

    def test_cummax_3d(self):
        print(
            "\n-----------------------3D---------------------------------------------"
        )
        dtypes = [torch.float, torch.bfloat16]
        for dtype in dtypes:
            for shape in test_3d_shapes:
                x2 = torch.randn(shape, device=cpu_device, dtype=dtype)
                x2_dpcpp = x2.to(dpcpp_device)
                if x2.size(0) > 100:
                    atol = 1
                else:
                    atol = 1e-1
                for dim_ in range(3):
                    print("3D, dtype:", dtype, ", shape:", shape, ", dim:", dim_)
                    re_cpu = torch.cummax(x2, dim=dim_)
                    re_xpu = torch.cummax(x2_dpcpp, dim=dim_)
                    self.assertEqual(
                        re_cpu[0],
                        re_xpu[0].to(cpu_device),
                        rtol=1.0e-4,
                        atol=atol,
                    )
                    self.assertEqual(
                        re_cpu[1],
                        re_xpu[1].to(cpu_device),
                        rtol=1.0e-4,
                        atol=atol,
                    )

    def test_mult_transposed_dim(self, dtype=torch.float):
        a_cpu = torch.randn(2, 2, 2)
        a_xpu = a_cpu.to(dpcpp_device)

        a_cpu = a_cpu.transpose(1, 2)
        a_xpu = a_xpu.transpose(1, 2)
        b_cpu = torch.cummax(a_cpu, dim=1)
        b_xpu = torch.cummax(a_xpu, dim=1)
        self.assertEqual(b_cpu[0], b_xpu[0].to(cpu_device), rtol=1e-5, atol=1e-4)
        self.assertEqual(b_cpu[1], b_xpu[1].to(cpu_device), rtol=1e-5, atol=1e-4)

    def test_mult_strided_dim(self, dtype=torch.float):
        a_cpu = torch.randn(8, 82, 814)
        a_xpu = a_cpu.to(dpcpp_device)

        a_strided_cpu = torch.as_strided(
            a_cpu, (4, 41, 407), (2 * 814 * 82, 2 * 814, 2)
        )
        a_strided_xpu = torch.as_strided(
            a_xpu, (4, 41, 407), (2 * 814 * 82, 2 * 814, 2)
        )

        b_cpu = torch.cummax(a_strided_cpu, dim=2)
        b_xpu = torch.cummax(a_strided_xpu, dim=2)
        self.assertEqual(b_cpu[0], b_xpu[0].to(cpu_device), rtol=1e-5, atol=1e-4)
        self.assertEqual(b_cpu[1], b_xpu[1].to(cpu_device), rtol=1e-5, atol=1e-4)

        b_cpu = torch.cummax(a_strided_cpu, dim=1)
        b_xpu = torch.cummax(a_strided_xpu, dim=1)
        self.assertEqual(b_cpu[0], b_xpu[0].to(cpu_device), rtol=1e-5, atol=1e-4)
        self.assertEqual(b_cpu[1], b_xpu[1].to(cpu_device), rtol=1e-5, atol=1e-4)

        b_cpu = torch.cummax(a_strided_cpu, dim=0)
        b_xpu = torch.cummax(a_strided_xpu, dim=0)
        self.assertEqual(b_cpu[0], b_xpu[0].to(cpu_device), rtol=1e-5, atol=1e-4)
        self.assertEqual(b_cpu[1], b_xpu[1].to(cpu_device), rtol=1e-5, atol=1e-4)

    def test_scan_single_kernel(self, dtype=torch.float):
        a_cpu = torch.ones((257, 2049), device=cpu_device).to(dtype)
        a_xpu = a_cpu.to(dpcpp_device).to(dtype)

        res_cpu = torch.cummax(a_cpu, dim=1)
        res_xpu = torch.cummax(a_xpu, dim=1)
        if a_cpu.size(0) > 100:
            atol = 1
        else:
            atol = 1e-1
        self.assertEqual(
            res_cpu[0], res_xpu[0].to(cpu_device).to(torch.float), rtol=10e-4, atol=atol
        )
        self.assertEqual(
            res_cpu[1], res_xpu[1].to(cpu_device), rtol=10e-4, atol=atol
        )
