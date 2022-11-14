import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_set(self, dtype=torch.float):
        x_cpu1 = torch.randn((5, 4))
        x_cpu2 = torch.randn((5, 4))
        x_dpcpp1 = x_cpu1.to("xpu")
        x_dpcpp2 = x_cpu2.to("xpu")

        print("Before:")
        print("self dpcpp", x_dpcpp1.to("cpu"))
        print("src dpcpp", x_dpcpp2.to("cpu"))

        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(x_cpu2, x_dpcpp2.cpu())

        x_cpu1.set_(x_cpu2)
        x_dpcpp1.set_(x_dpcpp2)

        print("After:")
        print("self dpcpp", x_dpcpp1.to("cpu"))
        print("src dpcpp", x_dpcpp2.to("cpu"))
        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(x_cpu2, x_dpcpp2.cpu())

    def test_tensor_set(self):
        x_xpu1 = torch.tensor([]).to("xpu")
        x_xpu2 = torch.empty(3, 4, 9, 10).uniform_().to("xpu")
        x_xpu1.set_(x_xpu2)
        self.assertEqual(x_xpu1.storage()._cdata, x_xpu2.storage()._cdata)
        size = torch.Size([9, 3, 4, 10])
        x_xpu1.set_(x_xpu2.storage(), 0, size)
        self.assertEqual(x_xpu1.size(), size)
        x_xpu1.set_(x_xpu2.storage(), 0, tuple(size))
        self.assertEqual(x_xpu1.size(), size)
        self.assertEqual(x_xpu1.stride(), (120, 40, 10, 1))
        stride = (10, 360, 90, 1)
        x_xpu1.set_(x_xpu2.storage(), 0, size, stride)
        self.assertEqual(x_xpu1.stride(), stride)
        x_xpu1.set_(x_xpu2.storage(), 0, size=size, stride=stride)
        self.assertEqual(x_xpu1.size(), size)
        self.assertEqual(x_xpu1.stride(), stride)

        # test argument names
        x_xpu1 = torch.tensor([]).to("xpu")
        # 1. case when source is tensor
        x_xpu1.set_(source=x_xpu2)
        self.assertEqual(x_xpu1.storage()._cdata, x_xpu2.storage()._cdata)
        # 2. case when source is storage
        x_xpu1.set_(source=x_xpu2.storage())
        self.assertEqual(x_xpu1.storage()._cdata, x_xpu2.storage()._cdata)
        # 3. case when source is storage, and other args also specified
        x_xpu1.set_(source=x_xpu2.storage(), storage_offset=0, size=size, stride=stride)
        self.assertEqual(x_xpu1.size(), size)
        self.assertEqual(x_xpu1.stride(), stride)

        x_xpu1 = torch.tensor([True, True], dtype=torch.bool).to("xpu")
        x_xpu2 = torch.tensor([False, False], dtype=torch.bool).to("xpu")
        x_xpu1.set_(x_xpu2)
        self.assertEqual(x_xpu1.storage()._cdata, x_xpu2.storage()._cdata)

    def test_tensor_set_errors(self):
        f_cpu = torch.randn((2, 3), dtype=torch.float32)

        f_xpu = torch.randn((2, 3), dtype=torch.float32).to("xpu")
        d_xpu = torch.randn((2, 3), dtype=torch.float64).to("xpu")

        # change dtype
        self.assertRaises(RuntimeError, lambda: f_xpu.set_(d_xpu.storage()))
        self.assertRaises(RuntimeError, lambda: f_xpu.set_(d_xpu.storage(), 0, d_xpu.size(), d_xpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_xpu.set_(d_xpu))

        # change device
        # cpu -> xpu
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_xpu.storage()))
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_xpu.storage(), 0, f_xpu.size(), f_xpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_xpu))

        # xpu -> cpu
        self.assertRaises(RuntimeError, lambda: f_xpu.set_(f_cpu.storage()))
        self.assertRaises(RuntimeError, lambda: f_xpu.set_(f_cpu.storage(), 0, f_cpu.size(), f_cpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_xpu.set_(f_cpu))
