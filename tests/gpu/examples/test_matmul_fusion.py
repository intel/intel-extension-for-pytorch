# from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase


import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
print_graph = True


class MatmulSum(torch.nn.Module):
    def __init__(self):
        super(MatmulSum, self).__init__()

    def forward(self, m1, m2, a):
        y = torch.matmul(m1, m2)
        y += a
        return y


class MatmulRelu(torch.nn.Module):
    def __init__(self) -> None:
        super(MatmulRelu, self).__init__()

    def forward(self, m1, m2):
        return torch.relu(torch.matmul(m1, m2))


class TMatmulRelu(torch.nn.Module):
    def __init__(self) -> None:
        super(TMatmulRelu, self).__init__()

    def forward(self, m1, m2):
        return torch.relu(torch.matmul(m1, m2.t()))


class TransMatmulAddGelu(torch.nn.Module):
    def __init__(self):
        super(TransMatmulAddGelu, self).__init__()

    def forward(self, m1, m2, add):
        return F.gelu(torch.add(torch.matmul(m1, m2.t()), add, alpha=2.0))


class TransMatmulAddAdd(torch.nn.Module):
    def __init__(self):
        super(TransMatmulAddAdd, self).__init__()

    def forward(self, m1, m2, add1, add2):
        return torch.add(torch.matmul(m1, m2.t()), add1, alpha=2.0) + add2


class LinearReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x))
        return x


class LinearAdd(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearAdd, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, x1):
        x = self.linear(x)
        y = x + x1
        return y


class LinearBinaryMul(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearBinaryMul, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        y = torch.mul(self.linear(x), a)
        return y


matmul_binary_shapes = [
    [[4, 2], [2], [4]],  # case 2
    [[4, 2], [2], [1]],
    [[2], [2, 6], [6]],  # case 3
    [[2], [2, 6], [1]],
    [[4, 2], [2, 6], [4, 6]],  # case 4
    [[4, 2], [2, 6], [4, 1]],
    [[4, 2], [2, 6], [1, 6]],
    [[4, 2], [2, 6], [1, 1]],
    [[4, 2], [2, 6], [6]],
    [[4, 2], [2, 6], [1]],
    [[3, 4, 2], [2, 6], [3, 4, 6]],  # case 5-1
    [[3, 4, 2], [2, 6], [3, 4, 1]],
    [[3, 4, 2], [2, 6], [1, 4, 6]],
    [[3, 4, 2], [2, 6], [3, 1, 6]],
    [[3, 4, 2], [2, 6], [3, 1, 1]],
    [[3, 4, 2], [2, 6], [1, 4, 1]],
    [[3, 4, 2], [2, 6], [1, 1, 6]],
    [[3, 4, 2], [2, 6], [1, 1, 1]],
    [[3, 4, 2], [2, 6], [4, 6]],
    [[3, 4, 2], [2, 6], [1, 6]],
    [[3, 4, 2], [2, 6], [4, 1]],
    [[3, 4, 2], [2, 6], [1, 1]],
    [[3, 4, 2], [2, 6], [6]],
    [[3, 4, 2], [2, 6], [1]],
    [[3, 4, 2], [2], [3, 4]],  # case 5-2
    [[3, 4, 2], [2], [3, 1]],
    [[3, 4, 2], [2], [1, 4]],
    [[3, 4, 2], [2], [1, 1]],
    [[3, 4, 2], [2], [4]],
    [[3, 4, 2], [2], [1]],
    [[2], [3, 2, 4], [3, 4]],  # case 6-1
    [[2], [3, 2, 4], [3, 1]],
    [[2], [3, 2, 4], [1, 4]],
    [[2], [3, 2, 4], [1, 1]],
    [[2], [3, 2, 4], [4]],
    [[2], [3, 2, 4], [1]],
    [[2], [2, 3, 2, 4], [1]],
    [[6, 2], [3, 2, 4], [3, 6, 4]],  # case 6-2
    [[6, 2], [3, 2, 4], [1, 6, 4]],
    [[6, 2], [3, 2, 4], [3, 1, 4]],
    [[6, 2], [3, 2, 4], [3, 6, 1]],
    [[6, 2], [3, 2, 4], [1, 1, 4]],
    [[6, 2], [3, 2, 4], [1, 6, 1]],
    [[6, 2], [3, 2, 4], [3, 1, 1]],
    [[6, 2], [3, 2, 4], [1, 1, 1]],
    [[6, 2], [3, 2, 4], [6, 4]],
    [[6, 2], [3, 2, 4], [1, 4]],
    [[6, 2], [3, 2, 4], [6, 1]],
    [[6, 2], [3, 2, 4], [1, 1]],
    [[6, 2], [3, 2, 4], [4]],
    [[6, 2], [3, 2, 4], [1]],
    [[6, 2], [2, 3, 2, 4], [1]],
    [[6, 2], [2, 3, 2, 4], [1]],
    [[3, 4, 2], [3, 2, 6], [3, 4, 6]],  # case 7-1
    [[3, 4, 2], [3, 2, 6], [1, 4, 6]],
    [[3, 4, 2], [3, 2, 6], [3, 1, 6]],
    [[3, 4, 2], [3, 2, 6], [3, 4, 1]],
    [[3, 4, 2], [3, 2, 6], [1, 1, 6]],
    [[3, 4, 2], [3, 2, 6], [1, 4, 1]],
    [[3, 4, 2], [3, 2, 6], [3, 1, 1]],
    [[3, 4, 2], [3, 2, 6], [1, 1, 1]],
    [[3, 4, 2], [3, 2, 6], [4, 6]],
    [[3, 4, 2], [3, 2, 6], [6]],
    [[3, 4, 2], [3, 2, 6], [1]],
    [[5, 1, 4, 2], [3, 2, 6], [5, 3, 4, 6]],  # case 7-2
    [[5, 1, 4, 2], [3, 2, 6], [1, 3, 4, 6]],
    [[5, 1, 4, 2], [3, 2, 6], [5, 1, 4, 6]],
    [[5, 1, 4, 2], [3, 2, 6], [1, 1, 4, 6]],
    [[5, 1, 4, 2], [3, 2, 6], [5, 3, 1, 6]],
    [[5, 1, 4, 2], [3, 2, 6], [5, 3, 4, 1]],
    [[5, 1, 4, 2], [3, 2, 6], [5, 3, 1, 1]],
    [[5, 1, 4, 2], [3, 2, 6], [5, 1, 1, 1]],
    [[5, 1, 4, 2], [3, 2, 6], [1, 3, 1, 1]],
    [[5, 1, 4, 2], [3, 2, 6], [1, 1, 1, 1]],
    [[5, 1, 4, 2], [3, 2, 6], [3, 4, 6]],
    [[5, 1, 4, 2], [3, 2, 6], [4, 6]],
    [[5, 1, 4, 2], [3, 2, 6], [6]],
    [[5, 1, 4, 2], [3, 2, 6], [1]],
]

linear_binary_shapes = [
    [[2], [6, 2], [6]],
    [[2], [6, 2], [1]],
    [[4, 2], [6, 2], [4, 6]],
    [[4, 2], [6, 2], [4, 1]],
    [[4, 2], [6, 2], [1, 6]],
    [[4, 2], [6, 2], [1, 1]],
    [[4, 2], [6, 2], [6]],
    [[4, 2], [6, 2], [1]],
    [[3, 4, 2], [6, 2], [3, 4, 6]],
    [[3, 4, 2], [6, 2], [3, 4, 1]],
    [[3, 4, 2], [6, 2], [1, 4, 6]],
    [[3, 4, 2], [6, 2], [3, 1, 6]],
    [[3, 4, 2], [6, 2], [3, 1, 1]],
    [[3, 4, 2], [6, 2], [1, 4, 1]],
    [[3, 4, 2], [6, 2], [1, 1, 6]],
    [[3, 4, 2], [6, 2], [1, 1, 1]],
    [[3, 4, 2], [6, 2], [4, 6]],
    [[3, 4, 2], [6, 2], [1, 6]],
    [[3, 4, 2], [6, 2], [4, 1]],
    [[3, 4, 2], [6, 2], [1, 1]],
    [[3, 4, 2], [6, 2], [6]],
    [[3, 4, 2], [6, 2], [1]],
    [[5, 3, 4, 2], [6, 2], [5, 3, 4, 6]],
    [[5, 3, 4, 2], [6, 2], [1, 3, 4, 6]],
    [[5, 3, 4, 2], [6, 2], [5, 1, 4, 6]],
    [[5, 3, 4, 2], [6, 2], [1, 1, 4, 6]],
    [[5, 3, 4, 2], [6, 2], [5, 3, 1, 6]],
    [[5, 3, 4, 2], [6, 2], [5, 3, 4, 1]],
    [[5, 3, 4, 2], [6, 2], [5, 3, 1, 1]],
    [[5, 3, 4, 2], [6, 2], [5, 1, 1, 1]],
    [[5, 3, 4, 2], [6, 2], [1, 3, 1, 1]],
    [[5, 3, 4, 2], [6, 2], [1, 1, 1, 1]],
    [[5, 3, 4, 2], [6, 2], [3, 4, 6]],
    [[5, 3, 4, 2], [6, 2], [4, 6]],
    [[5, 3, 4, 2], [6, 2], [6]],
    [[5, 3, 4, 2], [6, 2], [1]],
]


class TestNNMethod(TestCase):
    def test_matmul_relu_fusion(self, dtype=torch.float):
        shapes = [
            [[6], [6]],
            [[4, 2], [2]],
            [[2], [2, 6]],
            [[4, 2], [2, 6]],
            [[3, 4, 2], [2, 6]],
            [[3, 4, 2], [2]],
            [[2], [3, 2, 4]],
            [[2], [2, 3, 2, 4]],
            [[6, 2], [3, 2, 4]],
            [[6, 2], [2, 3, 2, 4]],
            [[1, 2], [2, 3, 2, 4]],
            [[3, 4, 2], [3, 2, 6]],
            [[5, 1, 4, 2], [3, 2, 6]],
        ]
        for shape in shapes:
            m1 = torch.randn(shape[0], device=cpu_device)
            m2 = torch.randn(shape[1], device=cpu_device)
            model = MatmulRelu()

            raw = model(m1.clone(), m2.clone())
            m1_dpcpp = m1.to(dpcpp_device)
            m2_dpcpp = m2.to(dpcpp_device)
            model = model.to(dpcpp_device)
            modelJit = torch.jit.script(model)
            for i in range(2):
                modelJit(m1_dpcpp, m2_dpcpp)

            with torch.no_grad():
                if print_graph:
                    print(modelJit.graph_for(m1_dpcpp, m2_dpcpp))
                real = modelJit(m1_dpcpp, m2_dpcpp)
            self.assertEqual(raw.shape, real.shape)
            self.assertEqual(raw, real.to(cpu_device))

    def test_matmul_binary_add_fusion(self, dtype=torch.float):
        for shape in matmul_binary_shapes:
            m1 = torch.randn(shape[0], device=cpu_device)
            m2 = torch.randn(shape[1], device=cpu_device)
            acc = torch.randn(shape[2], device=cpu_device)
            print("m1:", m1.shape)
            print("m2:", m2.shape)
            print("acc:", acc.shape)
            m1_dpcpp_orig = m1.clone().to(dpcpp_device)
            m2_dpcpp_orig = m2.clone().to(dpcpp_device)
            acc_dpcpp_orig = acc.clone().to(dpcpp_device)
            model = MatmulSum()
            raw = model(m1, m2, acc)
            print("raw: ", raw)

            model = model.to(dpcpp_device)
            modelJit = torch.jit.script(model)
            m1_dpcpp = m1_dpcpp_orig.clone()
            m2_dpcpp = m2_dpcpp_orig.clone()
            acc_dpcpp = acc_dpcpp_orig.clone()

            with torch.no_grad():
                m1_dpcpp = m1_dpcpp_orig.clone()
                m2_dpcpp = m2_dpcpp_orig.clone()
                acc_dpcpp = acc_dpcpp_orig.clone()
                for i in range(5):
                    real = modelJit(m1_dpcpp, m2_dpcpp, acc_dpcpp)
                print("real: ", real.cpu())
            self.assertEqual(raw.shape, real.shape)
            self.assertEqual(raw, real.to(cpu_device))
            del modelJit

    def test_trans_matmul_relu_fusion(self, dtype=torch.float):
        shapes = [
            [[2], [6, 2]],
            [[4, 2], [6, 2]],
            [[3, 4, 2], [6, 2]],
            [[5, 3, 4, 2], [6, 2]],
        ]
        for shape in shapes:
            m1 = torch.randn(shape[0], device=cpu_device)
            m2 = torch.randn(shape[1], device=cpu_device)
            model = TMatmulRelu()

            raw = model(m1.clone(), m2.clone())
            m1_dpcpp = m1.to(dpcpp_device)
            m2_dpcpp = m2.to(dpcpp_device)
            model = model.to(dpcpp_device)
            modelJit = torch.jit.script(model)
            for i in range(2):
                modelJit(m1_dpcpp, m2_dpcpp)

            with torch.no_grad():
                if print_graph:
                    print(modelJit.graph_for(m1_dpcpp, m2_dpcpp))
                real = modelJit(m1_dpcpp, m2_dpcpp)
            self.assertEqual(raw.shape, real.shape)
            self.assertEqual(raw, real.to(cpu_device))

    def test_trans_matmul_add_add(self, dtype=torch.float):
        for shape in linear_binary_shapes:
            m1 = torch.randn(shape[0], device=cpu_device)
            m2 = torch.randn(shape[1], device=cpu_device)
            add1 = torch.randn(shape[2], device=cpu_device)
            add2 = torch.randn(shape[2], device=cpu_device)

            model = TransMatmulAddAdd()
            raw = model(m1, m2, add1, add2)
            print("raw: ", raw)

            m1_dpcpp = m1.to(dpcpp_device)
            m2_dpcpp = m2.to(dpcpp_device)
            add1_dpcpp = add1.to(dpcpp_device)
            add2_dpcpp = add2.to(dpcpp_device)

            modelJit = torch.jit.script(model)
            with torch.no_grad():
                real = modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp, add2_dpcpp)
                print("real:", real.to(cpu_device))
            self.assertEqual(raw, real.to(cpu_device))
            del modelJit

    def test_trans_matmul_add_gelu(self, dtype=torch.float):
        for shape in linear_binary_shapes:
            m1 = torch.randn(shape[0], device=cpu_device)
            m2 = torch.randn(shape[1], device=cpu_device)
            add1 = torch.randn(shape[2], device=cpu_device)

            model = TransMatmulAddGelu()
            raw = model(m1, m2, add1)
            print("raw: ", raw)

            m1_dpcpp = m1.to(dpcpp_device)
            m2_dpcpp = m2.to(dpcpp_device)
            add1_dpcpp = add1.to(dpcpp_device)

            modelJit = torch.jit.script(model)
            with torch.no_grad():
                real = modelJit(m1_dpcpp, m2_dpcpp, add1_dpcpp)
                print("real:", real.to(cpu_device))
            self.assertEqual(raw, real.to(cpu_device))
            del modelJit

    def test_linear_relu_fusion(self, dtype=torch.float):
        shapes = [
            [[2], [2, 6]],
            [[4, 2], [2, 6]],
            [[3, 4, 2], [2, 6]],
            [[5, 3, 4, 2], [2, 6]],
        ]

        for shape in shapes:
            ic = shape[1][0]
            oc = shape[1][1]
            x = torch.randn(shape[0], device=cpu_device)

            model = LinearReLU(ic, oc)
            y = model(x)
            print("raw: ", y)

            x = x.to("xpu")
            model.to("xpu")
            modelJit = torch.jit.trace(model, x)

            with torch.no_grad():
                for i in range(3):
                    if print_graph and i == 2:
                        print(modelJit.graph_for(x))
                y_dpcpp = modelJit(x)
                print("fusion:", y_dpcpp.cpu())
            self.assertEqual(y, y_dpcpp.to(cpu_device))
            del modelJit

    def test_linear_binary_add_fusion(self, dtype=torch.float):
        for shape in linear_binary_shapes:
            ic = shape[1][1]
            oc = shape[1][0]
            x = torch.randn(shape[0], device=cpu_device)
            x1 = torch.randn(shape[2], device=cpu_device)

            model = LinearAdd(ic, oc)
            y = model(x, x1)
            print("raw: ", y)

            x = x.to("xpu")
            x1 = x1.to("xpu")
            model.to("xpu")
            modelJit = torch.jit.script(model)

            with torch.no_grad():
                for i in range(3):
                    if print_graph and i == 2:
                        print(modelJit.graph_for(x, x1))
                y_dpcpp = modelJit(x, x1)
                print("fusion:", y_dpcpp.cpu())
            self.assertEqual(y, y_dpcpp.to(cpu_device), atol=1e-3, rtol=1.3e-6)
        del modelJit

    def test_linear_binary_mul_fusion(self, dtype=torch.float):
        for shape in linear_binary_shapes:
            ic = shape[1][1]
            oc = shape[1][0]
            x = torch.randn(shape[0], device=cpu_device)
            x1 = torch.randn(shape[2], device=cpu_device)

            model = LinearBinaryMul(ic, oc)
            y = model(x, x1)
            print("raw: ", y)

            x_xpu = x.to("xpu")
            x1_xpu = x1.to("xpu")
            model.to("xpu")
            modelJit = torch.jit.script(model)

            with torch.no_grad():
                for i in range(3):
                    if print_graph and i == 2:
                        print(modelJit.graph_for(x_xpu, x1_xpu))
                y_dpcpp = modelJit(x_xpu, x1_xpu)
                print("fusion:", y_dpcpp.cpu())
            self.assertEqual(y, y_dpcpp.to(cpu_device), atol=1e-3, rtol=1.3e-6)
        del modelJit
