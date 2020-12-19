import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


def _test_cpu(input1, input2, target, reduc):
    loss = nn.CosineEmbeddingLoss(reduction=reduc)
    input1.requires_grad = True
    input2.requires_grad = True
    output = loss(input1, input2, target)
    print(output)
    if (reduc == "none"):
        output.backward(torch.ones((2, 1), dtype=torch.float))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float))
    print(input1.grad)
    print(input2.grad)

    input1.grad.zero_()
    input2.grad.zero_()
    return output


def _test_dpcpp(input1, input2, target, reduc):
    loss = nn.CosineEmbeddingLoss(reduction=reduc)
    input1.requires_grad = True
    input2.requires_grad = True
    output = loss(input1, input2, target)
    print(output.cpu())
    if (reduc == "none"):
        output.backward(torch.ones((2, 1), dtype=torch.float).to(dpcpp_device))
    else:
        output.backward(torch.tensor(
            (1.0), dtype=torch.float).to(dpcpp_device))
    print(input1.grad.cpu())
    print(input2.grad.cpu())
    input1.grad.zero_()
    input2.grad.zero_()
    return output


class TestNNMethod(TestCase):
    def test_cosine_embedding_loss(self, dtype=torch.float):
        input1 = torch.randn(1, 5)
        input2 = torch.randn(1, 5)
        target = torch.tensor([[1], [-1]])

        input1_cpu = input1
        input2_cpu = input2
        target_cpu = target

        input1_dpcpp = input1.to("xpu")
        input2_dpcpp = input2.to("xpu")
        target_dpcpp = target.to("xpu")
        print('none')
        print("cpu")
        output_cpu_1 = _test_cpu(input1_cpu, input2_cpu, target_cpu, "none")
        print("xpu")
        output_dpcpp_1 = _test_dpcpp(
            input1_dpcpp, input2_dpcpp, target_dpcpp, "none")

        print('sum')
        print("cpu")
        output_cpu_2 = _test_cpu(input1_cpu, input2_cpu, target_cpu, "sum")
        print("xpu")
        output_dpcpp_2 = _test_dpcpp(
            input1_dpcpp, input2_dpcpp, target_dpcpp, "sum")

        print('mean')
        print("cpu")
        output_cpu_3 = _test_cpu(input1_cpu, input2_cpu, target_cpu, "mean")
        print("xpu")
        output_dpcpp_3 = _test_dpcpp(
            input1_dpcpp, input2_dpcpp, target_dpcpp, "mean")

        self.assertEqual(output_cpu_1, output_dpcpp_1.to(cpu_device))
        self.assertEqual(output_cpu_2, output_dpcpp_2.to(cpu_device))
        self.assertEqual(output_cpu_3, output_dpcpp_3.to(cpu_device))
