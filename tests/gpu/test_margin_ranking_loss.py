import torch
import torch.nn as nn
import torch_ipex
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device('cpu')
dpcpp_device = torch.device('dpcpp')


class TestNNMethod(TestCase):
    def test_margin_ranking_loss(self, dtype=torch.float):

        input1 = torch.randn(3, 5)
        input2 = torch.randn(3, 5)
        target = torch.ones(3, 1)

        input1_cpu = input1
        input2_cpu = input2
        target_cpu = target

        input1_dpcpp = input1.to("dpcpp")
        input2_dpcpp = input2.to("dpcpp")
        target_dpcpp = target.to("dpcpp")

        def _test_cpu(input1, input2, target, reduc):
            loss = nn.MarginRankingLoss(reduction=reduc)
            input1.requires_grad = True
            input2.requires_grad = True
            output = loss(input1, input2, target)
            print(output)
            if(reduc == "none"):
                output.backward(torch.ones_like(input1, dtype=torch.float))
            else:
                output.backward(torch.tensor((1.0), dtype=torch.float))
            print(input1.grad)
            print(input2.grad)
            return input1, input2
            input1.grad.zero_()
            input2.grad.zero_()

        def _test_dpcpp(input1, input2, target, reduc):
            loss = nn.MarginRankingLoss(reduction=reduc)
            input1.requires_grad = True
            input2.requires_grad = True
            output = loss(input1, input2, target)
            print(output.cpu())
            if(reduc == "none"):
                output.backward(torch.ones_like(
                    input1, dtype=torch.float).to("dpcpp"))
            else:
                output.backward(torch.tensor(
                    (1.0), dtype=torch.float).to("dpcpp"))
            print(input1.grad.cpu())
            print(input2.grad.cpu())

            return input1, input2
            #finally:
            #    input1.grad.zero_()
            #    input2.grad.zero_()

        print('none')
        print("cpu")
        input1_cpu, input2_cpu = _test_cpu(
            input1_cpu, input2_cpu, target_cpu, "none")
        print("dpcpp")
        input1_dpcpp, input2_dpcpp = _test_dpcpp(
            input1_dpcpp, input2_dpcpp, target_dpcpp, "none")
        print(input1_cpu.grad)
        print(input1_dpcpp.grad.cpu())
        self.assertEqual(input1_cpu.grad, input1_dpcpp.grad.cpu())
        self.assertEqual(input2_cpu.grad, input2_dpcpp.grad.cpu())

        print('sum')
        print("cpu")
        input1_cpu, input2_cpu = _test_cpu(
            input1_cpu, input2_cpu, target_cpu, "sum")
        print("dpcpp")
        input1_dpcpp, input2_dpcpp = _test_dpcpp(
            input1_dpcpp, input2_dpcpp, target_dpcpp, "sum")
        self.assertEqual(input1_cpu.grad, input1_dpcpp.grad.cpu())
        self.assertEqual(input2_cpu.grad, input2_dpcpp.grad.cpu())

        print('mean')
        print("cpu")
        input1_cpu, input2_cpu = _test_cpu(
            input1_cpu, input2_cpu, target_cpu, "mean")
        print("dpcpp")
        input1_dpcpp, input2_dpcpp = _test_dpcpp(
            input1_dpcpp, input2_dpcpp, target_dpcpp, "mean")
        self.assertEqual(input1_cpu.grad, input1_dpcpp.grad.cpu())
        self.assertEqual(input2_cpu.grad, input2_dpcpp.grad.cpu())
