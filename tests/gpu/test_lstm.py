import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_ipex
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device('cpu')
dpcpp_device = torch.device('dpcpp')


class TestNNMethod(TestCase):
    def test_lstm(self, dtype=torch.float):
        rnn = nn.LSTM(2, 4, 2, bidirectional=True)

        input = torch.tensor([[[1, 2]]], dtype=torch.float,
                             device=cpu_device)  # (1,1,2)
        input.requires_grad = True
        h0 = torch.tensor([[[2, 3, 2, 3]], [[3, 4, 3, 4]], [[4, 5, 4, 5]], [
            [5, 6, 5, 6]]], dtype=torch.float, device=cpu_device)  # (4,1,2)
        c0 = torch.tensor([[[6, 7, 6, 7]], [[7, 8, 7, 8]], [[8, 9, 8, 9]], [
            [9, 1, 9, 1]]], dtype=torch.float, device=cpu_device)  # (4,1,2)
        h0.requires_grad = True
        c0.requires_grad = True
        output, (hn, cn) = rnn(input, (h0, c0))
        print("cpu output = ", output)
        print("cpu hn = ", hn)
        print("cpu cn = ", cn)

        grad_out = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]],
                                dtype=torch.float,  device=cpu_device)  # (1,1,8)
        grad_out = Variable(grad_out, requires_grad=True)
        output.backward(grad_out)
        input_grad = input.grad
        h0_grad = h0.grad
        c0_grad = c0.grad
        param_grad = []
        for param in rnn._parameters.values():
            param_grad.append(param._grad)
        print("cpu grad_src = ", input_grad)
        print("cpu grad_h0 = ", h0_grad)
        print("cpu grad_c0 = ", c0_grad)
        print("cpu grad_params = ")
        for i in range(len(param_grad)):
            print(param_grad[i])

        input_dpcpp = torch.tensor(
            [[[1, 2]]], dtype=torch.float, device=dpcpp_device)
        input_dpcpp.requires_grad = True
        rnn_dpcpp = rnn.to("dpcpp")
        rnn_dpcpp.zero_grad()
        h0_dpcpp = torch.tensor([[[2, 3, 2, 3]], [[3, 4, 3, 4]], [[4, 5, 4, 5]], [
            [5, 6, 5, 6]]], dtype=torch.float,  device=dpcpp_device)
        c0_dpcpp = torch.tensor([[[6, 7, 6, 7]], [[7, 8, 7, 8]], [[8, 9, 8, 9]], [
            [9, 1, 9, 1]]], dtype=torch.float,  device=dpcpp_device)
        h0_dpcpp.requires_grad = True
        c0_dpcpp.requires_grad = True
        output_dpcpp, (hn_dpcpp, cn_dpcpp) = rnn_dpcpp(
            input_dpcpp, (h0_dpcpp, c0_dpcpp))
        print("dpcpp output = ", output_dpcpp.cpu())
        print("dpcpp hn = ", hn_dpcpp.cpu())
        print("dpcpp cn = ", cn_dpcpp.cpu())

        grad_out_dpcpp = grad_out.to("dpcpp")
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)
        output_dpcpp.backward(grad_out_dpcpp)
        input_dpcpp_grad = input_dpcpp.grad
        h0_dpcpp_grad = h0_dpcpp.grad
        c0_dpcpp_grad = c0_dpcpp.grad
        param_dpcpp_grad = []
        for param_dpcpp in rnn_dpcpp._parameters.values():
            param_dpcpp_grad.append(param_dpcpp._grad)
        print("dpcpp grad_src = ", input_dpcpp_grad.cpu())
        print("dpcpp grad_h0 = ", h0_dpcpp_grad.cpu())
        print("dpcpp grad_c0 = ", c0_dpcpp_grad.cpu())
        print("dpcpp grad_params = ")
        for i in range(len(param_dpcpp_grad)):
            print(param_dpcpp_grad[i].cpu())

        # check result
        print("forward result:")
        print("output: ", output - output_dpcpp.cpu())
        print("hn: ", hn - hn_dpcpp.cpu())
        print("cn: ", cn - cn_dpcpp.cpu())
        self.assertEqual(output, output_dpcpp.cpu())
        self.assertEqual(hn, hn_dpcpp.cpu())
        self.assertEqual(cn, cn_dpcpp.cpu())

        print("backward result:")
        print("grad_src: ", input_grad - input_dpcpp_grad.cpu())
        print("grad_h0: ", h0_grad - h0_dpcpp_grad.cpu())
        print("grad_c0: ", c0_grad - c0_dpcpp_grad.cpu())
        self.assertEqual(input_grad, input_dpcpp_grad.cpu())
        self.assertEqual(h0_grad, h0_dpcpp_grad.cpu())
        self.assertEqual(c0_grad, c0_dpcpp_grad.cpu())
        print("grad_params: ")
        for i in range(len(param_dpcpp_grad)):
            print(param_grad[i] - param_dpcpp_grad[i].cpu())
            self.assertEqual(param_grad[i], param_dpcpp_grad[i].cpu())
