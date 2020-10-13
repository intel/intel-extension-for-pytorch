import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
# import matplotlib.pyplot as plt
import pytest

def remove_w(weight, i):
    temp = weight.clone()
    temp[i] = 0
    return temp


def renorm_1d(weight):
    temp = weight.unsqueeze(0).renorm(1, 0, 1)
    temp.squeeze_()
    return temp


def chose_prob(weight, n, i, k):
    if n == 0:
        temp = weight.unsqueeze(0).renorm(1, 0, 1).squeeze()
        return temp[i], remove_w(weight, i)  # remove chosen probs
    else:
        w = 0.0
        temp = weight.clone()
        for j in range(k):
            if j != i:
                p, _t = chose_prob(weight, n-1, j, k)
                w += p*renorm_1d(_t)[i]
        return w, remove_w(weight, i)  # remove chosen probs


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    @pytest.mark.skip(reason='Random Data Generate')
    def test_multinomial(self, dtype=torch.float):
        # create a tensor of weights
        w = torch.tensor([1, 10, 3, 2], dtype=torch.float)
        print("weights", w.unsqueeze(0).renorm(1, 0, 1))

        weights = torch.ones(1000, w.size()[0])
        weights[:] = w

        weights_dpcpp = weights.to(dpcpp_device)
        x_cpu = torch.multinomial(weights, 1000, replacement=True)
        x_dpcpp = torch.multinomial(weights_dpcpp, 1000, replacement=True)
        for i in range(len(x_cpu)):
            y_cpu = x_cpu[i].cpu().bincount().unsqueeze(
                0).float().renorm(1, 0, 1)
            y_dpcpp = x_dpcpp[i].cpu().bincount().unsqueeze(
                0).float().renorm(1, 0, 1)
            self.assertEqual(y_cpu, y_dpcpp)
        # print("x[0] replacement=True", x[0].cpu(
        # ).bincount().unsqueeze(0).float().renorm(1, 0, 1))
        # print("x[200] replacement=True", x[200].cpu(
        # ).bincount().unsqueeze(0).float().renorm(1, 0, 1))
        # print("x[999] replacement=True", x[999].cpu(
        # ).bincount().unsqueeze(0).float().renorm(1, 0, 1))

        x_cpu2 = torch.multinomial(weights, 4, replacement=False)
        x_cpu2 = x_cpu2.transpose(0, 1)
        x_dpcpp2 = torch.multinomial(weights_dpcpp, 4, replacement=False)
        x_dpcpp2 = x_cpu2.transpose(0, 1).to(cpu_device)
        for i in range(len(x_cpu2)):
            y_cpu2 = x_cpu2[i].bincount().unsqueeze(0).float().renorm(1, 0, 1)
            y_dpcp2 = x_dpcpp2[i].bincount().unsqueeze(
                0).float().renorm(1, 0, 1)

        # print("x.transpose(0,1)[0] replacement=False",
        # 	x[0].bincount().unsqueeze(0).float().renorm(1, 0, 1))

# TODO: test full bincount distribution over all trails with no replacement
# weights = weights.unsqueeze(0).renorm(1, 0, 1).squeeze()

# n_weights = (1- weights)
# print("1- weights", n_weights)
# weights[torch.argmax(weights)] = 0
# w0 = weights[1]*renorm_1d(remove_w(weights, 1))[0] + weights[2]*renorm_1d(remove_w(weights, 2))[0] + weights[3]*renorm_1d(remove_w(weights, 3))[0]
# w1 = weights[0]*renorm_1d(remove_w(weights, 0))[1] + weights[2]*renorm_1d(remove_w(weights, 2))[1] + weights[3]*renorm_1d(remove_w(weights, 3))[1]
# w2 = weights[0]*renorm_1d(remove_w(weights, 0))[2] + weights[1]*renorm_1d(remove_w(weights, 1))[2] + weights[3]*renorm_1d(remove_w(weights, 3))[2]
# w3 = weights[0]*renorm_1d(remove_w(weights, 0))[3] + weights[1]*renorm_1d(remove_w(weights, 1))[3] + weights[2]*renorm_1d(remove_w(weights, 2))[3]
# w1 = weights[1]*weights[0] + weights[1]*weights[2] + weights[1]*weights[3]
# w2 = weights[2]*weights[0] + weights[2]*weights[1] + weights[2]*weights[3]
# w3 = weights[3]*weights[0] + weights[3]*weights[1] + weights[3]*weights[2]

# for i in range(x.size()[0]):
#	 # a = weights*weights[0]
#	 # b = weights*weights[1]
#	 # c = weights*weights[2]
#	 # d = weights*weights[3]
#	 # print("a", a)
#	 # print("b", b)
#	 # print("c", c)
#	 # print("d", d)
#	 w0 = chose_prob(weights, i, 0, 4)
#	 w1 = chose_prob(weights, i, 1, 4)
#	 w2 = chose_prob(weights, i, 2, 4)
#	 w3 = chose_prob(weights, i, 3, 4)
#	 print("r ", w0, w1, w2, w3)
#	 print("x", x[i].bincount().unsqueeze(0).float().renorm(1, 0, 1))
#	 #
#	 # print("weights", weights)
# print("x", x.float().renorm(1, 0, 1))

# x = torch.multinomial(weights, 4, replacement=False)
# print("x replacement=False", x.to(cpu_device))
# torch.multinomial(weights, 4) # ERROR!
# y = torch.multinomial(weights, 4, replacement=True)
#
# print("numpy ", np_data)
# plt.hist(np_data)
# plt.show()
