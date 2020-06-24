import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import numpy as np
np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_index_select(self, dtype=torch.float):

        dim_size = 10
        dims = 3

        def _test_index_select(input, indcies):

            def _test(input, indcies, dim):
                y_cpu = torch.index_select(input, dim, indices)
                y_dpcpp = torch.index_select(
                    input.to(dpcpp_device), dim, indcies.to(dpcpp_device))
                print(y_dpcpp.size())
                print(y_dpcpp.cpu())
                self.assertEqual(y_cpu, y_dpcpp.cpu())
            _test(input, indcies, 0)
            _test(input, indcies, 1)
            _test(input, indcies, 2)
            _test(input, indcies, 3)

        # x = torch.linspace(0, dim_size ** dims - 1, steps=dim_size ** dims, dtype=torch.double,
        #					device=dpcpp_device).view([dim_size for d in range(dims)])

        x = torch.linspace(0, 6*7*8*9 - 1, steps=6*7*8*9).view(6, 7, 8, 9)
        indices = torch.LongTensor([0, 2])

        _test_index_select(x, indices)

        # input transpose

        _test_index_select(torch.transpose(x, 0, 1), indices)

        _test_index_select(torch.transpose(x, 0, 2), indices)

        _test_index_select(torch.transpose(x, 0, 3), indices)

        _test_index_select(torch.transpose(x, 1, 2), indices)

        _test_index_select(torch.transpose(x, 1, 3), indices)

        _test_index_select(torch.transpose(x, 2, 3), indices)

# indcies transposed
# test_index_select(x, torch.transpose(indices, 0, 1))


# # extra word embedding test
#
# print("extra word embedding test")
# print("cpu")
# # an Embedding module containing 10 tensors of size 3
# embedding = nn.Embedding(30522, 765)
# print(embedding.weight)
#
# # a batch of 2 samples of 4 indices each
# input = torch.LongTensor([101])
# res = embedding(input)
#
# print(res[0, 0:10])
#
# print("dpcpp")
# embedding.dpcpp()
#
# res = embedding(input.to(dpcpp_device))
# print(res.cpu()[0, 0:10])
#
# # test index select on bool tensor
# x = torch.randn([6, 7, 8, 9], dtype=torch.float, device=dpcpp_device)
# x = x.gt(0)
#
# test_index_select(x, indices)
