import torch
from torch import nn

import torch_ipex

# tensor_dev = torch.device("cpu")
tensor_dev = torch.device("dpcpp")

dim_size = 10
dims = 3

print(tensor_dev)

def test_index_select(input, indcies):

    def _test(input, indcies, dim):
        y = torch.index_select(input, dim, indices)
        print(y.size())
        print(y.cpu())

    _test(input, indcies, 0)
    _test(input, indcies, 1)
    _test(input, indcies, 2)
    _test(input, indcies, 3)

# x = torch.linspace(0, dim_size ** dims - 1, steps=dim_size ** dims, dtype=torch.double,
#                    device=tensor_dev).view([dim_size for d in range(dims)])

x = torch.linspace(0, 6*7*8*9 - 1, steps=6*7*8*9, dtype=torch.double,
                   device=tensor_dev).view(6, 7, 8, 9)


# print(x.size())
# print(x.cpu())

indices = torch.LongTensor([0, 2]).to(tensor_dev)

test_index_select(x, indices)

# input transpose

test_index_select(torch.transpose(x, 0, 1), indices)

test_index_select(torch.transpose(x, 0, 2), indices)

test_index_select(torch.transpose(x, 0, 3), indices)

test_index_select(torch.transpose(x, 1, 2), indices)

test_index_select(torch.transpose(x, 1, 3), indices)

test_index_select(torch.transpose(x, 2, 3), indices)

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
# res = embedding(input.to(tensor_dev))
# print(res.cpu()[0, 0:10])
# 
# # test index select on bool tensor
# x = torch.randn([6, 7, 8, 9], dtype=torch.float, device=tensor_dev)
# x = x.gt(0)
# 
# test_index_select(x, indices)
