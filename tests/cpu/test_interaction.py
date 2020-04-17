# numpy
import torch
import intel_pytorch_extension as ipex
# import pcl_embedding_bag
# import time

def interact_fusion(x, ly):
    A = [x] + ly
    R = ipex.interaction(*A)
    return R

def interact_features(x, ly):
    (batch_size, d) = x.shape
    T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
    # Z = pcl_embedding_bag.bdot(T)
    Z = torch.bmm(T, torch.transpose(T, 1, 2))
    _, ni, nj = Z.shape
    offset =  0
    li = torch.tensor([i for i in range(ni) for j in range(i + offset)], device='dpcpp')
    lj = torch.tensor([j for i in range(nj) for j in range(i + offset)], device='dpcpp')
    Zflat = Z[:, li, lj]
    # concatenate dense features and interactions
    R = torch.cat([x] + [Zflat], dim=1)
    return R

def run(dtype='float32'):
    print("##################### testing with %s"% str(dtype))
    x1 = torch.randn([2048, 128], device='dpcpp').to(dtype).clone().detach().requires_grad_()
    x2 = x1.clone().detach().requires_grad_()
    ly1 = []
    ly2 = []
    for i in range(0, 26):
        V = torch.randn([2048, 128], device='dpcpp').to(dtype).clone().detach().requires_grad_()
        ly1.append(V)
        ly2.append(V.clone().detach().requires_grad_())

    print("##################### interaction forward")
    A = interact_fusion(x1, ly1)
    B = interact_features(x2, ly2)
    if(A.allclose(B, rtol=1e-5, atol=1e-5)):
        print("##################### interaction forward PASS")
    else:
        print("##################### interaction forward FAIL")

    print("##################### interaction backward")
    A.mean().backward()
    B.mean().backward()
    ret = x1.grad.allclose(x2.grad, rtol=1e-5, atol=1e-5)
    ret = ret and all(ly1[i].grad.allclose(ly2[i].grad, rtol=1e-5, atol=1e-5) for i in range(0, 26))
    if (ret):
        print("##################### interaction backward PASS")
    else:
        print("##################### interaction backward FAIL")

#dtypes=[torch.float32, torch.bfloat16]
dtypes=[torch.float32]
for d in dtypes:
    run(d)
