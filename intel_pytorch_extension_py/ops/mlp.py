import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
import _torch_ipex as core

class IpexMLPHandle:
    def __init__(self, N, C, K, bn, bc, bk, dtype, fuse_bias, act_type):
        self.handle = core.mlp_create_handle(N, C, K, bn, bc, bk, 1 if dtype == torch.float32 else 2, fuse_bias, act_type)
        self.N = N
        self.C = C
        self.K = K
        self.bn = bn
        self.bc = bc
        self.bk = bk
        self.fuse_bias = fuse_bias
        self.act_type = act_type
        if act_type == 1:
            self.relu_mask_tensor = core.mlp_set_relu_mask(self.handle)

    def __del__(self):
        if self.handle: 
            core.mlp_release_handle(self.handle)
            self.handle = None
            self.relu_mask_tensor = None

class IpexMLPFC(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, handle):
        #print("Inside XsmmFCForward")
        #t1 = time.time()
        input = input.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        output = core.mlp_forward(handle.handle, input, weight, bias)
        #t2 = time.time()
        #print("XsmmFCFWD: q=%.3f" % ((t2-t1)*1000.0))
        ctx.ipex_mlp_handle = handle
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #print("Inside XsmmFCBackward")
        handle = ctx.ipex_mlp_handle
        del ctx.ipex_mlp_handle
        input, weight = ctx.saved_variables
        #t1 = time.time()
        grad_output = grad_output.contiguous()
        grad_input, grad_weight, grad_bias = core.mlp_backward(handle.handle, grad_output, input, weight)
        #t2 = time.time()
        #print("XsmmFCBWD: q=%.3f w=%.3f" % ((t2-t1)*1000.0, (t3-t2)*1000.0))
        return (grad_input, grad_weight, grad_bias, None)

class IpexMLPLinear(nn.Module):
    r"""PCL Linear module for using libxsmm blocked GEMM"""

    __constants__ = ['bias', 'C', 'K']

    def __init__(self, C, K, bias=True, act_type=None, output_stays_blocked=True, default_blocking=None):
        super(IpexMLPLinear, self).__init__()
        self.C = C
        self.K = K
        self.bc = 0 #self.get_blocking_factor(C, default_blocking) # 64 if C % 64 == 0 else C
        self.bk = 0 #self.get_blocking_factor(K, default_blocking) # 64 if K % 64 == 0 else K
        self.nbc = 0 # C // self.bc
        self.nbk = 0 # K // self.bk
        self.C_pad = 0
        self.padded_C = self.C
        self.N = 0
        self.nbn = 0
        self.bn = 0
        self.default_blocking = default_blocking
        self.ipex_mlp_handle = None
        self.set_activation_type(act_type)
        self.output_stays_blocked = output_stays_blocked
        self.weight = Parameter(torch.Tensor(K, C))

        if bias:
            self.bias = Parameter(torch.Tensor(K))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def set_activation_type(self, act_type):
        if not act_type:
            self.act_type = 0
        elif act_type == 'relu':
            self.act_type = 1
        elif act_type == 'sigmoid':
            self.act_type = 2
        else:
            raise RuntimeError("XsmmLinear: Unknown activation type %s" % act_type)

    def get_blocking_factor(self, dim_size, default_blocking=None):
        blocking_prio_list = [64, 48, 32, 50]
        if default_blocking:
            blocking_prio_list = [default_blocking] + blocking_prio_list
        for bs in blocking_prio_list:
            if dim_size % bs == 0: 
                #print("Returning block size of %d for dim_size of %d" % ( bs, dim_size))
                return bs
        #print("Returning block size of %d for dim_size of %d" % ( dim_size, dim_size))
        return dim_size

    def is_dtype_supported(self, dtype):
        if dtype == torch.float32:
            return True
        elif dtype == torch.bfloat16 and self.C % 2 == 0:
            return True
        else:
            return False

    def maybe_pad_input(self, input):
        if input.dim() == 2 and input.size(1) != self.padded_C:
            input = torch.cat([input, input.new_zeros([input.size(0), self.C_pad])], dim=1)
        return input

    def maybe_pad_weight(self, weight):
        if weight.dim() == 2 and weight.size(1) != self.padded_C:
            weight = torch.cat([weight, weight.new_zeros([self.K, self.C_pad])], dim=1)
        # elif weight.dim() == 4 and weight.size(1) * weight.size(2) != self.padded_C:
        #     raise RuntimeError("Trying to ad 4D weights")
        # elif weight.dim() == 5 and weight.size(1) * weight.size(2) * weight.size(4) != self.padded_C:
        #     raise RuntimeError("Trying to ad 5D weights")
        return weight

    def get_blocked_weight(self, to_dtype=None, block_for_dtype=None):
        weight = self.weight
        new_weight = None
        if to_dtype:
            weight = weight.to(to_dtype)
        if not block_for_dtype:
            block_for_dtype = weight.dtype
        if self.bc == 0 or self.bk == 0:
            self.update_blocking(block_for_dtype)

        weight = self.maybe_pad_weight(weight)
        if weight.dim() == 2:
            if block_for_dtype == torch.bfloat16:
                l_view = [self.nbk, self.bk, self.nbc, self.bc // 2, 2]
                l_perm = [0, 2, 3, 1, 4]
                new_weight = weight.view(l_view).permute(l_perm).contiguous()
            elif block_for_dtype == torch.float32:
                l_view = [self.nbk, self.bk, self.nbc, self.bc]
                l_perm = [0, 2, 3, 1]
                new_weight = weight.view(l_view).permute(l_perm).contiguous()
            else:
                raise RuntimeError("Invalid datatype for blocking: %s" % block_for_dtype)
        elif weight.dim() == 4:
            if block_for_dtype == torch.bfloat16:
                l_view = [self.nbk, self.nbc, self.bc // 2, 2, self.bk]
                l_perm = [0, 1, 2, 4, 3]
                new_weight = weight.view(l_view).permute(l_perm).contiguous()
            elif block_for_dtype == torch.float32:
                # We are already in correct format, do nothing
                new_weight = weight
            else:
                raise RuntimeError("Invalid datatype for blocking: %s" % block_for_dtype)
        elif weight.dim() == 5:
            if block_for_dtype == torch.bfloat16:
                # We are already in correct format, do nothing
                new_weight = weight
            elif block_for_dtype == torch.float32:
                l_view = [self.nbk, self.nbc, self.bc, self.bk]
                l_perm = [0, 1, 2, 4, 3]
                new_weight = weight.permute(l_perm).view(l_view).contiguous()
            else:
                raise RuntimeError("Invalid datatype for blocking: %s" % block_for_dtype)

        return new_weight

    def update_blocking(self, dtype):
        if dtype == torch.bfloat16 and self.padded_C % 2 != 0:
            self.C_pad = 1
            self.padded_C = self.C + self.C_pad
        self.bc = self.get_blocking_factor(self.padded_C, self.default_blocking)
        if dtype == torch.bfloat16 and self.bc % 2 != 0: self.bc *= 2
        self.nbc = self.padded_C // self.bc
        self.bk = self.get_blocking_factor(self.K, self.default_blocking)
        self.nbk = self.K // self.bk

    def reset_weight_shape(self, block_for_dtype=None):
        #if not self.is_dtype_supported(block_for_dtype):
        #    block_for_dtype = torch.float32
        #self.update_bc(block_for_dtype)
        self.weight = Parameter(self.get_blocked_weight(block_for_dtype=block_for_dtype))
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.C)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input_type = input.dtype
        #if not self.is_dtype_supported(input_type):
        #    input = input.to(torch.float32)
        if self.bc == 0 or self.bk == 0:
            self.update_blocking(input_type)
        input = self.maybe_pad_input(input)
        if input.dtype == torch.bfloat16:
            if self.bc % 2 != 0: raise RuntimeError("Bfloat16 requires even bc")

        if input.dim() == 2:
            N = input.size(0)
            bn = self.get_blocking_factor(N, 48) #64 if N % 64 == 0 else N
            input = input.view(N//bn, bn, self.nbc, self.bc).permute(0,2,1,3)
        elif input.dim() == 4:
            N = input.size(0) * input.size(2)
            bn = input.size(2)
        else:
            print("Invalid Input dimensions (%d)" % input.dim())

        input = input.contiguous()    

        if N != self.N or bn != self.bn:
            # print("Create handle: ", N, self.padded_C, self.K, bn, self.bc, self.bk, input.dtype, 0 if self.bias is None else 1, self.act_type)
            self.ipex_mlp_handle = IpexMLPHandle(N, self.padded_C, self.K, bn, self.bc, self.bk, input.dtype, 0 if self.bias is None else 1, self.act_type)
            self.N = N
            self.bn = bn
            self.nbn = N // bn
        
        wtensor = self.get_blocked_weight(to_dtype=input.dtype)
        btensor = self.bias.to(input.dtype)
        output =  IpexMLPFC.apply(input, wtensor, btensor, self.ipex_mlp_handle)
        if not self.output_stays_blocked:
            #output = output.permute(0, 2, 1, 3).view(self.N, self.K).contiguous()
            output = output.permute(0, 2, 1, 3).reshape(self.N, self.K).contiguous()
        output = output.to(input_type)
        return output

    def extra_repr(self):
        return 'C={}, K={}, bias={}'.format(
            self.C, self.K, self.bias is not None
        )
