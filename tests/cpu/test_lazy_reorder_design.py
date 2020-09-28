"""Tests for lazy reorder."""
from __future__ import division
from __future__ import print_function
import time
import sys
import unittest
import torch
import intel_pytorch_extension as ipex

ipex_device = ipex.DEVICE
def get_rand_seed():
    return int(time.time() * 1000000000)


def test_conv_add_relu_000():         ### 2 reorder 
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    ipex.core.enable_auto_dnnl()
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device="cpu")
    conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device="cpu")
    conv_op_output = conv_op(conv_op_input)
    add_src = torch.rand((1, 1, 4, 4)).to(device="cpu")
    conv_op_output += add_src
    conv_op_output.relu_()
    ipex.core.disable_auto_dnnl()
def test_conv_add_relu_111():         ### 1 reorder 
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    ipex.core.enable_auto_dnnl()
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_output = conv_op(conv_op_input)
    add_src = torch.rand((1, 1, 4, 4)).to(device=ipex_device)
    conv_op_output += add_src
    conv_op_output.relu_()
    ipex.core.disable_auto_dnnl()

def test_conv_add_bn_110():    ##2 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    conv_op_output = conv_op(conv_op_input)
    add_src = torch.rand((1, 1, 4, 4)).to(device=ipex_device)
    conv_op_output += add_src
    bn_op=torch.nn.BatchNorm2d(1).to(device="cpu")
    bn_op_output=bn_op(conv_op_output)
    ipex.core.disable_auto_dnnl()

def test_conv_bn_add_101():  ##2 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    conv_op_output = conv_op(conv_op_input)
    bn_op=torch.nn.BatchNorm2d(1).to(device="cpu")
    bn_op_output=bn_op(conv_op_output)
    add_src = torch.rand((1, 1, 4, 4)).to(device=ipex_device)
    bn_op_output += add_src
    ipex.core.disable_auto_dnnl()
def test_bn_conv_add_011():  ##1 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    bn_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    bn_op=torch.nn.BatchNorm2d(1).to(device="cpu")
    bn_op_output=bn_op(bn_op_input)

    conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_output = conv_op(bn_op_output)

    add_src = torch.rand((1, 1, 4, 4)).to(device=ipex_device)
    conv_op_output += add_src
    ipex.core.disable_auto_dnnl()
def test_conv_bn_pool_100():   ##2reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    conv_op_output = conv_op(conv_op_input)
    bn_op=torch.nn.BatchNorm2d(1).to(device="cpu")
    bn_op_output=bn_op(conv_op_output)
    pool_op=torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1).to(device="cpu")
    pool_op_output=pool_op(bn_op_output)
    ipex.core.disable_auto_dnnl()
def test_bn_conv_pool_010():   ##1 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    bn_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    bn_op=torch.nn.BatchNorm2d(1).to(device="cpu")
    bn_op_output=bn_op(bn_op_input)


    conv_op = torch.nn.Conv2d(1, 1, (3, 3)).to(device=ipex_device)
    conv_op_output = conv_op(bn_op_output)

    pool_op=torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1).to(device="cpu")
    pool_op_output=pool_op(conv_op_output)
    ipex.core.disable_auto_dnnl()
def test_bn_pool_conv_001():   ##1 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    bn_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    bn_op=torch.nn.BatchNorm2d(1).to(device="cpu")
    bn_op_output=bn_op(bn_op_input)
    pool_op=torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1).to(device="cpu")
    pool_op_output=pool_op(bn_op_output)

    conv_op = torch.nn.Conv2d(1, 1, (3, 3)).to(device=ipex_device)
    conv_op_output = conv_op(pool_op_output)
    ipex.core.disable_auto_dnnl()
def test_conv_conv_concate():   ##2 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    conv_op1 = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op2 = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    conv_op_output1 = conv_op1(conv_op_input)
    conv_op_output2 = conv_op2(conv_op_input)
    concate_out=torch.cat([conv_op_output1,conv_op_output2],dim=1).to(device=ipex_device)
    ipex.core.disable_auto_dnnl()
def test_conv_conv_add():   ##3 reorder
    ipex.core.enable_auto_dnnl()
    rand_seed = int(get_rand_seed())
    print("******{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    torch.manual_seed(rand_seed)
    conv_op1 = torch.nn.Conv2d(1, 1, (7, 7)).to(device='cpu')
    conv_op2 = torch.nn.Conv2d(1, 1, (7, 7)).to(device=ipex_device)
    conv_op_input = torch.rand((1, 1, 10, 10)).to(device=ipex_device)
    conv_op_output1 = conv_op1(conv_op_input)
    conv_op_output2 = conv_op2(conv_op_input)
    add_out=torch.add(conv_op_output1,conv_op_output2).to(device=ipex_device)
    ipex.core.disable_auto_dnnl()


if __name__ == '__main__':
    test_conv_conv_add()
