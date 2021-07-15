import math
import random
import unittest
import itertools
import copy
import os

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import intel_pytorch_extension as ipex
from torch.testing._internal.common_utils import TestCase
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD

class TestPrepackCases(TestCase):
    def _test_convolution_training_base(self, dim):
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [1, 2], [1, 4])
        for bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)

            model = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).float().train()

            model = model.to(memory_format=torch.channels_last)
            for dtype in [torch.float32, torch.bfloat16]:
                x = x.to(memory_format=torch.channels_last)
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                x3 = x.clone().requires_grad_()
                origin_model1 = copy.deepcopy(model).train()
                origin_optimizer1 = SGD(origin_model1.parameters(), lr=0.01, momentum=0.9)
                origin_model2 = copy.deepcopy(model).train()
                origin_optimizer2 = SGD(origin_model2.parameters(), lr=0.01, momentum=0.9)
                conf = ipex.AmpConf(dtype)
                ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
                # inplace case
                ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True)
                self.assertTrue(ipex_model1.weight.dtype == dtype)
                self.assertTrue(ipex_model2.weight.dtype == dtype)

                with ipex.amp.autocast(enabled=True, configure=conf):
                    # original path
                    y1 = origin_model1(x1)
                    loss1 = y1.sum()
                    origin_optimizer1.zero_grad()
                    loss1.backward()
                    origin_optimizer1.step()
                    # ipex path with inplace=False
                    y2 = ipex_model1(x2)
                    loss2 = y2.sum()
                    ipex_optimizer1.zero_grad()
                    loss2.backward()
                    ipex_optimizer1.step()
                    # ipex path with inplace=True
                    y3 = ipex_model2(x3)
                    loss3 = y3.sum()
                    ipex_optimizer2.zero_grad()
                    loss3.backward()
                    ipex_optimizer2.step()

                self.assertEqual(y1, y2)
                self.assertEqual(y1, y3)
                self.assertEqual(x1.grad, x2.grad)
                self.assertEqual(x1.grad, x3.grad)
                if bias:
                    self.assertEqual(origin_model1.bias.grad, ipex_model1.bias.grad)
                    self.assertEqual(origin_model1.bias.grad, ipex_model2.bias.grad)

                # compare origin_model parameters with origin_model parameters after grad updata
                origin_model_state = origin_model1.state_dict()
                ipex_model_state1 = ipex_model1.state_dict()
                ipex_model_state2 = ipex_model2.state_dict()
                for var_name in origin_model_state:
                    self.assertEqual(origin_model_state[var_name], ipex_model_state1[var_name])
                    self.assertEqual(origin_model_state[var_name], ipex_model_state2[var_name])

                # compare momentum_buffer in optimizer's state(sgd)
                # TODO: other optimizer.
                origin_optimizer_state = origin_optimizer1.state_dict()
                ipex_optimizer_state1 = ipex_optimizer1.state_dict()
                ipex_optimizer_state2 = ipex_optimizer2.state_dict()

                for var_name in origin_optimizer_state:
                    if var_name == 'state':
                        self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state1[var_name])
                        self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state2[var_name])

    def test_conv2d(self):
        self._test_convolution_training_base(dim = 2)
        # TODO: add inference case.

    def test_conv2d_nc11(self):
        # related issue: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/pull/86.
        model = torch.nn.Conv2d(256, 324, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        model = model.to(memory_format=torch.channels_last).train()
        x = torch.randn(32, 256, 1, 1).to(memory_format=torch.channels_last)
        for dtype in [torch.float32, torch.bfloat16]:
            conf = ipex.AmpConf(dtype)
            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            x3 = x.clone().requires_grad_()
            origin_model1 = copy.deepcopy(model).train()
            origin_optimizer1 = SGD(origin_model1.parameters(), lr=0.01, momentum=0.9)
            origin_model2 = copy.deepcopy(model).train()
            origin_optimizer2 = SGD(origin_model2.parameters(), lr=0.01, momentum=0.9)
            ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
            ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True)
            with ipex.amp.autocast(enabled=True, configure=conf):
                # train one step for origin.
                y1 = origin_model1(x1)
                loss1 = y1.sum()
                origin_optimizer1.zero_grad()
                loss1.backward()
                origin_optimizer1.step()
                # train one step for ipex with inplace=False
                y2 = ipex_model1(x2)
                loss2 = y2.sum()
                ipex_optimizer1.zero_grad()
                loss2.backward()
                ipex_optimizer1.step()
                # train one step for ipex with inplace=False
                y3 = ipex_model2(x3)
                loss3 = y3.sum()
                ipex_optimizer2.zero_grad()
                loss3.backward()
                ipex_optimizer2.step()

            self.assertEqual(y1, y2)
            self.assertEqual(y1, y3)
            self.assertEqual(x1.grad, x2.grad)
            self.assertEqual(x1.grad, x3.grad)
            # compare origin_model parameters with origin_model parameters after grad updata
            origin_model_state = origin_model1.state_dict()
            ipex_model_state1 = ipex_model1.state_dict()
            ipex_model_state2 = ipex_model2.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state1[var_name])
                self.assertEqual(origin_model_state[var_name], ipex_model_state2[var_name])

            # compare momentum_buffer in optimizer's state(sgd)
            # TODO: other optimizer.
            origin_optimizer_state = origin_optimizer1.state_dict()
            ipex_optimizer_state1 = ipex_optimizer1.state_dict()
            ipex_optimizer_state2 = ipex_optimizer2.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state1[var_name])
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state2[var_name])

    def test_model_serialization(self):
        model = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = model.to(memory_format=torch.channels_last).train()

        x = torch.randn(64, 3, 224, 224).to(memory_format=torch.channels_last)
        optimizer_options = [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]
        options = itertools.product([torch.float32, torch.bfloat16], optimizer_options)
        for dtype, optimizer in options:
            conf = ipex.AmpConf(dtype)
            origin_x = x.clone()
            ipex_x = x.clone()
            origin_model = copy.deepcopy(model).train()
            origin_optimizer = optimizer(origin_model.parameters(), lr=0.01)
            ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            with ipex.amp.autocast(enabled=True, configure=conf):
                # train one step for origin.
                y1 = origin_model(origin_x)
                loss1 = y1.sum()
                origin_optimizer.zero_grad()
                loss1.backward()
                origin_optimizer.step()
                # train one step for ipex.
                y2 = ipex_model(ipex_x)
                loss2 = y2.sum()
                ipex_optimizer.zero_grad()
                loss2.backward()
                ipex_optimizer.step()
            torch.save({'model_state_dict': origin_model.state_dict(),
                        'optimizer_state_dict': origin_optimizer.state_dict()
                        }, 'origin_checkpoint.pth')
            torch.save({'model_state_dict': ipex_model.state_dict(),
                        'optimizer_state_dict': ipex_optimizer.state_dict()
                        }, 'ipex_checkpoint.pth')
            self.assertEqual(y1, y2)
            self.assertEqual(loss1, loss2)
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])
            origin_model1 = copy.deepcopy(model).train()
            origin_optimizer1 = optimizer(origin_model1.parameters(), lr=0.01)
            origin_checkpoint = torch.load('origin_checkpoint.pth')
            origin_model1.load_state_dict(origin_checkpoint['model_state_dict'])
            origin_optimizer1.load_state_dict(origin_checkpoint['optimizer_state_dict'])
            origin_model2 = copy.deepcopy(model)
            origin_optimizer2 = optimizer(origin_model2.parameters(), lr=0.01)
            ipex_checkpoint = torch.load('ipex_checkpoint.pth')
            origin_model2.load_state_dict(ipex_checkpoint['model_state_dict'])
            origin_optimizer2.load_state_dict(ipex_checkpoint['optimizer_state_dict'])
            self.assertEqual(origin_model1.weight, origin_model2.weight)
            # check momentum_buffer works.
            ipex_model, ipex_optimizer = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
            with ipex.amp.autocast(enabled=True, configure=conf):
                # train second step for origin.
                y1 = origin_model1(origin_x)
                loss1 = y1.sum()
                origin_optimizer1.zero_grad()
                loss1.backward()
                origin_optimizer1.step()
                # train second step for origin using ipex checkpoint.
                y2 = origin_model2(origin_x)
                loss2 = y2.sum()
                origin_optimizer2.zero_grad()
                loss2.backward()
                origin_optimizer2.step()
                # traing second step for ipex model.
                y3 = ipex_model(origin_x)
                loss3 = y3.sum()
                ipex_optimizer.zero_grad()
                loss3.backward()
                ipex_optimizer.step()
            self.assertEqual(y1, y2)
            self.assertEqual(y1, y3)
            self.assertEqual(loss1, loss2)
            self.assertEqual(loss1, loss3)
            origin_model_state1 = origin_model1.state_dict()
            origin_model_state2 = origin_model2.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state1[var_name], origin_model_state2[var_name])
                self.assertEqual(origin_model_state1[var_name], ipex_model_state[var_name])
            os.remove('origin_checkpoint.pth')
            os.remove('ipex_checkpoint.pth')

    def _test_imagenet_model(self, model):
        model = model.to(memory_format=torch.channels_last)
        for dtype in [torch.float32, torch.bfloat16]:
            # inference case, will do conv+bn folding for 'O0' and 'O1'. will do weight' prepack for 'O1'.
            ipex_model1 = ipex.optimize(model.eval(), dtype=dtype, level='O0')
            ipex_model2 = ipex.optimize(model.eval(), dtype=dtype, level='O1')
            x = torch.randn(32, 3, 224, 224).to(memory_format=torch.channels_last)
            conf = ipex.AmpConf(dtype)
            with ipex.amp.autocast(enabled=True, configure=conf):
                y1 = ipex_model1(x)
                y2 = ipex_model2(x)
            self.assertEqual(y1, y2)
            # traing case.
            conf = ipex.AmpConf(dtype)
            origin_model = copy.deepcopy(model).train()
            origin_optimizer = SGD(origin_model.parameters(), lr=0.01, momentum=0.9)
            # do nothing for 'O0'
            ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O0')
            # do weight prepack for 'O1'
            ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            # run two iterations, and then compare the results.

            xx = [torch.randn(32, 3, 224, 224), torch.randn(32, 3, 224, 224)]
            for i in range(2):
                with ipex.amp.autocast(enabled=True, configure=conf):
                    x = xx[i]
                    # original case
                    y = origin_model(x)
                    loss = y.sum()
                    origin_optimizer.zero_grad()
                    loss.backward()
                    origin_optimizer.step()
                    # ipex case1.
                    y1 = ipex_model1(x)
                    loss1 = y1.sum()
                    ipex_optimizer1.zero_grad()
                    loss1.backward()
                    ipex_optimizer1.step()
                    # ipex case2.
                    y2 = ipex_model2(x)
                    loss2 = y2.sum()
                    ipex_optimizer2.zero_grad()
                    loss2.backward()
                    ipex_optimizer2.step()
            self.assertEqual(y, y1)
            self.assertEqual(y1, y2, rtol=1e-5, atol=1e-3)
            self.assertEqual(loss, loss1)
            self.assertEqual(loss1, loss2, rtol=1e-5, atol=1e-3)


    @skipIfNoTorchVision
    def test_resnet18(self):
        model = torchvision.models.resnet.resnet18(pretrained=False)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
        self._test_imagenet_model(model)

    def test_linear_inference(self):
        class L(torch.nn.Module):
            def __init__(self, in_f, out_f):
                super(L, self).__init__()
                self.linear = torch.nn.Linear(in_f, out_f)

            def forward(self, x):
                return self.linear(x)

        out_features = torch.randint(3, 10, (1,)).item()
        in_features = torch.randint(3, 10, (1,)).item()

        input_shapes = [(8, in_features), (2, 4, in_features), (2, 2, 2, in_features)]
        options = itertools.product([True, False], input_shapes)
        for bias, x_shape in options:
            x = torch.randn(x_shape, dtype=torch.float32)
            model = L(in_features, out_features).float().eval()
            for dtype in [torch.float32, torch.bfloat16]:
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                origin_model = copy.deepcopy(model).eval()
                conf = ipex.AmpConf(dtype)
                ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1')
                self.assertEqual(ipex_model.linear.weight.dtype, dtype)
                ipex_model = ipex_model.eval()
                with ipex.amp.autocast(enabled=True, configure=conf):
                    # original path
                    y1 = origin_model(x1)
                    loss1 = y1.sum()
                    # ipex path
                    y2 = ipex_model(x2)
                    loss2 = y2.sum()
                self.assertEqual(y1, y2)
                self.assertEqual(loss1, loss2)

    def test_linear_training(self):
        linear_module = torch.nn.Linear
        out_feature = [1024, 256, 1, torch.randint(3, 10, (1, )).item()]
        in_feature = [128, 479, torch.randint(3, 10, (1, )).item()]
        input_shapes=[]
        for s in in_feature:
            input_shapes += [(128, s), (2, 64, s), (2, 2, 32, s)]
        options = itertools.product(out_feature, [True, False], input_shapes)
        for out_features, bias, x_shape in options:
            in_features = x_shape[-1]
            x = torch.randn(x_shape, dtype=torch.float32)
            model = torch.nn.Linear(in_features, out_features).float().train()
            for dtype in [torch.float32, torch.bfloat16]:
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                origin_model = copy.deepcopy(model).train()
                origin_optimizer = SGD(origin_model.parameters(), lr=0.01, momentum=0.9)
                conf = ipex.AmpConf(dtype)
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
                self.assertTrue(ipex_model.weight.dtype == dtype)
                for i in range(2):
                    with ipex.amp.autocast(enabled=True, configure=conf):
                        # original path
                        y1 = origin_model(x1)
                        loss1 = y1.sum()
                        origin_optimizer.zero_grad()
                        loss1.backward()
                        origin_optimizer.step()
                        # ipex path
                        y2 = ipex_model(x2)
                        loss2 = y2.sum()
                        ipex_optimizer.zero_grad()
                        loss2.backward()
                        ipex_optimizer.step()
                self.assertEqual(y1, y2)
                self.assertEqual(loss1, loss2)
                self.assertEqual(x1.grad, x2.grad, rtol=1e-5, atol=1e-3)
                if bias:
                    self.assertEqual(origin_model.bias.grad, ipex_model.bias.grad)
                # compare origin_model parameters with origin_model parameters after grad updata
                origin_model_state = origin_model.state_dict()
                ipex_model_state = ipex_model.state_dict()
                for var_name in origin_model_state:
                    self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])
                # compare momentum_buffer in optimizer's state(sgd)
                # TODO: other optimizer.
                origin_optimizer_state = origin_optimizer.state_dict()
                ipex_optimizer_state = ipex_optimizer.state_dict()
                for var_name in origin_optimizer_state:
                    if var_name == 'state':
                        self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-5, atol=1e-3)

if __name__ == '__main__':
    test = unittest.main()
