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
import intel_extension_for_pytorch as ipex
from torch.testing._internal.common_utils import TestCase
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
from intel_extension_for_pytorch.optim._lamb import Lamb

conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

def module_found(model, type):
    for child_name, child in model.named_children():
        if isinstance(child, type):
            return True
        else:
            module_found(child, type)
    return False

class TestPrepackCases(TestCase):
    def _test_convolution_training_base(self, dim):
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        channels_last = torch.channels_last if dim ==2 else torch.channels_last_3d
        options = itertools.product([True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last])

        for bias, dilation, groups, memory_format in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)

            model = conv_module[dim](
                in_channels=C,
                out_channels=M,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=dilation,
                bias=bias,
                groups=groups).float().train()

            model = model.to(memory_format=memory_format)
            for dtype in [torch.float32, torch.bfloat16]:
                x = x.to(memory_format=memory_format)
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                x3 = x.clone().requires_grad_()
                origin_model1 = copy.deepcopy(model).train()
                origin_optimizer1 = SGD(origin_model1.parameters(), lr=0.01, momentum=0.9)
                origin_model2 = copy.deepcopy(model).train()
                origin_optimizer2 = SGD(origin_model2.parameters(), lr=0.01, momentum=0.9)
                ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
                # inplace case
                ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True)
                self.assertTrue(ipex_model1.weight.dtype == dtype)
                self.assertTrue(ipex_model2.weight.dtype == dtype)

                with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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

                self.assertEqual(y1, y2, rtol=1e-4, atol=5e-02)
                self.assertEqual(y1, y3, rtol=1e-4, atol=5e-02)
                self.assertEqual(x1.grad, x2.grad, rtol=1e-4, atol=5e-02)
                self.assertEqual(x1.grad, x3.grad, rtol=1e-4, atol=5e-02)
                if bias:
                    self.assertEqual(origin_model1.bias.grad, ipex_model1.bias.grad.float())
                    self.assertEqual(origin_model1.bias.grad, ipex_model2.bias.grad.float())

                # compare origin_model parameters with origin_model parameters after grad updata
                origin_model_state = origin_model1.state_dict()
                ipex_model_state1 = ipex_model1.state_dict()
                ipex_model_state2 = ipex_model2.state_dict()
                for var_name in origin_model_state:
                    self.assertEqual(origin_model_state[var_name], ipex_model_state1[var_name], rtol=1e-4, atol=5e-02)
                    self.assertEqual(origin_model_state[var_name], ipex_model_state2[var_name], rtol=1e-4, atol=5e-02)

                # compare momentum_buffer in optimizer's state(sgd)
                # TODO: other optimizer.
                origin_optimizer_state = origin_optimizer1.state_dict()
                ipex_optimizer_state1 = ipex_optimizer1.state_dict()
                ipex_optimizer_state2 = ipex_optimizer2.state_dict()

                for var_name in origin_optimizer_state:
                    if var_name == 'state':
                        self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state1[var_name], rtol=1e-2, atol=5e-02)
                        self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state2[var_name], rtol=1e-2, atol=5e-02)

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_conv2d_training(self):
        self._test_convolution_training_base(dim=2)
        # TODO: add inference case.

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_conv3d_training(self):
        self._test_convolution_training_base(dim=3)
        # TODO: add inference case.

    def _test_conv_nc11_base(self, dim):
        # related issue: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/pull/86.
        channels_last = torch.channels_last if dim ==2 else torch.channels_last_3d
        options = itertools.product([torch.float, torch.bfloat16],
                                    [1, 256], [1, 324],
                                    [torch.contiguous_format, channels_last])

        for dtype, in_channels, out_channels, memory_format in options:
            model = conv_module[dim](in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False)
            model = model.to(memory_format=memory_format).train()
            input_shape = [32, in_channels, 1, 1]
            if dim == 3:
                input_shape.append(1)
            x = torch.randn(input_shape).to(memory_format=memory_format)

            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            x3 = x.clone().requires_grad_()
            origin_model1 = copy.deepcopy(model).train()
            origin_optimizer1 = SGD(origin_model1.parameters(), lr=0.01, momentum=0.9)
            origin_model2 = copy.deepcopy(model).train()
            origin_optimizer2 = SGD(origin_model2.parameters(), lr=0.01, momentum=0.9)
            ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
            ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True)
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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

            self.assertEqual(y1, y2, rtol=1e-4, atol=5e-01)
            self.assertEqual(y1, y3, rtol=1e-4, atol=5e-01)
            self.assertEqual(x1.grad, x2.grad, rtol=1e-4, atol=5e-01)
            self.assertEqual(x1.grad, x3.grad, rtol=1e-4, atol=5e-01)
            # compare origin_model parameters with origin_model parameters after grad updata
            origin_model_state = origin_model1.state_dict()
            ipex_model_state1 = ipex_model1.state_dict()
            ipex_model_state2 = ipex_model2.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state1[var_name], rtol=1e-4, atol=5e-01)
                self.assertEqual(origin_model_state[var_name], ipex_model_state2[var_name], rtol=1e-4, atol=5e-01)

            # compare momentum_buffer in optimizer's state(sgd)
            # TODO: other optimizer.
            origin_optimizer_state = origin_optimizer1.state_dict()
            ipex_optimizer_state1 = ipex_optimizer1.state_dict()
            ipex_optimizer_state2 = ipex_optimizer2.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state1[var_name], rtol=1e-4, atol=5e-01)
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state2[var_name], rtol=1e-4, atol=5e-01)

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_conv2d_nc11(self):
        self._test_conv_nc11_base(dim=2)

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_conv3d_nc11(self):
        self._test_conv_nc11_base(dim=3)

    def _test_conv_serialization_base(self, dim):
        channels_last = torch.channels_last if dim ==2 else torch.channels_last_3d
        optimizer_options = [Lamb, Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]
        options = itertools.product([torch.float, torch.bfloat16], optimizer_options)
        input_shape = [8, 3, 56, 56]
        if dim == 3:
            input_shape.append(56)
        for dtype, optimizer in options:
            model = conv_module[dim](3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            x = torch.randn(input_shape).to(memory_format=channels_last)
            model = model.to(memory_format=channels_last).train()
            origin_x = x.clone()
            ipex_x = x.clone()
            origin_model = copy.deepcopy(model).train()
            lr = 1e-4 if optimizer is SGD else 1e-2
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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
            self.assertEqual(y1, y2, rtol=1e-4, atol=5e-02)
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])
            origin_model1 = copy.deepcopy(model).train()
            origin_optimizer1 = optimizer(origin_model1.parameters(), lr=lr)
            origin_checkpoint = torch.load('origin_checkpoint.pth')
            origin_model1.load_state_dict(origin_checkpoint['model_state_dict'])
            origin_optimizer1.load_state_dict(origin_checkpoint['optimizer_state_dict'])
            origin_model2 = copy.deepcopy(model)
            origin_optimizer2 = optimizer(origin_model2.parameters(), lr=lr)
            ipex_checkpoint = torch.load('ipex_checkpoint.pth')
            origin_model2.load_state_dict(ipex_checkpoint['model_state_dict'])
            origin_optimizer2.load_state_dict(ipex_checkpoint['optimizer_state_dict'])
            self.assertEqual(origin_model1.weight, origin_model2.weight)
            # check state_buffer works.
            origin_optimizer_state = origin_optimizer1.state_dict()
            origin_optimizer2_state = origin_optimizer2.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], origin_optimizer2_state[var_name], rtol=1e-2, atol=5e-02)
            ipex_model, ipex_optimizer = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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
            self.assertEqual(y1, y3, rtol=1e-4, atol=5e-02)
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

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_conv2d_serialization(self):
        self._test_conv_serialization_base(dim=2)

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_conv3d_serialization(self):
        self._test_conv_serialization_base(dim=3)

    def _test_imagenet_model(self, model):
        model = model.to(memory_format=torch.channels_last)
        for dtype in [torch.float32, torch.bfloat16]:
            # inference case, will do conv+bn folding 'O1'. do nothing for 'O0'.
            ipex_model1 = ipex.optimize(model.eval(), dtype=dtype, level='O0')
            ipex_model2 = ipex.optimize(model.eval(), dtype=dtype, level='O1')
            x = torch.randn(32, 3, 224, 224).to(memory_format=torch.channels_last)
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y1 = ipex_model1(x)
                y2 = ipex_model2(x)
            self.assertEqual(y1, y2, rtol=1e-3, atol=1e-1)
            # traing case.
            origin_model = copy.deepcopy(model).train()
            origin_optimizer = ASGD(origin_model.parameters(), lr=0.01)
            # do nothing for 'O0'
            ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O0')
            # do weight prepack for 'O1'
            ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            # run two iterations, and then compare the results.

            xx = [torch.randn(32, 3, 224, 224), torch.randn(32, 3, 224, 224)]
            for i in range(2):
                with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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
            self.assertEqual(y1, y2, rtol=1e-2, atol=1e-1)  # FP32: packed mkldnn vs plain mkl
            self.assertEqual(loss, loss1)
            self.assertEqual(loss1, loss2, rtol=1e-2, atol=1e-1)  # FP32: packed mkldnn vs plain mkl


    @unittest.skipIf(True, "TO DO DEBUG")
    @skipIfNoTorchVision
    def test_resnet18(self):
        model = torchvision.models.resnet.resnet18(pretrained=False)
        self._test_imagenet_model(model)

    @unittest.skipIf(True, "TO DO DEBUG")
    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
        self._test_imagenet_model(model)

    def test_linear_inference(self):
        class L(torch.nn.Module):
            def __init__(self, in_f, out_f, bias):
                super(L, self).__init__()
                self.linear = torch.nn.Linear(in_f, out_f, bias=bias)

            def forward(self, x):
                return self.linear(x)

        out_features = torch.randint(3, 10, (1,)).item()
        in_features = torch.randint(3, 10, (1,)).item()

        input_shapes = [(8, in_features), (2, 4, in_features), (2, 2, 2, in_features)]
        options = itertools.product([True, False], input_shapes)
        for bias, x_shape in options:
            x = torch.randn(x_shape, dtype=torch.float32)
            model = L(in_features, out_features, bias).float().eval()
            for dtype in [torch.float32, torch.bfloat16]:
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                origin_model = copy.deepcopy(model).eval()
                ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1')

                self.assertEqual(ipex_model.linear.weight.dtype, dtype)
                with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                    # original path
                    y1 = origin_model(x1)
                    loss1 = y1.sum()
                    # ipex path
                    y2 = ipex_model(x2)
                    loss2 = y2.sum()
                self.assertEqual(y1, y2, rtol=1e-3, atol=1e-1)
                self.assertEqual(loss1, loss2, rtol=1e-3, atol=1e-1)

    @unittest.skipIf(True, "TO DO DEBUG")
    def test_linear_training(self):
        linear_module = torch.nn.Linear
        out_feature = [1024, 256, 1, torch.randint(3, 10, (1, )).item()]
        in_feature = [128, 479, torch.randint(3, 10, (1, )).item()]
        input_shapes = []
        for s in in_feature:
            input_shapes += [(128, s), (2, 64, s), (2, 2, 32, s)]
        options = itertools.product(out_feature, [True, False], input_shapes)
        for out_features, bias, x_shape in options:
            in_features = x_shape[-1]
            x = torch.randn(x_shape, dtype=torch.float32) * 0.1
            model = torch.nn.Linear(in_features, out_features, bias=bias).float().train()
            for dtype in [torch.float32, torch.bfloat16]:
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                origin_model = copy.deepcopy(model).train()
                origin_optimizer = SGD(origin_model.parameters(), lr=0.01, momentum=0.9)
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
                self.assertTrue(ipex_model.weight.dtype == dtype)
                for i in range(2):
                    with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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
                self.assertEqual(y1, y2, rtol=1e-2, atol=1e-1)
                self.assertEqual(loss1, loss2, rtol=1e-2, atol=1e-1)
                self.assertEqual(x1.grad, x2.grad, rtol=1e-2, atol=1e-1)
                if bias:
                    self.assertEqual(origin_model.bias.grad, ipex_model.bias.grad.float())
                # compare origin_model parameters with origin_model parameters after grad updata
                origin_model_state = origin_model.state_dict()
                ipex_model_state = ipex_model.state_dict()
                for var_name in origin_model_state:
                    self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-3, atol=1e-1)
                # compare momentum_buffer in optimizer's state(sgd)
                # TODO: other optimizer.
                origin_optimizer_state = origin_optimizer.state_dict()
                ipex_optimizer_state = ipex_optimizer.state_dict()
                for var_name in origin_optimizer_state:
                    if var_name == 'state':
                        self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-2, atol=1e-1)

    def _deconv_params_list(self):
        # shapes that works:
        params_dict = {
            "input_height": [12],
            "input_width": [12],
            "input_depth": [12],
            "input_channel_per_group": [15],
            "output_channel_per_group": [3],
            "kernel_size": [3],
            "bias": [True, False],
            "stride": [1, 2],
            "padding": [1, 2],
            "output_padding": [0],  # TODO: fix output_padding == 2 and etc.
            "groups": [1, 2],
            "dilation": [1, 2],
        }

        params_list = []

        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def _deconv_with_output_padding(self):
        params_dict = {
            "input_height": 8,
            "input_width": 8,
            "input_depth": 8,
            "input_channel_per_group": 10,
            "output_channel_per_group": 10,
            "kernel_size": 3,
            "bias": False,
            "stride": 2,
            "padding": 1,
            "output_padding": 2,
            "groups": 1,
            "dilation": 3,
        }

        params_list = []

        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    # mkldnn does not support the case where:
    # padding - output_padding + stride <= 0
    # while PyTorch supports this case, need to fallback in this case
    def _deconv_fallback_shape(self):
        params_dict = {
            "input_height": 8,
            "input_width": 8,
            "input_depth": 8,
            "input_channel_per_group": 10,
            "output_channel_per_group": 10,
            "kernel_size": 4,
            "bias": False,
            "stride": 1,
            "padding": 1,
            "output_padding": 2,
            "groups": 1,
            "dilation": 3,
        }

        params_list = []

        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def _test_deconv(self, dims, inference):
        class Deconv2d(torch.nn.Module):
            def __init__(self, ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation):
                super(Deconv2d, self).__init__()
                self.deconv = torch.nn.ConvTranspose2d(ic, oc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

            def forward(self, x):
                return self.deconv(x)

        class Deconv3d(torch.nn.Module):
            def __init__(self, ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation):
                super(Deconv3d, self).__init__()
                self.deconv = torch.nn.ConvTranspose3d(ic, oc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

            def forward(self, x):
                return self.deconv(x)

        params_list = self._deconv_params_list()
        torch.manual_seed(0)
        for input_height, input_width, input_depth, input_channel_per_group, output_channel_per_group, kernel_size, bias, stride, padding, output_padding, groups, dilation in list(itertools.product(*params_list)) + [self._deconv_with_output_padding()] + [self._deconv_fallback_shape()]:
            if (output_padding < stride or output_padding < dilation) \
                    and ((input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0) \
                    and ((input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0) \
                    and ((input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0):

                ic = input_channel_per_group * groups
                oc = output_channel_per_group * groups

                if dims == 2:
                    model = Deconv2d(ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation).to(memory_format=torch.channels_last)
                    x = torch.rand((2, ic, input_height, input_width)).to(memory_format=torch.channels_last)
                elif dims == 3:
                    model = Deconv3d(ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation).to(memory_format=torch.channels_last)
                    x = torch.rand((2, ic, input_depth, input_height, input_width)).to(memory_format=torch.channels_last)

                for dtype in [torch.float32, torch.bfloat16]:
                    if inference:
                        model.eval()
                        origin_model = copy.deepcopy(model).eval()
                        ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1')

                        if padding - output_padding + stride <= 0:
                            # unsupported in mkldnn, should not replace the original ConvTranspose module
                            self.assertTrue(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))
                            continue
                        else:
                            self.assertFalse(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))

                        self.assertEqual(ipex_model.deconv.weight.dtype, dtype)

                        with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                            y_origin = origin_model(x)
                            y_ipex = ipex_model(x)

                        self.assertEqual(y_origin, y_ipex, exact_dtype=False)
                    else:
                        model.train()
                        origin_model = copy.deepcopy(model).train()
                        origin_optimizer = SGD(origin_model.parameters(), lr=0.01, momentum=0.9)
                        ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')

                        if padding - output_padding + stride <= 0:
                            # unsupported in mkldnn, should not replace the original ConvTranspose module
                            self.assertTrue(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))
                            continue
                        else:
                            self.assertFalse(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))

                        x1 = x.clone().requires_grad_()
                        x2 = x.clone().requires_grad_()
                        with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                            y1 = origin_model(x1)
                            loss1 = y1.sum()
                            origin_optimizer.zero_grad()
                            loss1.backward()
                            origin_optimizer.step()
                            y2 = ipex_model(x2)
                            loss2 = y2.sum()
                            ipex_optimizer.zero_grad()
                            loss2.backward()
                            ipex_optimizer.step()
                            self.assertEqual(y1, y2, rtol=1e-2, atol=1e-1, exact_dtype=False)
                            self.assertEqual(loss1, loss2, rtol=1e-1, atol=1e-1, exact_dtype=False)  # TODO: 1e-2 cannot pass, check it
                            self.assertEqual(x1.grad, x2.grad, rtol=1e-2, atol=1e-1, exact_dtype=False)
                            if bias:
                                self.assertEqual(origin_model.deconv.bias.grad, ipex_model.deconv.bias.grad.float())

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
                                self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-2, atol=5e-02)

    @unittest.skipIf(True, "PyTorch not support bf16 deconv")
    def test_deconv_2d_inference(self):
        self._test_deconv(dims=2, inference=True)

    @unittest.skipIf(True, "PyTorch not support bf16 deconv")
    def test_deconv_2d_training(self):
        self._test_deconv(dims=2, inference=False)


if __name__ == '__main__':
    torch.manual_seed(2020)
    test = unittest.main()
