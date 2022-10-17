import unittest
import itertools
import copy
import os
import time
import sys
from intel_extension_for_pytorch.utils.channels_last_1d import to_channels_last_1d, is_contiguous_channels_last_1d

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core

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

def get_rand_seed():
    return int(time.time() * 1000000000)

class TestPrepackCases(TestCase):
    def _test_convolution_inference_base(self, dim):
        class ConvNd(torch.nn.Module):
            def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, groups):
                super(ConvNd, self).__init__()
                self.conv = conv_module[dim](in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

            def forward(self, x):
                return self.conv(x)
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        if dim == 2:
            channels_last = torch.channels_last
        elif dim == 3:
            channels_last = torch.channels_last_3d
        if dim == 1:
            options = itertools.product([True, False], [1, 2], [1, 4], [True, False], [torch.contiguous_format])
        else:
            options = itertools.product([True, False], [1, 2], [1, 4], [True, False], [torch.contiguous_format, channels_last])

        for bias, dilation, groups, feed_sample_input, memory_format in options:
            N = torch.randint(1, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)
            model = ConvNd(
                dim=dim,
                in_channels=C,
                out_channels=M,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=dilation,
                bias=bias,
                groups=groups).float().eval()
            model = model.to(memory_format=memory_format)
            x = x.to(memory_format=memory_format)
            if dim == 1:
                x_nwc = to_channels_last_1d(copy.deepcopy(x))
                model = to_channels_last_1d(model)
            if feed_sample_input:
                ipex_model = ipex.optimize(model, dtype=torch.float32, level='O1', sample_input=x)
            else:
                ipex_model = ipex.optimize(model, dtype=torch.float32, level='O1')
            y_ipex = ipex_model(x)
            y = model(x)
            self.assertEqual(y, y_ipex)
            if dim == 1:
                y_ipex_nwc = ipex_model(x_nwc)
                self.assertEqual(y_ipex, y_ipex_nwc)
                self.assertTrue(is_contiguous_channels_last_1d(y_ipex))
                self.assertTrue(is_contiguous_channels_last_1d(y_ipex_nwc))

    def test_conv1d_inference(self):
        self._test_convolution_inference_base(dim=1)

    def test_conv2d_inference(self):
        self._test_convolution_inference_base(dim=2)

    def test_conv3d_inference(self):
        self._test_convolution_inference_base(dim=3)

    def test_channels_last_1d_forward(self):
        class Conv1d(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
                super(Conv1d, self).__init__()
                self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

            def forward(self, x):
                return self.conv(x)

        input_shapes = [(2, 2, 3), (4, 4, 4), (4, 4, 1), (4, 1, 4), (4, 1, 1), (1, 4, 4), (1, 4, 1), (1, 1, 4)]
        for x_shape in input_shapes:
            M = 5
            C = x_shape[1]
            x = torch.randn(x_shape, dtype=torch.float32)
            model = Conv1d(
                in_channels=C,
                out_channels=M,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False).float().eval()

            x_nwc = to_channels_last_1d(copy.deepcopy(x))
            model = to_channels_last_1d(model)

            ipex_model = ipex.optimize(model, dtype=torch.float32, level='O1')
            y_ipex = ipex_model(x)
            y = model(x)
            self.assertEqual(y, y_ipex)

            y_ipex_nwc = ipex_model(x_nwc)
            y_nwc = model(x_nwc)
            self.assertEqual(y_nwc, y_ipex_nwc)
            self.assertEqual(y_ipex, y_ipex_nwc)
            self.assertTrue(is_contiguous_channels_last_1d(y_ipex))
            self.assertTrue(is_contiguous_channels_last_1d(y_ipex_nwc))

    def _test_convolution_training_base(self, dim, dtype, rtol=None, atol=None):
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        channels_last = torch.channels_last if dim ==2 else torch.channels_last_3d
        # Currently, there is no channels_last_1d format for 1d input in IPEX.
        # After adding a python API named `to_channels_last_1d` which is to convert 1d input to channels last,
        # we will make some changes to fully support channels_last_1d.
        if dim == 1:
            options = itertools.product([True, False], [1, 2], [1, 4], [torch.contiguous_format], [True, False])
        else:
            options = itertools.product([True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last], [True, False])

        for bias, dilation, groups, memory_format, feed_sample_input in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(3, 10, (1,)).item() * groups
            C = torch.randint(3, 10, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32).to(dtype=dtype).float()

            model = conv_module[dim](
                in_channels=C,
                out_channels=M,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=dilation,
                bias=bias,
                groups=groups).to(dtype=dtype).float()
            model = model.to(memory_format=memory_format)
            x = x.to(memory_format=memory_format)
            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            x3 = x.clone().requires_grad_()
            origin_model1 = copy.deepcopy(model).train()
            origin_optimizer1 = SGD(origin_model1.parameters(), lr=0.01, momentum=0.9)
            origin_model2 = copy.deepcopy(model).train()
            origin_optimizer2 = SGD(origin_model2.parameters(), lr=0.01, momentum=0.9)
            if feed_sample_input:
                ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1', sample_input=x)
                ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True, sample_input=x)
            else:
                ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
                ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True)
            self.assertTrue(ipex_model1.weight.dtype == dtype)
            self.assertTrue(ipex_model2.weight.dtype == dtype)

            # original fp32 path
            y1 = origin_model1(x1)
            loss1 = y1.sum()
            origin_optimizer1.zero_grad()
            loss1.backward()
            origin_optimizer1.step()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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

            self.assertEqual(y1, y2.float(), rtol=rtol, atol=atol)
            self.assertEqual(y1, y3.float(), rtol=rtol, atol=atol)
            self.assertEqual(x1.grad, x2.grad, rtol=rtol, atol=atol)
            self.assertEqual(x1.grad, x3.grad, rtol=rtol, atol=atol)
            if bias:
                self.assertEqual(origin_model1.bias.grad, ipex_model1.bias.grad.float(), rtol=rtol, atol=atol)
                self.assertEqual(origin_model1.bias.grad, ipex_model2.bias.grad.float(), rtol=rtol, atol=atol)

            # compare origin_model parameters with origin_model parameters after grad updata
            origin_model_state = origin_model1.state_dict()
            ipex_model_state1 = ipex_model1.state_dict()
            ipex_model_state2 = ipex_model2.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state1[var_name], rtol=rtol, atol=atol)
                self.assertEqual(origin_model_state[var_name], ipex_model_state2[var_name], rtol=rtol, atol=atol)
            # compare momentum_buffer in optimizer's state(sgd)
            # TODO: other optimizer.
            origin_optimizer_state = origin_optimizer1.state_dict()
            ipex_optimizer_state1 = ipex_optimizer1.state_dict()
            ipex_optimizer_state2 = ipex_optimizer2.state_dict()

            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state1[var_name], rtol=rtol, atol=atol)
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state2[var_name], rtol=rtol, atol=atol)

    def test_conv1d_training(self):
        self._test_convolution_training_base(dim=1, dtype=torch.float)
        if core.onednn_has_bf16_support():
            self._test_convolution_training_base(dim=1, dtype=torch.bfloat16, rtol=1e-2, atol=1e-03)
        # TODO: add inference case.

    def test_conv2d_training(self):
        self._test_convolution_training_base(dim=2, dtype=torch.float)
        if core.onednn_has_bf16_support(): 
            self._test_convolution_training_base(dim=2, dtype=torch.bfloat16, rtol=1e-2, atol=1e-03)

        # TODO: add inference case.

    def test_conv3d_training(self):
        # Stock Pytorch does not support channelslast for Conv3d
        # self._test_convolution_training_base(dim=3, dtype=torch.float)
        if core.onednn_has_bf16_support():
            self._test_convolution_training_base(dim=3, dtype=torch.bfloat16, rtol=1e-2, atol=1e-03)
        # TODO: add inference case.

    def _test_conv_nc11_base(self, dim):
        # related issue: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/pull/86.
        channels_last = torch.channels_last if dim ==2 else torch.channels_last_3d
        test_dtypes = [torch.float]
        if core.onednn_has_bf16_support():
            test_dtypes.append(torch.bfloat16)
        options = itertools.product(test_dtypes,
                                    [1, 256], [1, 324],
                                    [torch.contiguous_format, channels_last],
                                    [True, False])

        for dtype, in_channels, out_channels, memory_format, feed_sample_input in options:
            model = conv_module[dim](in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False)
            model = model.to(memory_format=memory_format).to(dtype=dtype).float().train()
            input_shape = [32, in_channels, 1, 1]
            if dim == 3:
                input_shape.append(1)
            x = torch.randn(input_shape).to(memory_format=memory_format).to(dtype=dtype).float()

            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            x3 = x.clone().requires_grad_()
            origin_model1 = copy.deepcopy(model).train()
            origin_optimizer1 = SGD(origin_model1.parameters(), lr=0.01, momentum=0.9)
            origin_model2 = copy.deepcopy(model).train()
            origin_optimizer2 = SGD(origin_model2.parameters(), lr=0.01, momentum=0.9)
            if feed_sample_input:
                ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1', sample_input=x)
                ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True, sample_input=x)
            else:
                ipex_model1, ipex_optimizer1 = ipex.optimize(origin_model1, dtype=dtype, optimizer=origin_optimizer1, level='O1')
                ipex_model2, ipex_optimizer2 = ipex.optimize(origin_model2, dtype=dtype, optimizer=origin_optimizer2, level='O1', inplace=True)

            # train one step for origin.
            y1 = origin_model1(x1)
            loss1 = y1.sum()
            origin_optimizer1.zero_grad()
            loss1.backward()
            origin_optimizer1.step()

            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
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

            self.assertEqual(y1, y2.float(), rtol=1e-2, atol=1e-03)
            self.assertEqual(y1, y3.float(), rtol=1e-2, atol=1e-03)
            self.assertEqual(x1.grad, x2.grad, rtol=1e-2, atol=1e-03)
            self.assertEqual(x1.grad, x3.grad, rtol=1e-2, atol=1e-03)
            # compare origin_model parameters with origin_model parameters after grad updata
            origin_model_state = origin_model1.state_dict()
            ipex_model_state1 = ipex_model1.state_dict()
            ipex_model_state2 = ipex_model2.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state1[var_name], rtol=1e-2, atol=1e-03)
                self.assertEqual(origin_model_state[var_name], ipex_model_state2[var_name], rtol=1e-2, atol=1e-03)

            # compare momentum_buffer in optimizer's state(sgd)
            # TODO: other optimizer.
            origin_optimizer_state = origin_optimizer1.state_dict()
            ipex_optimizer_state1 = ipex_optimizer1.state_dict()
            ipex_optimizer_state2 = ipex_optimizer2.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state1[var_name], rtol=1e-2, atol=1e-03)
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state2[var_name], rtol=1e-2, atol=1e-03)

    def test_conv2d_nc11(self):
        self._test_conv_nc11_base(dim=2)

    def test_conv3d_nc11(self):
        self._test_conv_nc11_base(dim=3)

    def _test_conv_serialization_base(self, dim):
        channels_last = torch.channels_last if dim ==2 else torch.channels_last_3d
        optimizer_options = [Lamb, Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]
        test_dtypes = [torch.float]
        if core.onednn_has_bf16_support():
            test_dtypes.append(torch.bfloat16)
        options = itertools.product(test_dtypes, optimizer_options, [True, False])
        input_shape = [8, 3, 56, 56]
        if dim == 3:
            input_shape.append(56)
        for dtype, optimizer, feed_sample_input in options:
            model = conv_module[dim](3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            x = torch.randn(input_shape).to(dtype=dtype).float().to(memory_format=channels_last)
            model = model.to(dtype=dtype).float().to(memory_format=channels_last).train()
            origin_x = x.clone()
            ipex_x = x.clone()
            origin_model = copy.deepcopy(model).train()
            lr = 1e-2
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            if feed_sample_input:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1', sample_input=x)
            else:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            # train one step for origin.
            y1 = origin_model(origin_x)
            loss1 = y1.sum()
            origin_optimizer.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_value_(origin_model.parameters(), 10)
            origin_optimizer.step()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                # train one step for ipex.
                y2 = ipex_model(ipex_x)
                loss2 = y2.sum()
                ipex_optimizer.zero_grad()
                loss2.backward()
                torch.nn.utils.clip_grad_value_(ipex_model.parameters(), 10)
                ipex_optimizer.step()

            torch.save({'model_state_dict': origin_model.state_dict(),
                        'optimizer_state_dict': origin_optimizer.state_dict()
                        }, 'origin_checkpoint.pth')
            torch.save({'model_state_dict': ipex_model.state_dict(),
                        'optimizer_state_dict': ipex_optimizer.state_dict()
                        }, 'ipex_checkpoint.pth')

            self.assertEqual(y1, y2.float(), rtol=1e-4, atol=5e-02)
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-2, atol=1e-03)

            # check state_buffer works.
            origin_optimizer_state = origin_optimizer.state_dict()
            ipex_optimizer_state = ipex_optimizer.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-2, atol=5e-02)

            origin_model = copy.deepcopy(model).train()
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            origin_checkpoint = torch.load('origin_checkpoint.pth')
            origin_model.load_state_dict(origin_checkpoint['model_state_dict'])
            origin_optimizer.load_state_dict(origin_checkpoint['optimizer_state_dict'])
            # load ipex model state
            origin_ipex_model = copy.deepcopy(model)
            origin_ipex_optimizer = optimizer(origin_ipex_model.parameters(), lr=lr)
            ipex_checkpoint = torch.load('ipex_checkpoint.pth')
            origin_ipex_model.load_state_dict(ipex_checkpoint['model_state_dict'])
            origin_ipex_optimizer.load_state_dict(ipex_checkpoint['optimizer_state_dict'])
            if feed_sample_input:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1', sample_input=x)
            else:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            # train second step for origin.
            y1 = origin_model(origin_x)
            loss = y1.sum()
            origin_optimizer.zero_grad()
            loss.backward()
            origin_optimizer.step()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                # traing second step for ipex model.
                y3 = ipex_model(ipex_x)
                loss3 = y3.sum()
                ipex_optimizer.zero_grad()
                loss3.backward()
                ipex_optimizer.step()

            self.assertEqual(y1, y3.float(), rtol=1e-2, atol=5e-02)
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-2, atol=5e-02)
            os.remove('origin_checkpoint.pth')
            os.remove('ipex_checkpoint.pth')

    def test_conv2d_serialization(self):
        self._test_conv_serialization_base(dim=2)

    def test_conv3d_serialization(self):
        self._test_conv_serialization_base(dim=3)

    def _test_imagenet_model(self, model):
        model = model.to(memory_format=torch.channels_last)
        test_dtypes = [torch.float]
        if core.onednn_has_bf16_support():
            test_dtypes.append(torch.bfloat16)
        for dtype, feed_sample_input in itertools.product(test_dtypes, [True, False]):
            model = model.to(dtype).float()
            # inference case, will do conv+bn folding 'O1'. do nothing for 'O0'.
            x = torch.randn(1, 3, 224, 224).to(dtype=dtype).float().to(memory_format=torch.channels_last)
            # inference case, will do conv+bn folding 'O1'. do nothing for 'O0'.
            if feed_sample_input:
                ipex_model2 = ipex.optimize(model.eval(), dtype=dtype, level='O1', sample_input=x)
            else:
                ipex_model2 = ipex.optimize(model.eval(), dtype=dtype, level='O1')
            y1 = model(x)
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y2 = ipex_model2(x)

            self.assertEqual(y1, y2.float(), rtol=1e-2, atol=5e-2)
            # traing case.
            origin_model = copy.deepcopy(model).train()
            origin_optimizer = ASGD(origin_model.parameters(), lr=0.01)
            # do weight prepack for 'O1'
            if feed_sample_input:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1', sample_input=x)
            else:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            # run two iterations, and then compare the results.

            xx = [torch.randn(1, 3, 224, 224).to(dtype=dtype).float().to(memory_format=torch.channels_last),
                  torch.randn(1, 3, 224, 224).to(dtype=dtype).float().to(memory_format=torch.channels_last)]
            for i in range(2):
                x = xx[i]
                # original case
                y1 = origin_model(x.clone())
                loss1 = y1.sum()
                origin_optimizer.zero_grad()
                loss1.backward()
                origin_optimizer.step()
                with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                    y2 = ipex_model(x.clone())
                    loss2 = y2.sum()
                    ipex_optimizer.zero_grad()
                    loss2.backward()
                    ipex_optimizer.step()
            self.assertEqual(y1, y2.float(), rtol=6e-2, atol=1e-2)


    @skipIfNoTorchVision
    def test_resnet18(self):
        model = torchvision.models.resnet.resnet18(pretrained=False)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
        self._test_imagenet_model(model)

    def test_blas_backend(self):
        class L(torch.nn.Module):
            def __init__(self, in_f, out_f, bias):
                super(L, self).__init__()
                self.linear = torch.nn.Linear(in_f, out_f, bias=bias)

            def forward(self, x):
                return self.linear(x)

        out_features = torch.randint(3, 10, (1,)).item()
        in_features = torch.randint(3, 10, (1,)).item()

        input_shape = (8, in_features)
        x = torch.randn(input_shape, dtype=torch.float32)
        model = L(in_features, out_features, True)
        origin_model = copy.deepcopy(model).eval()

        def test_mkl():
            self.assertTrue(ipex._using_dnnl()==False)
            ipex_model_mkl = ipex.optimize(origin_model, dtype=torch.float32, level='O1')
            with torch.no_grad():
                graph = torch.jit.trace(ipex_model_mkl.eval(), x)
                graph = torch.jit.freeze(graph)
                graph(x)
                trace_graph = graph.graph_for(x)
                self.assertTrue(any(n.kind() == "ipex_prepack::mkl_sgemm_run" for n in trace_graph.nodes()))
        test_mkl()

        ipex_model_dnnl = ipex.optimize(origin_model, dtype=torch.float32, level='O1', auto_kernel_selection=True)
        self.assertTrue(ipex._using_dnnl())
        with torch.no_grad():
            dnnl_graph = torch.jit.trace(ipex_model_dnnl.eval(), x)
            dnnl_graph = torch.jit.freeze(dnnl_graph)
            dnnl_graph(x)
            trace_graph = dnnl_graph.graph_for(x)
            self.assertTrue(any(n.kind() == "ipex_prepack::linear_run" for n in trace_graph.nodes()))

        ipex._disable_dnnl()
        test_mkl()

        ipex_model = ipex.optimize(origin_model, dtype=torch.float32, level='O1', weights_prepack=False)
        self.assertTrue(ipex._using_dnnl()==False)
        with torch.no_grad():
            graph = torch.jit.trace(ipex_model.eval(), x)
            graph = torch.jit.freeze(graph)
            graph(x)
            trace_graph = graph.graph_for(x)
            self.assertTrue(any(n.kind() == "aten::linear" for n in trace_graph.nodes()))

        ipex_model = ipex.optimize(origin_model, dtype=torch.float32, level='O1', auto_kernel_selection=True, weights_prepack=False)
        self.assertTrue(ipex._using_dnnl())
        with torch.no_grad():
            graph = torch.jit.trace(ipex_model.eval(), x)
            graph = torch.jit.freeze(graph)
            graph(x)
            trace_graph = graph.graph_for(x)
            self.assertTrue(any(n.kind() == "aten::linear" for n in trace_graph.nodes()))

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
        test_dtypes = [torch.float]
        if core.onednn_has_bf16_support():
            test_dtypes.append(torch.bfloat16)
        options = itertools.product([True, False], input_shapes, [True, False], test_dtypes)
        for bias, x_shape, feed_sample_input, dtype in options:
            x = torch.randn(x_shape, dtype=torch.float32).to(dtype=dtype).float()
            model = L(in_features, out_features, bias).to(dtype=dtype).float().eval()
            x1 = x.clone().requires_grad_(False)
            x2 = x.clone().requires_grad_(False)
            origin_model = copy.deepcopy(model).eval()
            if feed_sample_input:
                ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1', sample_input=x)
            else:
                ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1')

            self.assertEqual(ipex_model.linear.weight.dtype, dtype)
            y1 = origin_model(x1)
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                # ipex path
                y2 = ipex_model(x2)
            self.assertEqual(y1, y2.float(), rtol=1e-2, atol=1e-3)

    @unittest.skipIf(not core.onednn_has_bf16_support(), "ipex linear bf16 is not supported on this CPU device")
    def test_linear_training(self):
        linear_module = torch.nn.Linear
        out_feature = [1024, 256, 1, torch.randint(3, 10, (1, )).item()]
        in_feature = [128, 479, torch.randint(3, 10, (1, )).item()]
        input_shapes = []
        for s in in_feature:
            input_shapes += [(128, s), (2, 64, s), (2, 2, 32, s)]
        
        options = itertools.product(out_feature, [True, False], input_shapes, [torch.bfloat16], [True, False])
        for out_features, bias, x_shape, dtype, feed_sample_input in options:
            in_features = x_shape[-1]
            model = torch.nn.Linear(in_features, out_features, bias=bias).to(dtype=dtype).float().train()
            x = torch.randn(x_shape, dtype=torch.float32).to(dtype=dtype).float()
            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            origin_model = copy.deepcopy(model).train()
            origin_optimizer = SGD(origin_model.parameters(), lr=0.01, momentum=0.9)
            if feed_sample_input:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1', sample_input=x)
            else:
                ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            self.assertTrue(ipex_model.weight.dtype == dtype)

            for i in range(1):
                # original fp32 path
                y1 = origin_model(x1)
                loss1 = y1.sum()
                origin_optimizer.zero_grad()
                loss1.backward()
                origin_optimizer.step()
                with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                    # ipex path
                    y2 = ipex_model(x2)
                    loss2 = y2.sum()
                    ipex_optimizer.zero_grad()
                    loss2.backward()
                    ipex_optimizer.step()
            self.assertEqual(y1, y2.float(), rtol=1e-2, atol=1e-3)
            self.assertEqual(x1.grad, x2.grad, rtol=1e-2, atol=1e-3)
            if bias:
                self.assertEqual(origin_model.bias.grad, ipex_model.bias.grad.float(), rtol=1e-2, atol=1e-3)
            # compare origin_model parameters with origin_model parameters after grad updata
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-2, atol=1e-3)
            # compare momentum_buffer in optimizer's state(sgd)
            # TODO: other optimizer.
            origin_optimizer_state = origin_optimizer.state_dict()
            ipex_optimizer_state = ipex_optimizer.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-2, atol=1e-3)

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
                self.deconv = torch.nn.ConvTranspose2d(ic, oc, kernel_size=kernel_size, stride=stride, \
                                                       padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

            def forward(self, x):
                return self.deconv(x)

        class Deconv3d(torch.nn.Module):
            def __init__(self, ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation):
                super(Deconv3d, self).__init__()
                self.deconv = torch.nn.ConvTranspose3d(ic, oc, kernel_size=kernel_size, stride=stride, padding=padding, \
                                                       output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

            def forward(self, x):
                return self.deconv(x)

        params_list = self._deconv_params_list()
        torch.manual_seed(0)
        for input_height, input_width, input_depth, input_channel_per_group, output_channel_per_group, kernel_size, bias, stride, \
                padding, output_padding, groups, dilation in list(itertools.product(*params_list)) + [self._deconv_with_output_padding()] + [self._deconv_fallback_shape()]:
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
                    model = Deconv3d(ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation).to(memory_format=torch.channels_last_3d)
                    x = torch.rand((2, ic, input_depth, input_height, input_width)).to(memory_format=torch.channels_last_3d)
                test_dtypes = [torch.float]
                if core.onednn_has_bf16_support():
                    test_dtypes.append(torch.bfloat16)
                for dtype, feed_sample_input in itertools.product(test_dtypes, [True, False]):
                    x = x.to(dtype=dtype).float()
                    model = model.to(dtype=dtype).float()
                    if inference:
                        model.eval()
                        origin_model = copy.deepcopy(model).eval()
                        if feed_sample_input:
                            ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1', sample_input=x)
                        else:
                            ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1')

                        if padding - output_padding + stride <= 0:
                            # unsupported in mkldnn, should not replace the original ConvTranspose module
                            self.assertTrue(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))
                            continue
                        else:
                            self.assertFalse(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))

                        self.assertEqual(ipex_model.deconv.weight.dtype, dtype)
                        y_origin = origin_model(x)
                        with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                            y_ipex = ipex_model(x)
                        self.assertEqual(y_origin, y_ipex.float(), rtol=1e-2, atol=1e-03)
                    else:
                        model.train()
                        origin_model = copy.deepcopy(model).train()
                        origin_optimizer = SGD(origin_model.parameters(), lr=0.01, momentum=0.9)
                        if feed_sample_input:
                            ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1', sample_input=x)
                        else:
                            ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
                        
                        if padding - output_padding + stride <= 0:
                            # unsupported in mkldnn, should not replace the original ConvTranspose module
                            self.assertTrue(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))
                            continue
                        else:
                            self.assertFalse(module_found(ipex_model, torch.nn.ConvTranspose2d if dims == 2 else torch.nn.ConvTranspose3d))                        
                        
                        x1 = x.clone().requires_grad_()
                        x2 = x.clone().requires_grad_()

                        y1 = origin_model(x1)
                        loss1 = y1.sum()
                        origin_optimizer.zero_grad()
                        loss1.backward()
                        origin_optimizer.step()
                        with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                            y2 = ipex_model(x2)
                            loss2 = y2.sum()
                            ipex_optimizer.zero_grad()
                            loss2.backward()
                            ipex_optimizer.step()
                            self.assertEqual(y1, y2.float(), rtol=1e-2, atol=1e-3)
                            self.assertEqual(x1.grad, x2.grad, rtol=1e-2, atol=1e-3)
                            if bias:
                                self.assertEqual(origin_model.deconv.bias.grad, ipex_model.deconv.bias.grad.float(), rtol=1e-2, atol=1e-3)

                            # compare origin_model parameters with origin_model parameters after grad updata
                            origin_model_state = origin_model.state_dict()
                            ipex_model_state = ipex_model.state_dict()
                            for var_name in origin_model_state:
                                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-2, atol=1e-3)

                        # compare momentum_buffer in optimizer's state(sgd)
                        # TODO: other optimizer.
                        origin_optimizer_state = origin_optimizer.state_dict()
                        ipex_optimizer_state = ipex_optimizer.state_dict()

                        for var_name in origin_optimizer_state:
                            if var_name == 'state':
                                self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-2, atol=1e-03)

    def test_deconv_2d_inference(self):
        self._test_deconv(dims=2, inference=True)

    def test_deconv_2d_training(self):
        self._test_deconv(dims=2, inference=False)

    def test_deconv_3d_inference(self):
        self._test_deconv(dims=3, inference=True)

    def test_deconv_3d_training(self):
        self._test_deconv(dims=3, inference=False)

    def test_hook(self):
        class ConvNd(torch.nn.Module):
            def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, groups):
                super(ConvNd, self).__init__()
                self.conv = conv_module[dim](in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

            def forward(self, x):
                return self.conv(x)

        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [1, 2], [1, 4])
        for dim in [1, 2, 3]:
            for bias, dilation, groups in options:
                N = torch.randint(3, 10, (1,)).item()
                M = torch.randint(1, 3, (1,)).item() * groups
                C = torch.randint(1, 3, (1,)).item() * groups
                x_shape = (N, C) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model_base = ConvNd(
                    dim=dim,
                    in_channels=C,
                    out_channels=M,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=dilation,
                    bias=bias,
                    groups=groups).float().eval()

                module_type = torch.nn.Conv1d
                if dim == 2:
                    module_type = torch.nn.Conv2d
                if dim == 3:
                    module_type = torch.nn.Conv3d

                # IPEX will replace Conv with IPEX Conv
                model = copy.deepcopy(model_base)
                ipex_model = ipex.optimize(model, dtype=torch.float32, level='O1')
                ipex_model(x)
                self.assertFalse(isinstance(ipex_model.conv, module_type))

                # Use torch.nn.utils.weight_norm to hook weight in model.conv,
                # hook function here is WeightNorm, it has 'name' attribute,
                # IPEX will not do prepack and not replace Conv with IPEX Conv.
                hook_weight_model = copy.deepcopy(model_base)
                hook_weight_model.conv = torch.nn.utils.weight_norm(hook_weight_model.conv, name="weight")
                hook_weight_model = ipex.optimize(hook_weight_model, dtype=torch.float32, level='O1', inplace=True)
                hook_weight_model(x)
                self.assertTrue(isinstance(hook_weight_model.conv, module_type))

                # User-defined hook function, it maybe has 'name' attribute or not,
                # hook.name maybe is 'weight' or 'bias' or not. Only when hook function
                # has 'name' attribute and hook on 'weight' or 'bias', IPEX will
                # not do prepack. In other situations, IPEX will do prepack as usual.
                dict_features = {}
                options = itertools.product(['pre', 'forward', 'backward'], [True, False], ['weight', 'bias', 'others'])
                for hook_type, has_name_attr, name in options:
                    hook_model = copy.deepcopy(model_base)
                    if hook_type == 'pre':
                        def forward_pre_hook(self, input):
                            dict_features['input'] = input
                        if has_name_attr:
                            forward_pre_hook.name = name
                        hook_model.conv.register_forward_pre_hook(forward_pre_hook)
                    elif hook_type == 'forward':
                        def forward_hook(self, input, output):
                            dict_features['input'] = input
                            dict_features['output'] = output
                        if has_name_attr:
                            forward_hook.name = name
                        hook_model.conv.register_forward_hook(forward_hook)
                    else:
                        def backward_hook(self, grad_input, grad_output):
                            dict_features['grad_input'] = grad_input
                            dict_features['grad_output'] = grad_output
                        if has_name_attr:
                            backward_hook.name = name
                        hook_model.conv.register_backward_hook(backward_hook)
                    hook_model = ipex.optimize(hook_model, dtype=torch.float32, level='O1')
                    hook_model(x)
                    if has_name_attr and (name == 'weight' or name == 'bias'):
                        self.assertTrue(isinstance(hook_model.conv, module_type))
                    else:
                        self.assertFalse(isinstance(hook_model.conv, module_type))

    def _lstm_params_list(self):
        params_dict = {
            "input_size": [1, 2],
            "hidden_size": [5, 16],
            "num_layers": [1, 3],
            "bidirectional": [False, True],
            "bias": [False, True],
            "empty_state": [False, True],
            "batch_first": [False, True],
            "dropout": [0, 0.4, 0.7, 1],
            "batch_size": [1, 2],
            "seq_len": [1, 3]
        }

        params_list = []
        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def _test_lstm(self, inference):
        class Lstm(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional, bias, dropout, batch_first):
                super(Lstm, self).__init__()
                self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

            def forward(self, x, h=None):
                x, h = self.lstm(x, h)
                return x, h

        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        params_list = self._lstm_params_list()
        for input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, dropout, batch_size, seq_len in itertools.product(*params_list):
            # dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
            if dropout > 0 and num_layers == 1:
                continue

            num_directions = 2 if bidirectional else 1

            if batch_first:
                input = torch.randn(batch_size, seq_len, input_size)
            else:
                input = torch.randn(seq_len, batch_size, input_size)
            h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

            model = Lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

            test_dtypes = [torch.float]
            if core.onednn_has_bf16_support():
                test_dtypes.append(torch.bfloat16)
            for dtype in test_dtypes:
                # fp32 cannot afford default tolerance. After debug, weight grad after
                # backward has error, but they are in the tolerance range. If replacing
                # with same weight grad in origin_model and ipex_model, model state
                # and optimizer state will be the same. Therefore, optimizer update is
                # correct. Before optimizer update, it is backward, its input x, weight
                # are the same, y(hy) has error, but they are in tolerance range. So
                # that the difference of lstm kernel between pytorch and oneDNN will lead
                # to model state diff.
                rtol=1e-5
                atol=1e-5
                if dtype == torch.bfloat16:
                    # align atol with that in _test_lstm in test_autocast.py of bf16
                    rtol=2e-2
                    atol=3e-2
                x = input.to(dtype=dtype).float()
                h = h.to(dtype=dtype).float()
                c = c.to(dtype=dtype).float()
                model = model.to(dtype=dtype).float()
                if inference:
                    model.eval()
                    origin_model = copy.deepcopy(model).eval()
                    ipex_model = ipex.optimize(origin_model, dtype=dtype, level='O1')
                    self.assertEqual(ipex_model.lstm.weight_ih_l0.dtype, dtype)
                    self.assertEqual(ipex_model.lstm.weight_hh_l0.dtype, dtype)
                    if empty_state:
                        y_origin, hy_origin = origin_model(x)
                    else:
                        y_origin, hy_origin = origin_model(x, (h, c))

                    with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                        if empty_state:
                            y_ipex, hy_ipex = ipex_model(x)
                        else:
                            y_ipex, hy_ipex = ipex_model(x, (h, c))

                    self.assertEqual(y_origin, y_ipex.float(), rtol=rtol, atol=atol)
                    self.assertEqual(hy_origin[0], hy_ipex[0].float(), rtol=rtol, atol=atol)
                    self.assertEqual(hy_origin[1], hy_ipex[1].float(), rtol=rtol, atol=atol)
                else:
                    model.train()
                    origin_model = copy.deepcopy(model).train()
                    origin_optimizer = SGD(origin_model.parameters(), lr=0.01)
                    ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')

                    x1 = x.clone().requires_grad_()
                    x2 = x.clone().requires_grad_()
                    h1 = h.clone().requires_grad_()
                    h2 = h.clone().requires_grad_()
                    c1 = c.clone().requires_grad_()
                    c2 = c.clone().requires_grad_()

                    if empty_state:
                        torch.manual_seed(rand_seed)
                        y1, hy1 = origin_model(x1)
                    else:
                        torch.manual_seed(rand_seed)
                        y1, hy1 = origin_model(x1, (h1, c1))
                    loss1 = y1.sum()
                    if not empty_state:
                        loss1_hy1_0 = hy1[0].sum()
                        loss1_hy1_1 = hy1[1].sum()
                    origin_optimizer.zero_grad()
                    loss1.backward(retain_graph=True)
                    if not empty_state:
                        loss1_hy1_0.backward(retain_graph=True)
                        loss1_hy1_1.backward(retain_graph=True)
                    origin_optimizer.step()

                    with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                        if empty_state:
                            torch.manual_seed(rand_seed)
                            y2, hy2 = ipex_model(x2)
                        else:
                            torch.manual_seed(rand_seed)
                            y2, hy2 = ipex_model(x2, (h2, c2))
                        loss2 = y2.sum()
                        if not empty_state:
                            loss2_hy2_0 = hy2[0].sum()
                            loss2_hy2_1 = hy2[1].sum()

                        ipex_optimizer.zero_grad()
                        loss2.backward(retain_graph=True)
                        if not empty_state:
                            loss2_hy2_0.backward(retain_graph=True)
                            loss2_hy2_1.backward(retain_graph=True)
                        for name, para in origin_model.lstm.named_parameters():
                            self.assertEqual(para.grad, getattr(ipex_model.lstm, name).grad.float(), rtol=rtol, atol=atol)
                        ipex_optimizer.step()
                    self.assertEqual(y1, y2.float(), rtol=rtol, atol=atol)
                    self.assertEqual(hy1[0], hy2[0].float(), rtol=rtol, atol=atol)
                    self.assertEqual(hy1[1], hy2[1].float(), rtol=rtol, atol=atol)
                    self.assertEqual(x1.grad, x2.grad, rtol=rtol, atol=atol)

                    if not empty_state:
                        self.assertEqual(h1.grad, h2.grad, rtol=rtol, atol=atol)
                        self.assertEqual(c1.grad, c2.grad, rtol=rtol, atol=atol)

                    origin_model_state = origin_model.state_dict()
                    ipex_model_state = ipex_model.state_dict()
                    for var_name in origin_model_state:
                        self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=rtol, atol=atol)

                    origin_optimizer_state = origin_optimizer.state_dict()
                    ipex_optimizer_state = ipex_optimizer.state_dict()
                    for var_name in origin_optimizer_state:
                        if var_name == 'state':
                            self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=rtol, atol=atol)

    def test_lstm_inference(self):
        self._test_lstm(inference=True)

    def test_lstm_training(self):
        self._test_lstm(inference=False)

    def test_lstm_serialization(self):
        class Lstm(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional, bias, dropout, batch_first):
                super(Lstm, self).__init__()
                self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

            def forward(self, x, h=None):
                x, h = self.lstm(x, h)
                return x, h

        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        optimizer_options = [Lamb, Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, Rprop, SGD]
        input_size = 1
        hidden_size = 5
        num_layers = 3
        bidirectional = True
        bias = True
        empty_state = False
        batch_first = True
        dropout = 0.2
        batch_size = 2
        seq_len = 3

        num_directions = 2 if bidirectional else 1

        if batch_first:
            input = torch.randn(batch_size, seq_len, input_size)
        else:
            input = torch.randn(seq_len, batch_size, input_size)
        h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
        c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

        model = Lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

        test_dtypes = [torch.float]
        if core.onednn_has_bf16_support():
            test_dtypes.append(torch.bfloat16)
        options = itertools.product(test_dtypes, optimizer_options)
        for dtype, optimizer in options:
            rtol=1.3e-6
            atol=1e-5
            if dtype == torch.bfloat16:
                rtol=2e-2
                atol=3e-2
            x = input.to(dtype=dtype).float()
            h = h.to(dtype=dtype).float()
            c = c.to(dtype=dtype).float()
            model = model.to(dtype=dtype).float()
            model.train()
            lr = 1e-2
            origin_model = copy.deepcopy(model).train()
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')

            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            h1 = h.clone().requires_grad_()
            h2 = h.clone().requires_grad_()
            c1 = c.clone().requires_grad_()
            c2 = c.clone().requires_grad_()

            if empty_state:
                torch.manual_seed(rand_seed)
                y1, hy1 = origin_model(x1)
            else:
                torch.manual_seed(rand_seed)
                y1, hy1 = origin_model(x1, (h1, c1))
            loss1 = y1.sum()
            loss1_hy1_0 = hy1[0].sum()
            loss1_hy1_1 = hy1[1].sum()
            origin_optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            if not empty_state:
                loss1_hy1_0.backward(retain_graph=True)
                loss1_hy1_1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_value_(origin_model.parameters(), 10)
            origin_optimizer.step()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                if empty_state:
                    torch.manual_seed(rand_seed)
                    y2, hy2 = ipex_model(x2)
                else:
                    torch.manual_seed(rand_seed)
                    y2, hy2 = ipex_model(x2, (h2, c2))
                loss2 = y2.sum()
                loss2_hy2_0 = hy2[0].sum()
                loss2_hy2_1 = hy2[1].sum()
                ipex_optimizer.zero_grad()
                loss2.backward(retain_graph=True)
                if not empty_state:
                    loss2_hy2_0.backward(retain_graph=True)
                    loss2_hy2_1.backward(retain_graph=True)
                torch.nn.utils.clip_grad_value_(ipex_model.parameters(), 10)
                ipex_optimizer.step()

            torch.save({'model_state_dict': origin_model.state_dict(),
                        'optimizer_state_dict': origin_optimizer.state_dict()
                        }, 'origin_checkpoint.pth')
            torch.save({'model_state_dict': ipex_model.state_dict(),
                        'optimizer_state_dict': ipex_optimizer.state_dict()
                        }, 'ipex_checkpoint.pth')

            self.assertEqual(y1, y2.float(), rtol=rtol, atol=atol)
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=rtol, atol=atol)

            origin_optimizer_state = origin_optimizer.state_dict()
            ipex_optimizer_state = ipex_optimizer.state_dict()
            for var_name in origin_optimizer_state:
                if var_name == 'state':
                    self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=rtol, atol=atol)

            origin_model = copy.deepcopy(model).train()
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            origin_checkpoint = torch.load('origin_checkpoint.pth')
            origin_model.load_state_dict(origin_checkpoint['model_state_dict'])
            origin_optimizer.load_state_dict(origin_checkpoint['optimizer_state_dict'])
            # load ipex model state
            origin_ipex_model = copy.deepcopy(model)
            origin_ipex_optimizer = optimizer(origin_ipex_model.parameters(), lr=lr)
            ipex_checkpoint = torch.load('ipex_checkpoint.pth')
            origin_ipex_model.load_state_dict(ipex_checkpoint['model_state_dict'])
            origin_ipex_optimizer.load_state_dict(ipex_checkpoint['optimizer_state_dict'])

            ipex_model, ipex_optimizer = ipex.optimize(origin_model, dtype=dtype, optimizer=origin_optimizer, level='O1')
            # train second step for origin.
            if empty_state:
                torch.manual_seed(rand_seed)
                y1, hy1 = origin_model(x1)
            else:
                torch.manual_seed(rand_seed)
                y1, hy1 = origin_model(x1, (h1, c1))
            loss1 = y1.sum()
            loss1_hy1_0 = hy1[0].sum()
            loss1_hy1_1 = hy1[1].sum()
            origin_optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            if not empty_state:
                loss1_hy1_0.backward(retain_graph=True)
                loss1_hy1_1.backward(retain_graph=True)
            origin_optimizer.step()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                # traing second step for ipex model.
                if empty_state:
                    torch.manual_seed(rand_seed)
                    y3, hy3 = ipex_model(x2)
                else:
                    torch.manual_seed(rand_seed)
                    y3, hy3 = ipex_model(x2, (h2, c2))
                loss3 = y3.sum()
                loss3_hy3_0 = hy3[0].sum()
                loss3_hy3_1 = hy3[1].sum()
                ipex_optimizer.zero_grad()
                loss3.backward(retain_graph=True)
                if not empty_state:
                    loss3_hy3_0.backward(retain_graph=True)
                    loss3_hy3_1.backward(retain_graph=True)
                ipex_optimizer.step()

            self.assertEqual(y1, y3.float(), rtol=rtol, atol=atol)
            origin_model_state = origin_model.state_dict()
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=rtol, atol=atol)
            os.remove('origin_checkpoint.pth')
            os.remove('ipex_checkpoint.pth')

if __name__ == '__main__':
    torch.manual_seed(2020)
    test = unittest.main()
