# from turtle import forward
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import copy
from math import sqrt

import intel_extension_for_pytorch  # noqa

from torch.quantization.quantize_jit import (convert_jit, prepare_jit)
from torch.jit._recursive import wrap_cpp_module

import pytest
class TestNNMethod(TestCase):
    def test_rnn_weight_norm(self):
        def check_weight_norm(l, name, num_params):
            # This Module has 4 or 5 parameters called:
            # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', weight_hr_l0

            # Applying weight norm on one of them causes it to become a tensor
            l = torch.nn.utils.weight_norm(l, name=name)
            self.assertEqual(
                sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights]),
                num_params - 1,
            )

            # Removing the weight norm reparametrization restores the Parameter
            l = torch.nn.utils.remove_weight_norm(l, name=name)
            self.assertEqual(
                sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights]),
                num_params,
            )

            # Make sure that, upon removal of the reparametrization, the
            # `._parameters` and `.named_parameters` contain the right params.
            # Specifically, the original weight ('weight_ih_l0') should be placed
            # back in the parameters, while the reparametrization components
            # ('weight_ih_l0_v' and 'weight_ih_l0_g') should be removed.
            self.assertTrue(name in l._parameters)
            self.assertIsNotNone(l._parameters[name])
            self.assertTrue(name + '_v' not in l._parameters)
            self.assertTrue(name + '_g' not in l._parameters)
            self.assertTrue(name in dict(l.named_parameters()))
            self.assertIsNotNone(dict(l.named_parameters())[name])
            self.assertTrue(name + '_v' not in dict(l.named_parameters()))
            self.assertTrue(name + '_g' not in dict(l.named_parameters()))

        check_weight_norm(torch.nn.LSTM(32, 32).to("xpu"), 'weight_ih_l0', 4)
        check_weight_norm(torch.nn.LSTM(32, 32, proj_size=16).to("xpu"), 'weight_hr_l0', 5)


    def test_weight_norm(self):
        input = torch.randn(3, 5).to('xpu')
        m = nn.Linear(5, 7).to('xpu')
        expected_output = m(input)

        # add weight normalization
        m = torch.nn.utils.weight_norm(m)
        self.assertEqual(m.weight_v.size(), m.weight.size())
        self.assertEqual(m.weight_g.size(), (7, 1))
        print("expected: ", expected_output)
        print("obtained: ", m(input))
        self.assertEqual(m(input), expected_output)

        # remove weight norm
        m = torch.nn.utils.remove_weight_norm(m)
        self.assertFalse(hasattr(m, 'weight_g'))
        self.assertFalse(hasattr(m, 'weight_v'))
        self.assertEqual(m(input), expected_output)

        # test with dim=1
        m = torch.nn.utils.weight_norm(m, dim=1)
        self.assertEqual(m.weight_v.size(), m.weight.size())
        self.assertEqual(m.weight_g.size(), (1, 5))
        self.assertEqual(m(input), expected_output)

        # test with dim=None
        m = nn.Linear(5, 7).to("xpu")
        expected_output = m(input)
        m = torch.nn.utils.weight_norm(m, dim=None)
        self.assertEqual(m(input), expected_output)

        with self.assertRaisesRegex(RuntimeError, 'register two weight_norm hooks'):
            m = torch.nn.utils.weight_norm(m)
            m = torch.nn.utils.weight_norm(m)

    def test_weight_norm_native(self):
        input = torch.randn(3, 5, requires_grad=False)
        g = torch.randn(3, 5)
        res, norm = torch._weight_norm_interface(input, g, dim=0)
        xpu_res, xpu_norm = torch._weight_norm_interface(input.to("xpu"), g.to("xpu"), dim=0)
        self.assertEqual(xpu_res.cpu(), res)
        m = nn.Linear(5, 7)
        xpu_m = copy.deepcopy(m).to("xpu")
        m = torch.nn.utils.weight_norm(m, dim=0)
        xpu_m = torch.nn.utils.weight_norm(xpu_m, dim=0)
        input_xpu = input.to("xpu")
        res = m(input)
        res_xpu = xpu_m(input_xpu)
        res = torch.sum(res)
        res_xpu = torch.sum(res_xpu)
        res.backward()
        res_xpu.backward()
        self.assertEqual(input.grad, input_xpu.grad)

    def test_weight_norm_dim0(self):
        v = torch.randn(32, 8193 * 253).requires_grad_(True)
        g = torch.randn(32).requires_grad_(True)
        gw = torch.randn(32, 8193 * 253)
        w, n = torch._weight_norm_interface(v, g, dim=0)
        w.backward(gw)
        v_xpu = v.detach().clone().to("xpu").requires_grad_(True)
        g_xpu = g.detach().clone().to("xpu").requires_grad_(True)
        w_xpu, n_xpu = torch._weight_norm_interface(v_xpu, g_xpu, dim=0)
        w_xpu.backward(gw.to("xpu"))
        self.assertEqual(w, w_xpu.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(n, n_xpu.cpu(), atol=1e-1, rtol=1e-5)
        self.assertEqual(v.grad, v_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(g.grad, g_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)

    def test_weight_norm_dim1(self):
        v = torch.randn(8193 * 253, 32).requires_grad_(True)
        g = torch.randn(32).requires_grad_(True)
        gw = torch.randn(8193 * 253, 32)
        w, n = torch._weight_norm_interface(v, g, dim=1)
        w.backward(gw)
        v_xpu = v.detach().clone().to("xpu").requires_grad_(True)
        g_xpu = g.detach().clone().to("xpu").requires_grad_(True)
        w_xpu, n_xpu = torch._weight_norm_interface(v_xpu, g_xpu, dim=1)
        w_xpu.backward(gw.to("xpu"))
        self.assertEqual(w, w_xpu.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(n, n_xpu.cpu(), atol=1e-1, rtol=1e-5)
        self.assertEqual(v.grad, v_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(g.grad, g_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)

    def test_weight_norm_dim2(self):
        v = torch.randn(8193, 253, 32).requires_grad_(True)
        g = torch.randn(32).requires_grad_(True)
        gw = torch.randn(8193, 253, 32)
        w, n = torch._weight_norm_interface(v, g, dim=2)
        w.backward(gw)
        v_xpu = v.detach().clone().to("xpu").requires_grad_(True)
        g_xpu = g.detach().clone().to("xpu").requires_grad_(True)
        w_xpu, n_xpu = torch._weight_norm_interface(v_xpu, g_xpu, dim=2)
        w_xpu.backward(gw.to("xpu"))
        self.assertEqual(w, w_xpu.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(n, n_xpu.cpu(), atol=1e-1, rtol=1e-5)
        self.assertEqual(v.grad, v_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(g.grad, g_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)

    def test_weight_norm_large_batch(self):
        v = torch.randn(8193, 8193).requires_grad_(True)
        g = torch.randn(8193).requires_grad_(True)
        gw = torch.randn(8193, 8193)
        w, n = torch._weight_norm_interface(v, g, dim=0)
        w.backward(gw)
        v_xpu = v.detach().clone().to("xpu").requires_grad_(True)
        g_xpu = g.detach().clone().to("xpu").requires_grad_(True)
        w_xpu, n_xpu = torch._weight_norm_interface(v_xpu, g_xpu, dim=0)
        w_xpu.backward(gw.to("xpu"))
        self.assertEqual(w, w_xpu.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(n, n_xpu.cpu(), atol=1e-1, rtol=1e-5)
        self.assertEqual(v.grad, v_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)
        self.assertEqual(g.grad, g_xpu.grad.cpu(), atol=1e-3, rtol=1e-5)
