import torch
import intel_extension_for_pytorch as ipex  # flake8: noqa
import itertools
import unittest
from torch.testing._internal.common_utils import TestCase
from common_utils import TestModule

class TestOptimizers(TestCase):

    def _test_update(self, module, optimizer, dtype):
        ipex_module, ipex_optimizer = ipex.optimize(module, dtype=dtype, optimizer=optimizer)
        with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
            # torch optmizer
            module.attach_grad()
            optimizer.step()
            # ipex optimizer
            ipex_module.attach_grad(dtype)
            ipex_optimizer.step()
        origin_model_state = module.state_dict()
        ipex_model_state = ipex_module.state_dict()
        for var_name in origin_model_state:
            self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-3, atol=1e-3)
        origin_optimizer_state = optimizer.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in origin_optimizer_state:
            if var_name == 'state':
                self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], rtol=1e-3, atol=1e-3)


    def test_sgd(self):
        M = TestModule()
        options1 = itertools.product([torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [0.1, 0], [False])
        options2 = itertools.product([torch.float, torch.bfloat16], [0.1], [0.1, 0], [0], [True])
        for dtype, momentum, weight_decay, dampening, nesterov in list(options1) + list(options2):
            sgd = torch.optim.SGD(
              M.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay,
              dampening=dampening, nesterov=nesterov)
            self._test_update(M, sgd, dtype)

    def test_adagrad(self):
        M = TestModule()
        options = itertools.product([torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [0.1, 0], [1e-5, 0])
        for dtype, lr_decay, weight_decay, initial_accumulator_value, eps in options:
            adagrad = torch.optim.Adagrad(
              M.parameters(), lr=0.001, lr_decay=lr_decay, weight_decay=weight_decay,
              initial_accumulator_value=initial_accumulator_value, eps=eps)
            self._test_update(M, adagrad, dtype)

    def test_lamb(self):
        M = TestModule()
        options = itertools.product([torch.bfloat16], [(0.1, 0.111), (0.9, 0.999)], [0, 1e-8], [0, 0.1], [False])
        for dtype, betas, eps, weight_decay, fused in options:
            lamb = ipex.optim.Lamb(
              M.parameters(), lr=0.001, betas=betas, eps=eps,
              weight_decay=weight_decay, fused=fused)
            self._test_update(M, lamb, dtype)

class TestFusedSteps(TestCase):

    def non_fused_lamb(self, param, exp_avg, exp_avg_sq, grad, step, beta1, beta2, lr, weight_decay, eps):
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        adam_step = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)
        if weight_decay != 0:
            adam_step.add_(param, alpha=weight_decay)
        weight_norm = param.norm(p=2)
        rtw_norm = adam_step.norm(p=2)
        true_ratio = weight_norm / rtw_norm
        param.add_(adam_step, alpha=-lr * true_ratio)

    def non_fused_adagrad(self, param, grad, state_sum, step, lr, weight_decay, lr_decay, eps):
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        clr = lr / (1 + (step - 1) * lr_decay)
        state_sum.addcmul_(grad, grad, value=1)
        std = state_sum.sqrt().add_(eps)
        param.addcdiv_(grad, std, value=-clr)

    def non_fused_sgd(self, param, grad, momentum_buf, momentum, lr, weight_decay, dampening, nesterov):
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buf

            if buf is None:
                buf = torch.clone(grad).detach()
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        param.add_(grad, alpha=-lr)

    def test_lamb_step(self):
        fused = torch.ops.torch_ipex.lamb_fused_step
        non_fused = self.non_fused_lamb

        # fused fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        exp_avg = torch.randn(80, 100).abs()
        exp_avg_sq = torch.randn(80, 100).abs()
        trail = torch.Tensor()

        # fused bf16 params (master weight split)
        param2, trail2 = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        exp_avg2 = exp_avg.clone()
        exp_avg_sq2 = exp_avg_sq.clone()

        # fused bf16 params (master weight)
        param3 = param.clone()
        grad3 = grad.bfloat16()
        exp_avg3 = exp_avg.clone()
        exp_avg_sq3 = exp_avg_sq.clone()
        bf16_param = param3.bfloat16()

        # non-fused fp32 params
        param4 = param.clone()
        grad4 = grad.clone()
        exp_avg4 = exp_avg.clone()
        exp_avg_sq4 = exp_avg_sq.clone()

        step = 10
        beta1 = 0.8
        beta2 = 0.9
        learning_rate = 0.1
        weight_decay = 0.3
        eps = 0.001

        fused(param, exp_avg, exp_avg_sq, grad, trail, step, beta1, beta2, learning_rate, weight_decay, eps)
        fused(param2, exp_avg2, exp_avg_sq2, grad2, trail2, step, beta1, beta2, learning_rate, weight_decay, eps)
        fused(param3, exp_avg3, exp_avg_sq3, grad3, bf16_param, step, beta1, beta2, learning_rate, weight_decay, eps)
        non_fused(param4, exp_avg4, exp_avg_sq4, grad4, step, beta1, beta2, learning_rate, weight_decay, eps)

        # compare fused and non-fused
        self.assertEqual(param, param4)
        self.assertEqual(exp_avg, exp_avg4)
        self.assertEqual(exp_avg_sq, exp_avg_sq4)
        # compare fused fp32 and fused bf16
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg, exp_avg2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg_sq, exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)
        # compare split vs non-split
        self.assertEqual(param3, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg3, exp_avg2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg_sq3, exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)
        # make sure bf16_param are updated
        self.assertEqual(bf16_param, param3.bfloat16())

    def test_adagrad_step(self):
        fused = torch.ops.torch_ipex.adagrad_fused_step
        non_fused = self.non_fused_adagrad

        # fused fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        state_sum = torch.randn(80, 100).abs()
        trail = torch.Tensor()

        # fused bf16 args( master weight split )
        param2, trail2 = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        state_sum2 = state_sum.clone()

        # fused bf16 args( master weight )
        param3 = param.clone()
        grad3 = grad.bfloat16()
        state_sum3 = state_sum.clone()
        bf16_param = param3.bfloat16()

        # non-fused fp32 args
        param4 = param.clone()
        grad4 = grad.clone()
        state_sum4 = state_sum.clone()

        step = 10
        learning_rate = 0.1
        weight_decay = 0.3
        lr_decay = 0.01
        eps = 0.001

        fused(param, grad, state_sum, trail, step, learning_rate, weight_decay, lr_decay, eps)
        fused(param2, grad2, state_sum2, trail2, step, learning_rate, weight_decay, lr_decay, eps)
        fused(param3, grad3, state_sum3, bf16_param, step, learning_rate, weight_decay, lr_decay, eps)
        non_fused(param4, grad4, state_sum4, step, learning_rate, weight_decay, lr_decay, eps)

        # compare fused fp32 vs non-fused fp32
        self.assertEqual(param, param4)
        self.assertEqual(state_sum, state_sum4)
        # compare fused fp32 vs fused bf16  fused
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(state_sum, state_sum2.float(), rtol=1e-4, atol=1e-1)
        # compare split vs non-split
        self.assertEqual(param3, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(state_sum3, state_sum2, rtol=1e-4, atol=1e-1)
        # make sure bf16_param are updated
        self.assertEqual(bf16_param, param3.bfloat16())

    def test_sgd_step(self):
        fused = torch.ops.torch_ipex.sgd_fused_step
        non_fused = self.non_fused_sgd

        # fused fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        momentum_buf = torch.randn(80, 100)
        trail = torch.Tensor()

        # fused bf16 args ( master weight split )
        param2, trail2 = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        momentum_buf2 = momentum_buf.clone()

        # fused bf16 args( master weight )
        param3 = param.clone()
        grad3 = grad.bfloat16()
        momentum_buf3 = momentum_buf.clone()
        bf16_param = param3.bfloat16()

        # non-fused fp32 args
        param4 = param.clone()
        grad4 = grad.clone()
        momentum_buf4 = momentum_buf.clone()

        learning_rate = 0.1
        weight_decay = 0.3
        momentum = 0.5
        dampening = 0.5
        nesterov = True

        fused(param, grad, momentum_buf, trail, momentum, learning_rate, weight_decay, dampening, nesterov)
        fused(param2, grad2, momentum_buf2, trail2, momentum, learning_rate, weight_decay, dampening, nesterov)
        fused(param3, grad3, momentum_buf3, bf16_param, momentum, learning_rate, weight_decay, dampening, nesterov)
        non_fused(param4, grad4, momentum_buf4, momentum, learning_rate, weight_decay, dampening, nesterov)

        # compare fused fp32 vs non-fused fp32
        self.assertEqual(param, param4)
        self.assertEqual(momentum_buf, momentum_buf4)
        # compare fused fp32 vs fused bf16
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(momentum_buf, momentum_buf2, rtol=1e-4, atol=1e-1)
        # compare split vs non-split
        self.assertEqual(param3, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(momentum_buf3, momentum_buf2, rtol=1e-4, atol=1e-1)
        # make sure bf16_param are updated
        self.assertEqual(bf16_param, param3.bfloat16())

    def _test_packed_add(self, param, grad, param2, trail, grad2):
        packed_add = torch.ops.torch_ipex.packed_add
        learning_rate = 0.1
        param.add_(grad, alpha=-learning_rate)
        packed_add(param2, trail, grad2, alpha=-learning_rate)
        # compare fp32 vs bf16 fused
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)

    def test_packed_add(self):
        # contiguous case
        # fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        self._test_packed_add(param, grad, param2, trail, grad2)

        # transposed case
        # fp32 args
        param = torch.randn(80, 100).t().contiguous().t()
        grad = torch.randn(80, 100).t().contiguous().t()
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16().t().contiguous().t()
        self._test_packed_add(param, grad, param2, trail, grad2)

        # sliced-out case
        # fp32 args
        base_param = torch.randn(80, 100)
        base_grad = torch.randn(80, 100)
        param = base_param[10:20, 10:20]
        grad = base_grad[10:20, 10:20]
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(base_param)
        param2 = param2[10:20, 10:20]
        trail = trail[10:20, 10:20]
        grad2 = base_grad.bfloat16()[10:20, 10:20]
        self._test_packed_add(param, grad, param2, trail, grad2)

if __name__ == '__main__':
    test = unittest.main()
