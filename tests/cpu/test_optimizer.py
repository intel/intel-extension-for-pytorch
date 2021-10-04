import torch
import intel_extension_for_pytorch as ipex
import unittest
import copy
from torch.testing._internal.common_utils import TestCase

class TestOptimizer(TestCase):

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


    def test_lamb(self):
        fused = torch.ops.torch_ipex.lamb_fused_step
        non_fused = self.non_fused_lamb

        # fused fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        exp_avg = torch.randn(80, 100).abs()
        exp_avg_sq = torch.randn(80, 100).abs()
        trail = torch.Tensor()

        # fused bf16 params
        param2, trail2 = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        exp_avg2 = exp_avg.clone()
        exp_avg_sq2 = exp_avg_sq.clone()

        # non-fused fp32 params
        param3 = param.clone()
        grad3 = grad.clone()
        exp_avg3 = exp_avg.clone()
        exp_avg_sq3 = exp_avg_sq.clone()

        step = 10
        beta1 = 0.8
        beta2 = 0.9
        learning_rate = 0.1
        weight_decay = 0.3
        eps = 0.001

        fused(param, exp_avg, exp_avg_sq, grad, trail, step, beta1, beta2, learning_rate, weight_decay, eps)
        fused(param2, exp_avg2, exp_avg_sq2, grad2, trail2, step, beta1, beta2, learning_rate, weight_decay, eps)
        non_fused(param3, exp_avg3, exp_avg_sq3, grad3, step, beta1, beta2, learning_rate, weight_decay, eps)

        # compare fused and non-fused
        self.assertEqual(param, param3)
        self.assertEqual(exp_avg, exp_avg3)
        self.assertEqual(exp_avg_sq, exp_avg_sq3)

        # compare fused fp32 and fused bf16
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg, exp_avg2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg_sq, exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)

    def test_adagrad(self):
        fused = torch.ops.torch_ipex.adagrad_fused_step
        non_fused = self.non_fused_adagrad

        # fused fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        state_sum = torch.randn(80, 100).abs()
        trail = torch.Tensor()

        # fused bf16 args
        param2, trail2 = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        state_sum2 = state_sum.clone()

        # non-fused fp32 args
        param3 = param.clone()
        grad3 = grad.clone()
        state_sum3 = state_sum.clone()
        trail3 = torch.randn(80, 100).clone()

        step = 10
        learning_rate = 0.1
        weight_decay = 0.3
        lr_decay = 0.01
        eps = 0.001

        fused(param, grad, state_sum, trail, step, learning_rate, weight_decay, lr_decay, eps)
        fused(param2, grad2, state_sum2, trail2, step, learning_rate, weight_decay, lr_decay, eps)
        non_fused(param3, grad3, state_sum3, step, learning_rate, weight_decay, lr_decay, eps)

        # compare fused fp32 vs non-fused fp32
        self.assertEqual(param, param3)
        self.assertEqual(state_sum, state_sum3)
        # compare fused fp32 vs fused bf16  fused
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(state_sum, state_sum2.float(), rtol=1e-4, atol=1e-1)

    def _test_split_sgd(self, param, grad, param2, trail, grad2):
        packed_add = torch.ops.torch_ipex.packed_add
        non_fused = self.non_fused_adagrad

        learning_rate = 0.1

        param.add_(grad, alpha=-learning_rate)
        packed_add(param2, trail, grad2, alpha=-learning_rate)

        # compare fp32 vs bf16 fused
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)

    def test_split_sgd(self):
        # contiguous case
        # fp32 args
        param = torch.randn(80, 100)
        grad = torch.randn(80, 100)
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        self._test_split_sgd(param, grad, param2, trail, grad2)

        # transposed case
        # fp32 args
        param = torch.randn(80, 100).t().contiguous().t()
        grad = torch.randn(80, 100).t().contiguous().t()
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16().t().contiguous().t()
        self._test_split_sgd(param, grad, param2, trail, grad2)

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
        self._test_split_sgd(param, grad, param2, trail, grad2)

if __name__ == '__main__':
    test = unittest.main()
