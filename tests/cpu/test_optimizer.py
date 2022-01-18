import torch
import intel_extension_for_pytorch as ipex  # flake8: noqa
import itertools
import unittest
from torch.testing._internal.common_utils import TestCase
from common_utils import TestModule
import bench.custom_op_bench.optimizer

class TestOptimizers(TestCase):

    def _test_update(self, module, optimizer, dtype, split_master_weight_for_bf16, set_to_none):
        atol, rtol = None, None
        if dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        ipex_module, ipex_optimizer = ipex.optimize(module, dtype=dtype, optimizer=optimizer, split_master_weight_for_bf16=split_master_weight_for_bf16)
        for i in range(2):
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                # torch optmizer
                y = module(*module.input).sum()
                optimizer.zero_grad(set_to_none=set_to_none)
                y.backward()
                optimizer.step()
                # ipex optimizer
                y1 = ipex_module(*ipex_module.input).sum()
                ipex_optimizer.zero_grad(set_to_none=set_to_none)
                y1.backward()
                ipex_optimizer.step()
        origin_model_state = module.state_dict()
        ipex_model_state = ipex_module.state_dict()
        for var_name in origin_model_state:
            self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], atol=atol, rtol=rtol)
        origin_optimizer_state = optimizer.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in origin_optimizer_state:
            if var_name == 'state':
                self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name], atol=atol, rtol=rtol)


    def test_sgd(self):
        M = TestModule()
        options1 = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [0.1, 0], [False])
        options2 = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1], [0.1, 0], [0], [True])
        for set_to_none, split_master_weight_for_bf16, dtype, momentum, weight_decay, dampening, nesterov in list(options1) + list(options2):
            sgd = torch.optim.SGD(
                M.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay,
                dampening=dampening, nesterov=nesterov)
            self._test_update(M, sgd, dtype, split_master_weight_for_bf16, set_to_none)

    def test_adagrad(self):
        M = TestModule()
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [0.1, 0], [1e-5, 0])
        for set_to_none, split_master_weight_for_bf16, dtype, lr_decay, weight_decay, initial_accumulator_value, eps in options:
            adagrad = torch.optim.Adagrad(
                M.parameters(), lr=0.001, lr_decay=lr_decay, weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value, eps=eps)
            self._test_update(M, adagrad, dtype, split_master_weight_for_bf16, set_to_none)

    def test_lamb(self):
        M = TestModule()
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [(0.1, 0.111), (0.9, 0.999)], [1e-8], [0, 0.1], [True, False])
        for set_to_none, split_master_weight_for_bf16, dtype, betas, eps, weight_decay, fused in options:
            lamb = ipex.optim._lamb.Lamb(
                M.parameters(), lr=0.001, betas=betas, eps=eps,
                weight_decay=weight_decay, fused=fused)
            self._test_update(M, lamb, dtype, split_master_weight_for_bf16, set_to_none)

class TestFusedSteps(TestCase):

    def test_lamb_step(self):
        fused = torch.ops.torch_ipex.lamb_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_lamb

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

        # fused and non-contiguous fp32 args
        param5 = param.clone().t().contiguous().t()
        grad5 = grad.clone().t().contiguous().t()
        exp_avg5 = exp_avg.clone().t().contiguous().t()
        exp_avg_sq5 = exp_avg_sq.clone().t().contiguous().t()

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
        fused(param5, exp_avg5, exp_avg_sq5, grad5, trail, step, beta1, beta2, learning_rate, weight_decay, eps)

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

        # compare fused contiguous and fused non-contiguous()
        self.assertEqual(param, param5)
        self.assertEqual(exp_avg, exp_avg5)
        self.assertEqual(exp_avg_sq, exp_avg_sq5)

    def test_adagrad_step(self):
        fused = torch.ops.torch_ipex.adagrad_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_adagrad

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

        # compare fused contiguous and fused non-contiguous()
        param5 = param.clone().t().contiguous().t()
        grad5 = grad.clone().t().contiguous().t()
        state_sum5 = state_sum.clone().t().contiguous().t()

        step = 10
        learning_rate = 0.1
        weight_decay = 0.3
        lr_decay = 0.01
        eps = 0.001

        fused(param, grad, state_sum, trail, step, learning_rate, weight_decay, lr_decay, eps)
        fused(param2, grad2, state_sum2, trail2, step, learning_rate, weight_decay, lr_decay, eps)
        fused(param3, grad3, state_sum3, bf16_param, step, learning_rate, weight_decay, lr_decay, eps)
        non_fused(param4, grad4, state_sum4, step, learning_rate, weight_decay, lr_decay, eps)
        fused(param5, grad5, state_sum5, trail, step, learning_rate, weight_decay, lr_decay, eps)

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
        # compare fused contiguous and fused non-contiguous()
        self.assertEqual(param, param5)
        self.assertEqual(state_sum, state_sum5)

    def test_sgd_step(self):
        fused = torch.ops.torch_ipex.sgd_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_sgd

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

        # compare fused contiguous and fused non-contiguous()
        param5 = param.clone().t().contiguous().t()
        grad5 = grad.clone().t().contiguous().t()
        momentum_buf5 = momentum_buf.clone().t().contiguous().t()
        trail5 = torch.Tensor()

        learning_rate = 0.1
        weight_decay = 0.3
        momentum = 0.5
        dampening = 0.5
        nesterov = True

        fused(param, grad, momentum_buf, trail, momentum, learning_rate, weight_decay, dampening, nesterov)
        fused(param2, grad2, momentum_buf2, trail2, momentum, learning_rate, weight_decay, dampening, nesterov)
        fused(param3, grad3, momentum_buf3, bf16_param, momentum, learning_rate, weight_decay, dampening, nesterov)
        non_fused(param4, grad4, momentum_buf4, momentum, learning_rate, weight_decay, dampening, nesterov)
        fused(param5, grad5, momentum_buf5, trail, momentum, learning_rate, weight_decay, dampening, nesterov)

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
        # compare fused contiguous and fused non-contiguous()
        self.assertEqual(param, param5)
        self.assertEqual(momentum_buf, momentum_buf5)

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
