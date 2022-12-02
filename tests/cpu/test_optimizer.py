import torch
import intel_extension_for_pytorch as ipex  # flake8: noqa
import itertools
import unittest
from torch.testing._internal.common_utils import TestCase
from common_utils import TestModule
import bench.custom_op_bench.optimizer
from torch.optim import Adadelta, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
import copy

class TestOptimizers(TestCase):

    def _test_update(self, module, optimizer, dtype, split_master_weight_for_bf16, set_to_none, fused):
        atol, rtol = None, None
        if dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        ipex_module, ipex_optimizer = ipex.optimize(module, dtype=dtype, optimizer=optimizer, split_master_weight_for_bf16=split_master_weight_for_bf16, fuse_update_step=fused)
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
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [0.1, 0], [True, False], [True, False], [True, False], [True, False])
        for set_to_none, split_master_weight_for_bf16, dtype, momentum, weight_decay, dampening, nesterov, foreach, maximize, fused in options:
            if nesterov and (momentum <= 0 or dampening != 0):
                # dose not support such configs
                continue
            sgd = torch.optim.SGD(
                M.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay,
                dampening=dampening, nesterov=nesterov, foreach=foreach, maximize=maximize)
            self._test_update(M, sgd, dtype, split_master_weight_for_bf16, set_to_none, fused=fused)

    def test_sgd_fallback(self):
        # for sparse grad with weight_decay/momentum !=0, stock pytorch will also failed
        M = TestModule(has_sparse_grad=True)
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1, 0], [True, False], [True, False])
        for set_to_none, split_master_weight_for_bf16, dtype, dampening, foreach, maximize in options:
            if foreach:
                # stock pytorch will fail while foreach and has_sparse_grad
                continue
            sgd = torch.optim.SGD(
                M.parameters(), lr=0.001,
                dampening=dampening, foreach=foreach, maximize=maximize)
            self._test_update(M, sgd, dtype, split_master_weight_for_bf16, set_to_none, fused=True)

    def test_adagrad(self):
        M = TestModule()
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [0.1, 0], [1e-5, 0], [True, False], [True, False], [True])
        for set_to_none, split_master_weight_for_bf16, dtype, lr_decay, weight_decay, initial_accumulator_value, eps, foreach, maximize, fused in options:
            adagrad = torch.optim.Adagrad(
                M.parameters(), lr=0.001, lr_decay=lr_decay, weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value, eps=eps,
                foreach=foreach, maximize=maximize)
            self._test_update(M, adagrad, dtype, split_master_weight_for_bf16, set_to_none, fused)

    def test_adagrad_fallback(self):
        M = TestModule(has_sparse_grad=True)
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [0.1, 0], [0.1, 0], [1e-5, 0], [True, False])
        for set_to_none, split_master_weight_for_bf16, dtype, lr_decay, initial_accumulator_value, eps, maximize in options:
            adagrad = torch.optim.Adagrad(
                M.parameters(), lr=0.001, lr_decay=lr_decay,
                initial_accumulator_value=initial_accumulator_value, eps=eps,
                maximize=maximize)
            self._test_update(M, adagrad, dtype, split_master_weight_for_bf16, set_to_none, fused=True)

    def test_lamb(self):
        M = TestModule()
        options = itertools.product([True, False], [True, False], [torch.float, torch.bfloat16], [(0.1, 0.111), (0.9, 0.999)], [1e-8], [0, 0.1], [True, False])
        for set_to_none, split_master_weight_for_bf16, dtype, betas, eps, weight_decay, fused in options:
            lamb = ipex.optim._lamb.Lamb(
                M.parameters(), lr=0.001, betas=betas, eps=eps,
                weight_decay=weight_decay, fused=fused)
            self._test_update(M, lamb, dtype, split_master_weight_for_bf16, set_to_none, fused)

    def test_adam(self):
        M = TestModule()
        options = itertools.product([True, False], [True, False], [True, False], [torch.float, torch.bfloat16], [(0.1, 0.111), (0.9, 0.999)], [1e-8], [0, 0.1], [True, False], [True, False], [True, False])
        for set_to_none, split_master_weight_for_bf16, amsgrad, dtype, betas, eps, weight_decay, foreach, maximize, fused in options:
            if foreach:
                # there is a bug for foreach option in stock PT： https://github.com/pytorch/pytorch/issues/78807
                continue
            adam = torch.optim.Adam(
                M.parameters(), lr=0.001, betas=betas, eps=eps, weight_decay=weight_decay,
                amsgrad=amsgrad, foreach=foreach, maximize=maximize)
            self._test_update(M, adam, dtype, split_master_weight_for_bf16, set_to_none, fused)

class TestFusedSteps(TestCase):

    def test_lamb_step(self):
        fused = torch.ops.torch_ipex.lamb_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_lamb

        # fused fp32 args
        param = torch.randn(31, 33)
        grad = torch.randn(31, 33)
        exp_avg = torch.randn(31, 33).abs()
        exp_avg_sq = torch.randn(31, 33).abs()
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

        # fused double args
        param = torch.randn(31, 33).double()
        grad = torch.randn(31, 33).double()
        exp_avg = torch.randn(31, 33).double().abs()
        exp_avg_sq = torch.randn(31, 33).double().abs()
        trail = torch.Tensor()

        # non-fused double params
        param2 = param.clone()
        grad2 = grad.clone()
        exp_avg2 = exp_avg.clone()
        exp_avg_sq2 = exp_avg_sq.clone()

        fused(param, exp_avg, exp_avg_sq, grad, trail, step, beta1, beta2, learning_rate, weight_decay, eps)
        non_fused(param2, exp_avg2, exp_avg_sq2, grad2, step, beta1, beta2, learning_rate, weight_decay, eps)

        # compare fused and non-fused for double
        self.assertEqual(param, param2)
        self.assertEqual(exp_avg, exp_avg2)
        self.assertEqual(exp_avg_sq, exp_avg_sq2)

    def test_adam_step(self):
        fused = torch.ops.torch_ipex.adam_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_adam

        # fused fp32 args
        param = torch.randn(31, 33)
        grad = torch.randn(31, 33)
        exp_avg = torch.randn(31, 33).abs()
        exp_avg_sq = torch.randn(31, 33).abs()
        max_exp_avg_sq = torch.randn(31, 33).abs()
        trail = torch.Tensor()

        # fused bf16 params (master weight split)
        param2, trail2 = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        exp_avg2 = exp_avg.clone()
        exp_avg_sq2 = exp_avg_sq.clone()
        max_exp_avg_sq2 = max_exp_avg_sq.clone()

        # fused bf16 params (master weight)
        param3 = param.clone()
        grad3 = grad.bfloat16()
        exp_avg3 = exp_avg.clone()
        exp_avg_sq3 = exp_avg_sq.clone()
        max_exp_avg_sq3 = max_exp_avg_sq.clone()
        bf16_param = param3.bfloat16()

        # non-fused fp32 params
        param4 = param.clone()
        grad4 = grad.clone()
        exp_avg4 = exp_avg.clone()
        exp_avg_sq4 = exp_avg_sq.clone()
        max_exp_avg_sq4 = max_exp_avg_sq.clone()

        # fused and non-contiguous fp32 args
        param5 = param.clone().t().contiguous().t()
        grad5 = grad.clone().t().contiguous().t()
        exp_avg5 = exp_avg.clone().t().contiguous().t()
        exp_avg_sq5 = exp_avg_sq.clone().t().contiguous().t()
        max_exp_avg_sq5 = max_exp_avg_sq.clone().t().contiguous().t()

        step = 10
        beta1 = 0.8
        beta2 = 0.9
        learning_rate = 0.1
        weight_decay = 0.3
        eps = 0.001
        amsgrad = True
        fused(param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad, trail, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)
        fused(param2, exp_avg2, exp_avg_sq2, max_exp_avg_sq2, grad2, trail2, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)
        fused(param3, exp_avg3, exp_avg_sq3, max_exp_avg_sq3, grad3, bf16_param, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)
        non_fused(param4, exp_avg4, exp_avg_sq4, max_exp_avg_sq4, grad4, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)
        fused(param5, exp_avg5, exp_avg_sq5, max_exp_avg_sq5, grad5, trail, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)

        # compare fused and non-fused
        self.assertEqual(param, param4)
        self.assertEqual(exp_avg, exp_avg4)
        self.assertEqual(exp_avg_sq, exp_avg_sq4)
        self.assertEqual(max_exp_avg_sq, max_exp_avg_sq4)
        # compare fused fp32 and fused bf16
        self.assertEqual(param, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg, exp_avg2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg_sq, exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(max_exp_avg_sq, max_exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)
        # compare split vs non-split
        self.assertEqual(param3, param2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg3, exp_avg2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(exp_avg_sq3, exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)
        self.assertEqual(max_exp_avg_sq3, max_exp_avg_sq2.float(), rtol=1e-4, atol=1e-1)
        # make sure bf16_param are updated
        self.assertEqual(bf16_param, param3.bfloat16())

        # compare fused contiguous and fused non-contiguous()
        self.assertEqual(param, param5)
        self.assertEqual(exp_avg, exp_avg5)
        self.assertEqual(exp_avg_sq, exp_avg_sq5)
        self.assertEqual(max_exp_avg_sq, max_exp_avg_sq5)

        # fused double args
        param = torch.randn(31, 33).double()
        grad = torch.randn(31, 33).double()
        exp_avg = torch.randn(31, 33).double().abs()
        exp_avg_sq = torch.randn(31, 33).double().abs()
        max_exp_avg_sq = torch.randn(31, 33).double().abs()
        trail = torch.Tensor()

        # non-fused double params
        param2 = param.clone()
        grad2 = grad.clone()
        exp_avg2 = exp_avg.clone()
        exp_avg_sq2 = exp_avg_sq.clone()
        max_exp_avg_sq2 = max_exp_avg_sq.clone()

        fused(param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad, trail, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)
        non_fused(param2, exp_avg2, exp_avg_sq2, max_exp_avg_sq2, grad2, amsgrad, step, beta1, beta2, learning_rate, weight_decay, eps)

        # compare fused and non-fused for double
        self.assertEqual(param, param2)
        self.assertEqual(exp_avg, exp_avg2)
        self.assertEqual(exp_avg_sq, exp_avg_sq2)
        self.assertEqual(max_exp_avg_sq, max_exp_avg_sq2)

    def test_adagrad_step(self):
        fused = torch.ops.torch_ipex.adagrad_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_adagrad

        # fused fp32 args
        param = torch.randn(31, 33)
        grad = torch.randn(31, 33)
        state_sum = torch.randn(31, 33).abs()
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

        # fused double args
        param = torch.randn(31, 33).double()
        grad = torch.randn(31, 33).double()
        state_sum = torch.randn(31, 33).double().abs()

        # non-fused double params
        param2 = param.clone()
        grad2 = grad.clone()
        state_sum2 = state_sum.clone()

        fused(param, grad, state_sum, trail, step, learning_rate, weight_decay, lr_decay, eps)
        non_fused(param2, grad2, state_sum2, step, learning_rate, weight_decay, lr_decay, eps)

        # compare fused and non-fused for double
        self.assertEqual(param, param2)
        self.assertEqual(state_sum, state_sum2)

    def test_sgd_step(self):
        fused = torch.ops.torch_ipex.sgd_fused_step
        non_fused = bench.custom_op_bench.optimizer.non_fused_sgd

        # fused fp32 args
        param = torch.randn(31, 33)
        grad = torch.randn(31, 33)
        momentum_buf = torch.randn(31, 33)
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

        # fused double args
        param = torch.randn(31, 33).double()
        grad = torch.randn(31, 33).double()
        momentum_buf = torch.randn(31, 33).double().abs()

        # non-fused double params
        param2 = param.clone()
        grad2 = grad.clone()
        momentum_buf2 = momentum_buf.clone()

        fused(param, grad, momentum_buf, trail, momentum, learning_rate, weight_decay, dampening, nesterov)
        non_fused(param2, grad2, momentum_buf2, momentum, learning_rate, weight_decay, dampening, nesterov)

        # compare fused and non-fused for double
        self.assertEqual(param, param2)
        self.assertEqual(momentum_buf, momentum_buf2)

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
        param = torch.randn(31, 33)
        grad = torch.randn(31, 33)
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16()
        self._test_packed_add(param, grad, param2, trail, grad2)

        # transposed case
        # fp32 args
        param = torch.randn(31, 33).t().contiguous().t()
        grad = torch.randn(31, 33).t().contiguous().t()
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(param)
        grad2 = grad.bfloat16().t().contiguous().t()
        self._test_packed_add(param, grad, param2, trail, grad2)

        # sliced-out case
        # fp32 args
        base_param = torch.randn(31, 33)
        base_grad = torch.randn(31, 33)
        param = base_param[10:20, 10:20]
        grad = base_grad[10:20, 10:20]
        # bf16 args
        param2, trail = torch.ops.torch_ipex.split_float_bfloat16(base_param)
        param2 = param2[10:20, 10:20]
        trail = trail[10:20, 10:20]
        grad2 = base_grad.bfloat16()[10:20, 10:20]
        self._test_packed_add(param, grad, param2, trail, grad2)

class TestPatchedMethod(TestCase):

    def test_zero_grad(self):

        def count_zero_grad(evt_list):
            count = 0
            for evt in evt_list:
                if 'zero_grad' in evt.name:
                    count +=1
            return count

        M = TestModule().train()
        optimizers_list = [Adadelta, AdamW, Adamax, ASGD, RMSprop, Rprop]
        for optimizer, set_to_none in itertools.product(optimizers_list, [True, False]):
            ori_model = copy.deepcopy(M)
            ori_optimizer = optimizer(ori_model.parameters(), lr=0.1)
            ipex_model, ipex_optimizer = ipex.optimize(ori_model, torch.bfloat16, ori_optimizer)

            # original
            with torch.cpu.amp.autocast():
                y = ori_model(*ori_model.input).sum()
            y.backward()
            with torch.autograd.profiler.profile() as ori_prof:
                ori_optimizer.zero_grad(set_to_none=set_to_none)
            
            # ipex
            with torch.cpu.amp.autocast():
                y1 = ipex_model(*ipex_model.input).sum()
            y1.backward()
            # check grad are correctly attached
            for param in ipex_model.parameters():
                self.assertTrue(param.grad != None)
            uncast_weight = [ipex_model.bn.weight.data_ptr(), ipex_model.bn.bias.data_ptr()]
            for param in ipex_optimizer.param_groups[0]['params']:
                if param.data_ptr() not in uncast_weight:
                    self.assertTrue(param.grad == None)
                    self.assertTrue(ipex_optimizer.params_attr[param]['bf16_param'].grad != None)
                else:
                    self.assertTrue(param.grad != None)

            with torch.autograd.profiler.profile() as ipex_prof:
                ipex_optimizer.zero_grad(set_to_none=set_to_none)
            # check grad are zeroed or are set to none
            for param in ipex_model.parameters():
                expected_grad = None if set_to_none else torch.zeros_like(param)
                self.assertEqual(expected_grad, param.grad)

            for param in ipex_optimizer.param_groups[0]['params']:
                if param.data_ptr() not in uncast_weight:
                    expected_grad = None if set_to_none else torch.zeros_like(param).bfloat16()
                    self.assertEqual(expected_grad, ipex_optimizer.params_attr[param]['bf16_param'].grad)
                else:
                    expected_grad = None if set_to_none else torch.zeros_like(param)
                    self.assertEqual(expected_grad, param.grad)

            # check the num of calls for 'zero_grad' are same
            self.assertEqual(count_zero_grad(ori_prof.function_events), count_zero_grad(ipex_prof.function_events))

if __name__ == '__main__':
    test = unittest.main()
