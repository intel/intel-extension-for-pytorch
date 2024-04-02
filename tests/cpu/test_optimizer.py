# This Python file uses the following encoding: utf-8
# !/usr/bin/env python
import torch
import intel_extension_for_pytorch as ipex  # flake8: noqa
import intel_extension_for_pytorch._C as core
import itertools
import unittest
from torch.testing._internal.common_utils import TestCase
from common_utils import TestModule, _empty_weight_bias_parameter_names
import bench.custom_op_bench.optimizer
from torch.optim import Adadelta, AdamW, Adamax, ASGD, RMSprop, Rprop
import copy


class TestOptimizers(TestCase):
    def _test_update(
        self, module, optimizer, dtype, split_master_weight_for_bf16, set_to_none, fused
    ):
        atol, rtol = None, None
        if dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        scaler = None
        if dtype == torch.float16:
            split_master_weight_for_bf16 = False
            scaler = torch.cpu.amp.GradScaler(init_scale=1)
        ipex_module, ipex_optimizer = ipex.optimize(
            module,
            dtype=dtype,
            optimizer=optimizer,
            split_master_weight_for_bf16=split_master_weight_for_bf16,
            fuse_update_step=fused,
        )
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
                if dtype == torch.float16:
                    scaler.scale(y1).backward()
                    scaler.step(ipex_optimizer)
                    scaler.update(new_scale=1.0)
                else:
                    y1.backward()
                    ipex_optimizer.step()
        gradscaler_inf = torch.float16 == dtype and sum(
            v.item() for v in scaler._check_inf_per_device(ipex_optimizer).values()
        )
        origin_model_state = module.state_dict()
        ipex_model_state = ipex_module.state_dict()
        if not gradscaler_inf:
            for var_name in origin_model_state:
                self.assertEqual(
                    origin_model_state[var_name],
                    ipex_model_state[var_name],
                    atol=atol,
                    rtol=rtol,
                )
        origin_optimizer_state = optimizer.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        if not gradscaler_inf:
            for var_name in origin_optimizer_state:
                if var_name == "state":
                    self.assertEqual(
                        origin_optimizer_state[var_name],
                        ipex_optimizer_state[var_name],
                        atol=atol,
                        rtol=rtol,
                    )

    def test_sgd(self):
        M = TestModule()
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        options = itertools.product(
            [True, False],
            [True, False],
            dtypes,
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        )
        for (
            set_to_none,
            split_master_weight_for_bf16,
            dtype,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            foreach,
            maximize,
            fused,
        ) in options:
            if nesterov and (momentum <= 0 or dampening != 0):
                # dose not support such configs
                continue
            sgd = torch.optim.SGD(
                M.parameters(),
                lr=0.001,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=dampening,
                nesterov=nesterov,
                foreach=foreach,
                maximize=maximize,
            )
            self._test_update(
                M, sgd, dtype, split_master_weight_for_bf16, set_to_none, fused=fused
            )

    def test_sgd_fallback(self):
        # for sparse grad with weight_decay/momentum !=0, stock pytorch will also failed
        M = TestModule(has_sparse_grad=True)
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        options = itertools.product(
            [True, False],
            [True, False],
            dtypes,
            [0.1, 0],
            [True, False],
            [True, False],
        )
        for (
            set_to_none,
            split_master_weight_for_bf16,
            dtype,
            dampening,
            foreach,
            maximize,
        ) in options:
            if foreach:
                # stock pytorch will fail while foreach and has_sparse_grad
                continue
            sgd = torch.optim.SGD(
                M.parameters(),
                lr=0.001,
                dampening=dampening,
                foreach=foreach,
                maximize=maximize,
            )
            self._test_update(
                M, sgd, dtype, split_master_weight_for_bf16, set_to_none, fused=True
            )

    def test_adagrad(self):
        M = TestModule()
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        options = itertools.product(
            [True, False],
            [True, False],
            dtypes,
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [1e-5, 0],
            [True, False],
            [True, False],
            [True],
        )
        for (
            set_to_none,
            split_master_weight_for_bf16,
            dtype,
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            eps,
            foreach,
            maximize,
            fused,
        ) in options:
            adagrad = torch.optim.Adagrad(
                M.parameters(),
                lr=0.001,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
                foreach=foreach,
                maximize=maximize,
            )
            self._test_update(
                M, adagrad, dtype, split_master_weight_for_bf16, set_to_none, fused
            )

    def test_adagrad_fallback(self):
        M = TestModule(has_sparse_grad=True)
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        options = itertools.product(
            [True, False],
            [True, False],
            dtypes,
            [0.1, 0],
            [0.1, 0],
            [1e-5, 0],
            [True, False],
        )
        for (
            set_to_none,
            split_master_weight_for_bf16,
            dtype,
            lr_decay,
            initial_accumulator_value,
            eps,
            maximize,
        ) in options:
            adagrad = torch.optim.Adagrad(
                M.parameters(),
                lr=0.001,
                lr_decay=lr_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
                maximize=maximize,
            )
            self._test_update(
                M, adagrad, dtype, split_master_weight_for_bf16, set_to_none, fused=True
            )

    def test_lamb(self):
        M = TestModule()
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        options = itertools.product(
            [True, False],
            [True, False],
            dtypes,
            [(0.1, 0.111), (0.9, 0.999)],
            [1e-8],
            [0, 0.1],
            [True, False],
        )
        for (
            set_to_none,
            split_master_weight_for_bf16,
            dtype,
            betas,
            eps,
            weight_decay,
            fused,
        ) in options:
            lamb = ipex.optim._lamb.Lamb(
                M.parameters(),
                lr=0.001,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=fused,
            )
            self._test_update(
                M, lamb, dtype, split_master_weight_for_bf16, set_to_none, fused
            )

    def test_adam(self):
        M = TestModule()
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        options = itertools.product(
            [True, False],
            [True, False],
            [True, False],
            dtypes,
            [(0.1, 0.111), (0.9, 0.999)],
            [1e-8],
            [0, 0.1],
            [True, False],
            [True, False],
            [True, False],
        )
        for (
            set_to_none,
            split_master_weight_for_bf16,
            amsgrad,
            dtype,
            betas,
            eps,
            weight_decay,
            foreach,
            maximize,
            fused,
        ) in options:
            if foreach:
                # there is a bug for foreach option in stock PT： https://github.com/pytorch/pytorch/issues/78807
                continue
            adam = torch.optim.Adam(
                M.parameters(),
                lr=0.001,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                foreach=foreach,
                maximize=maximize,
            )
            self._test_update(
                M, adam, dtype, split_master_weight_for_bf16, set_to_none, fused
            )

    def test_grad_scaling_unscale(self):
        inv_scale = torch.full((1,), 0.25, dtype=torch.float)
        found_inf = torch.full((1,), 0.0, dtype=torch.float)

        size = 20
        g = torch.full((size, size), 4.0, dtype=torch.float)
        ginf = g.clone()
        ginf[2, 2] = float("inf")
        gnan = g.clone()
        gnan[2, 2] = float("nan")

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            core._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        grads = [g.clone(), g.to(dtype=torch.float16)]
        core._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        for grad in grads:
            self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # If inject_inf >= 0, writes an inf into one grad for _unscale_grads_ to find.
        def perfect_storm_grads(inject_inf):
            grads = [
                g.clone(),
                g.clone()[:, :5],
                g.to(dtype=torch.float16),
                g.to(dtype=torch.float16),
            ]
            if inject_inf >= 0:
                grads[inject_inf][2, 2] = float("inf")
            return grads

        scaler = torch.cpu.amp.GradScaler()
        dummy_params = [torch.empty_like(g) for g in perfect_storm_grads(-1)]
        dummy_opt = torch.optim.SGD(dummy_params, lr=1.0)

        # Ensures the inf/nan checking can find an inf injected onto any grad in the perfect storm.
        for inject_inf in range(-1, len(dummy_params)):
            found_inf = torch.full((1,), 0.0, dtype=torch.float)
            grads = perfect_storm_grads(inject_inf)
            for i, p in enumerate(dummy_params):
                p.grad = grads[i]
            found_inf_per_device = scaler._unscale_grads_(
                dummy_opt, inv_scale, found_inf, True
            )
            if inject_inf < 0:
                # No inf was injected, ensures unscaling worked normally.
                self.assertTrue(
                    sum(v.item() for v in found_inf_per_device.values()) == 0
                )
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)
            else:
                # inf was injected, ensures inf was found.
                self.assertTrue(
                    sum(v.item() for v in found_inf_per_device.values()) == 1
                )

    def test_grad_scaling_unscale_sparse(self):
        scaler = torch.cpu.amp.GradScaler()

        inv_scale = torch.full((1,), 0.25, dtype=torch.float)
        found_inf = torch.empty((1,), dtype=torch.float)
        cur = found_inf.device

        i = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
        v = torch.tensor([16.0, 32.0, 64.0], dtype=torch.float)
        s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), dtype=torch.float)

        p = s.clone()
        assert p.is_sparse
        opt = torch.optim.SGD([p], lr=1.0)

        p.grad = s.clone()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertEqual(p.grad.to_dense(), (s / 4).to_dense())

        v = torch.FloatTensor([16.0, 32.0, float("inf")])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), dtype=torch.float)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        v = torch.FloatTensor([16.0, 32.0, float("nan")])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), dtype=torch.float)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        p = s.clone().half()
        assert p.is_sparse
        opt = torch.optim.SGD([p], lr=1.0)

        p.grad = s.clone().half()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertEqual(p.grad.to_dense(), (s.half() / 4).to_dense())

        # Creates fp16 sparse tensor with duplicated indices (uncoalesced).  The uncoalesced representation
        # does not overflow in fp16, but the coalesced representation would, because 64000 + 64000 > fp16 max.
        # _amp_non_finite_check_and_unscale_ should report an overflow here.
        i = torch.LongTensor([[0, 1, 0], [2, 0, 2]])
        v = torch.FloatTensor([64000.0, 32.0, 64000.0])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), dtype=torch.float16)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 1.0)

    def test_grad_scale_will_not_overflow(self):
        model = torch.nn.Linear(5, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cpu.amp.GradScaler(
            growth_interval=1, growth_factor=2**4, init_scale=1e38
        )
        optimizer.zero_grad()
        x = torch.randn(1, 5)
        y = 1e-30 * torch.randn(1, 1)
        l = ((model(x) - y) ** 2).mean()
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        assert scaler._scale != float("inf") and scaler._scale != float("nan")


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

        # fused fp32 zero params
        param6 = torch.zeros(31, 33)
        grad6 = torch.zeros(31, 33)
        exp_avg6 = torch.zeros(31, 33).abs()
        exp_avg_sq6 = torch.zeros(31, 33).abs()

        # non-fused fp32 zero params
        param7 = param6.clone()
        grad7 = grad6.clone()
        exp_avg7 = exp_avg6.clone()
        exp_avg_sq7 = exp_avg_sq6.clone()

        step = 10
        beta1 = 0.8
        beta2 = 0.9
        learning_rate = 0.1
        weight_decay = 0.3
        eps = 0.001

        fused(
            param,
            exp_avg,
            exp_avg_sq,
            grad,
            trail,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param2,
            exp_avg2,
            exp_avg_sq2,
            grad2,
            trail2,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param3,
            exp_avg3,
            exp_avg_sq3,
            grad3,
            bf16_param,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        non_fused(
            param4,
            exp_avg4,
            exp_avg_sq4,
            grad4,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param5,
            exp_avg5,
            exp_avg_sq5,
            grad5,
            trail,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param6,
            exp_avg6,
            exp_avg_sq6,
            grad6,
            trail,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        non_fused(
            param7,
            exp_avg7,
            exp_avg_sq7,
            grad7,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )

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

        # compare param6 and param7 has no nan
        self.assertFalse(param6.isnan().any())
        self.assertFalse(param7.isnan().any())

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

        # fused double zero args
        param3 = torch.zeros(31, 33).double()
        grad3 = torch.zeros(31, 33).double()
        exp_avg3 = torch.zeros(31, 33).double().abs()
        exp_avg_sq3 = torch.zeros(31, 33).double().abs()

        # non-fused double zero params
        param4 = param3.clone()
        grad4 = grad3.clone()
        exp_avg4 = exp_avg3.clone()
        exp_avg_sq4 = exp_avg_sq3.clone()

        fused(
            param,
            exp_avg,
            exp_avg_sq,
            grad,
            trail,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        non_fused(
            param2,
            exp_avg2,
            exp_avg_sq2,
            grad2,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param3,
            exp_avg3,
            exp_avg_sq3,
            grad3,
            trail,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        non_fused(
            param4,
            exp_avg4,
            exp_avg_sq4,
            grad4,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )

        # compare fused and non-fused for double
        self.assertEqual(param, param2)
        self.assertEqual(exp_avg, exp_avg2)
        self.assertEqual(exp_avg_sq, exp_avg_sq2)
        self.assertFalse(param3.isnan().any())
        self.assertFalse(param4.isnan().any())

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
        fused(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            trail,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param2,
            exp_avg2,
            exp_avg_sq2,
            max_exp_avg_sq2,
            grad2,
            trail2,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param3,
            exp_avg3,
            exp_avg_sq3,
            max_exp_avg_sq3,
            grad3,
            bf16_param,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        non_fused(
            param4,
            exp_avg4,
            exp_avg_sq4,
            max_exp_avg_sq4,
            grad4,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        fused(
            param5,
            exp_avg5,
            exp_avg_sq5,
            max_exp_avg_sq5,
            grad5,
            trail,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )

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

        fused(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            trail,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )
        non_fused(
            param2,
            exp_avg2,
            exp_avg_sq2,
            max_exp_avg_sq2,
            grad2,
            amsgrad,
            step,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            eps,
        )

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

        fused(
            param,
            grad,
            state_sum,
            trail,
            step,
            learning_rate,
            weight_decay,
            lr_decay,
            eps,
        )
        fused(
            param2,
            grad2,
            state_sum2,
            trail2,
            step,
            learning_rate,
            weight_decay,
            lr_decay,
            eps,
        )
        fused(
            param3,
            grad3,
            state_sum3,
            bf16_param,
            step,
            learning_rate,
            weight_decay,
            lr_decay,
            eps,
        )
        non_fused(
            param4, grad4, state_sum4, step, learning_rate, weight_decay, lr_decay, eps
        )
        fused(
            param5,
            grad5,
            state_sum5,
            trail,
            step,
            learning_rate,
            weight_decay,
            lr_decay,
            eps,
        )

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

        fused(
            param,
            grad,
            state_sum,
            trail,
            step,
            learning_rate,
            weight_decay,
            lr_decay,
            eps,
        )
        non_fused(
            param2, grad2, state_sum2, step, learning_rate, weight_decay, lr_decay, eps
        )

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

        fused(
            param,
            grad,
            momentum_buf,
            trail,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )
        fused(
            param2,
            grad2,
            momentum_buf2,
            trail2,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )
        fused(
            param3,
            grad3,
            momentum_buf3,
            bf16_param,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )
        non_fused(
            param4,
            grad4,
            momentum_buf4,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )
        fused(
            param5,
            grad5,
            momentum_buf5,
            trail,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )

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

        fused(
            param,
            grad,
            momentum_buf,
            trail,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )
        non_fused(
            param2,
            grad2,
            momentum_buf2,
            momentum,
            learning_rate,
            weight_decay,
            dampening,
            nesterov,
        )

        # compare fused and non-fused for double
        self.assertEqual(param, param2)
        self.assertEqual(momentum_buf, momentum_buf2)

    def _test_packed_add(self, param, grad, param2, trail, grad2):
        packed_add = torch.ops.torch_ipex.packed_add
        learning_rate = 0.1
        param.add_(grad, alpha=-learning_rate)
        packed_add(param2, trail, grad2, -learning_rate)
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
                if "zero_grad" in evt.name:
                    count += 1
            return count

        M = TestModule().train()
        optimizers_list = [Adadelta, AdamW, Adamax, ASGD, RMSprop, Rprop]
        for optimizer, set_to_none in itertools.product(optimizers_list, [True, False]):
            ori_model = copy.deepcopy(M)
            ori_optimizer = optimizer(ori_model.parameters(), lr=0.1)
            ipex_model, ipex_optimizer = ipex.optimize(
                ori_model, torch.bfloat16, ori_optimizer
            )

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
            for name, param in ipex_model.named_parameters():
                # We won't use the grad of the empty weight and bias tensor.
                # These tensors are only used during inference.
                if name in _empty_weight_bias_parameter_names(
                    prefixes=["conv", "linear"]
                ):
                    continue
                self.assertTrue(param.grad is not None)
            uncast_weight = [
                ipex_model.bn.weight.data_ptr(),
                ipex_model.bn.bias.data_ptr(),
            ]
            for param in ipex_optimizer.param_groups[0]["params"]:
                if param.data_ptr() not in uncast_weight:
                    self.assertTrue(param.grad is None)
                    self.assertTrue(
                        ipex_optimizer.params_attr[param].parameter.grad is not None
                    )
                else:
                    self.assertTrue(param.grad is not None)

            with torch.autograd.profiler.profile() as ipex_prof:
                ipex_optimizer.zero_grad(set_to_none=set_to_none)
            # check grad are zeroed or are set to none
            for name, param in ipex_model.named_parameters():
                # We won't use the grad of the empty weight and bias tensor.
                # These tensors are only used during inference.
                if name in _empty_weight_bias_parameter_names(
                    prefixes=["conv", "linear"]
                ):
                    continue
                expected_grad = None if set_to_none else torch.zeros_like(param)
                self.assertEqual(expected_grad, param.grad)

            for param in ipex_optimizer.param_groups[0]["params"]:
                if param.data_ptr() not in uncast_weight:
                    expected_grad = (
                        None if set_to_none else torch.zeros_like(param).bfloat16()
                    )
                    self.assertEqual(
                        expected_grad,
                        ipex_optimizer.params_attr[param].parameter.grad,
                    )
                else:
                    expected_grad = None if set_to_none else torch.zeros_like(param)
                    self.assertEqual(expected_grad, param.grad)

            # check the num of calls for 'zero_grad' are same
            self.assertEqual(
                count_zero_grad(ori_prof.function_events),
                count_zero_grad(ipex_prof.function_events),
            )


if __name__ == "__main__":
    test = unittest.main()
