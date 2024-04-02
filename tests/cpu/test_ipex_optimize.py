import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    _IPEXLinear as _IPEXLinear,
    _IPEXConv2d as _IPEXConv2d,
)
from torch.testing._internal.common_utils import TestCase
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    Adamax,
    ASGD,
    RMSprop,
    Rprop,
    SGD,
)
import unittest
import itertools
import copy
from common_utils import TestModule, _empty_weight_bias_parameter_names
from intel_extension_for_pytorch.optim._lamb import Lamb
import os

try:
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
skipIfNoTransformers = unittest.skipIf(not HAS_TRANSFORMERS, "no transformers")

curpath = os.path.abspath(os.path.dirname(__file__))


class ConvBatchNorm(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ConvBatchNorm, self).__init__()
        self.input1 = torch.randn(1, 3, 224, 224)
        self.conv = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        self.bn = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

    def forward(self, x):
        return self.bn(self.conv(x))


class TwoLayerMLP(torch.nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.input1 = torch.randn(2, 2)
        self.input2 = torch.randn(3, 3)
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(3, 3)

    def forward(self, x1, x2):
        return self.l1(x1).sum() + self.l2(x2).sum()


class OneLayerMLP(torch.nn.Module):
    def __init__(self):
        super(OneLayerMLP, self).__init__()
        self.input1 = torch.randn(2, 2)
        self.l1 = torch.nn.Linear(2, 2)

    def forward(self, x1):
        return self.l1(x1)


class ConvTranspose2d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ConvTranspose2d, self).__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 5, (3, 3))
        self.input1 = torch.randn(5, 5, 3, 3)

    def forward(self, x):
        x = self.conv_transpose2d(x)
        return x


class LinearBatchNormNd(torch.nn.Module):
    def __init__(self, dim):
        super(LinearBatchNormNd, self).__init__()
        self.linear = torch.nn.Linear(32, 32)
        if dim == 1:
            self.input1 = torch.randn(1, 32)
            self.bn = torch.nn.BatchNorm1d(32)
        elif dim == 2:
            self.input1 = torch.randn(1, 32, 32, 32)
            self.bn = torch.nn.BatchNorm2d(32)
        elif dim == 3:
            self.input1 = torch.randn(1, 32, 32, 32, 32)
            self.bn = torch.nn.BatchNorm3d(32)

    def forward(self, x):
        return self.bn(self.linear(x))


class ConvBatchNormLinearBatchNorm(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ConvBatchNormLinearBatchNorm, self).__init__()
        self.input1 = torch.randn(1, 32, 32, 32)
        self.conv = torch.nn.Conv2d(32, 32, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.linear = torch.nn.Linear(32, 32)
        self.bn2 = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        return self.bn2(self.linear(self.bn1(self.conv(x))))


class TestOptimizeCases(TestCase):
    def test_optimize_conv_bn_parameters_behavior(self):
        model = ConvBatchNorm().eval()
        pre_te_enable_status = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(False)
        for level in ["O0", "O1"]:
            for conv_bn_folding in [True, False]:
                opt_M = ipex.optimize(
                    model,
                    level=level,
                    dtype=torch.float,
                    conv_bn_folding=conv_bn_folding,
                )
                with torch.no_grad():
                    x = model.input1
                    traced_model = torch.jit.trace(opt_M, x)
                    trace_graph = traced_model.graph_for(x)
                self.assertEqual(
                    any(n.kind() == "ipex::batch_norm" for n in trace_graph.nodes()),
                    not (conv_bn_folding),
                )
            # TODO check weight_prepack.
        torch._C._jit_set_texpr_fuser_enabled(pre_te_enable_status)

    def test_optimize_linear_bn_parameters_behavior(self):
        for dim in [1, 2, 3]:
            model = LinearBatchNormNd(dim=dim).eval()
            for level in ["O0", "O1"]:
                for linear_bn_folding in [True, False]:
                    opt_M = ipex.optimize(
                        model,
                        level=level,
                        dtype=torch.float,
                        linear_bn_folding=linear_bn_folding,
                    )
                    with torch.no_grad():
                        x = model.input1
                        traced_model = torch.jit.trace(opt_M, x)
                        trace_graph = traced_model.graph_for(x)
                    self.assertEqual(
                        any(
                            n.kind() == "ipex::batch_norm" for n in trace_graph.nodes()
                        ),
                        not (linear_bn_folding),
                    )

    def test_optimize_conv_bn_linear_bn_parameters_behavior(self):
        model = ConvBatchNormLinearBatchNorm().eval()
        max_num_folding = 2
        for level in ["O0", "O1"]:
            for conv_bn_folding in [True, False]:
                for linear_bn_folding in [True, False]:
                    opt_M = ipex.optimize(
                        model,
                        level=level,
                        dtype=torch.float,
                        conv_bn_folding=conv_bn_folding,
                        linear_bn_folding=linear_bn_folding,
                    )
                    with torch.no_grad():
                        x = model.input1
                        traced_model = torch.jit.trace(opt_M, x)
                        trace_graph = traced_model.graph_for(x)
                    self.assertEqual(
                        len(
                            [
                                n
                                for n in trace_graph.nodes()
                                if n.kind() == "ipex::batch_norm"
                            ]
                        ),
                        max_num_folding - (conv_bn_folding + linear_bn_folding),
                    )

    def test_optimize_bf16_model(self):
        model = ConvBatchNorm()
        optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)
        # model should not has master weight attr for infernence model.
        self.assertTrue(not hasattr(optimized_model.conv, "master_weight"))
        # model should has master weight attr for infernence model.
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        optimized_model, optimized_sgd = ipex.optimize(
            model.train(),
            optimizer=sgd,
            dtype=torch.bfloat16,
            split_master_weight_for_bf16=False,
        )
        self.assertEqual(optimized_model.conv.weight.dtype, torch.bfloat16)

        def found_wrapper(parameter, params_attr):
            for _, v in params_attr.items():
                if parameter is v.parameter:
                    return v
            return None

        wrapper = found_wrapper(optimized_model.conv.weight, optimized_sgd.params_attr)
        self.assertTrue(wrapper is not None)
        self.assertEqual(wrapper.master_parameter.dtype, torch.float)

    @skipIfNoTransformers
    def test_optimize_bf16_AlbertMLMHead(self):
        from transformers.models import albert
        from intel_extension_for_pytorch.nn.utils import _parameter_wrapper

        config = transformers.AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/albert-base-v1"
        )
        model = albert.modeling_albert.AlbertForMaskedLM(config)
        params_attr = {}
        _parameter_wrapper.get_shared_parameter_status(model, params_attr)
        for name, param in model.named_parameters():
            if name == "albert.embeddings.word_embeddings.weight":
                self.assertTrue(
                    albert.modeling_albert.AlbertMLMHead
                    in params_attr[param].modules_cls
                )
                self.assertEqual(param.dtype, torch.float32)
                self.assertTrue(params_attr[param].can_cast_inference(torch.bfloat16))
                params_attr[param].cast_for_inference(torch.bfloat16)
                self.assertEqual(param.dtype, torch.bfloat16)
                break

    def test_optimize_pretrain_model(self):
        optimizer_options = [
            Lamb,
            Adadelta,
            Adagrad,
            Adam,
            AdamW,
            Adamax,
            ASGD,
            # RMSprop, # TODO: accuracy fails on SPR starting from oneDNN commit 0f354d
            Rprop,
            SGD,
        ]

        options = itertools.product([torch.float, torch.bfloat16], optimizer_options)
        for dtype, optimizer in options:
            model = ConvBatchNorm().to(memory_format=torch.channels_last).train()
            model.conv.weight.requires_grad_(False)
            model.conv.bias.requires_grad_(False)
            origin_model = copy.deepcopy(model)
            lr = 1e-4 if optimizer is SGD else 1e-2
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            ipex_model, ipex_optimizer = ipex.optimize(
                origin_model, optimizer=origin_optimizer, dtype=dtype
            )
            self.assertEqual(
                origin_model.conv.weight.requires_grad,
                ipex_model.conv.weight.requires_grad,
            )
            self.assertEqual(
                origin_model.conv.bias.requires_grad, ipex_model.conv.bias.requires_grad
            )
            self.assertEqual(
                origin_model.bn.weight.requires_grad, ipex_model.bn.weight.requires_grad
            )
            self.assertEqual(
                origin_model.bn.bias.requires_grad, ipex_model.bn.bias.requires_grad
            )

            x = model.input1.to(memory_format=torch.channels_last)
            origin_x = x.clone()
            ipex_x = x.clone()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y1 = origin_model(origin_x)
                grad_y = torch.ones_like(y1)
                origin_optimizer.zero_grad()
                y1.backward(grad_y)
                origin_optimizer.step()
                # train one step for ipex.
                y2 = ipex_model(ipex_x)
                ipex_optimizer.zero_grad()
                y2.backward(grad_y)
                ipex_optimizer.step()
                self.assertEqual(y1, y2, rtol=1e-4, atol=5e-02)
                origin_model_state = origin_model.state_dict()
                ipex_model_state = ipex_model.state_dict()
                for var_name in origin_model_state:
                    self.assertEqual(
                        origin_model_state[var_name],
                        ipex_model_state[var_name],
                        rtol=1e-4,
                        atol=5e-02,
                    )
                self.assertTrue(origin_model.conv.weight.grad is None)
                self.assertTrue(ipex_model.conv.weight.grad is None)

    def test_optimize_unsupport_dtype_conversion(self):
        class Conv(torch.nn.Module):
            def __init__(
                self,
            ):
                super(Conv, self).__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )

            def forward(self, x):
                return self.conv(x)

        model = Conv().double()
        with self.assertLogs("IPEX", level="WARNING") as cm:
            optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)
        expected_msg = [
            "WARNING:IPEX:[NotSupported]Can't convert model's parameters dtype from torch.float64 to torch.bfloat16"
        ]
        self.assertEqual(cm.output, expected_msg)

    def test_optimize_bf16_upsupported(self):
        class Conv(torch.nn.Module):
            def __init__(
                self,
            ):
                super(Conv, self).__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )

        def forward(self, x):
            return self.conv(x)

        model = Conv()
        if not core.onednn_has_bf16_support():
            msg = r"BF16 weight prepack needs the cpu support avx512bw, avx512vl and avx512dq, \
                please set dtype to torch.float or set weights_prepack to False."
            with self.assertRaisesRegex(AssertionError, msg):
                optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)

    def test_optimize_unsupport_freeze_optimization(self):
        model = ConvBatchNorm().eval()
        x = model.input1
        with torch.no_grad():
            traced_model = torch.jit.trace(model, x)
            frozen_model = torch.jit.freeze(traced_model)
        optimized_model = ipex.optimize(frozen_model)
        self.assertTrue(frozen_model == optimized_model)

    def test_optimize_inplace_behavior_eval_mode(self):
        M_ori = TestModule()
        options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
        for dtype, level in options:
            # non-inplace
            M = copy.deepcopy(M_ori).eval()
            opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=False)
            self.assertTrue(
                M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr()
            )
            self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
            self.assertTrue(
                M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr()
            )

            # inplace
            M = copy.deepcopy(M_ori).eval()
            opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=True)
            # After ConvBN folding,  opt_M will be Graph Module while the M is original nn.Module which they
            # share parameters. But the changes on Graph Module cannot be reflected on original module. So
            # only the un-opitimized weight will use same mem buffer with original module.
            if level == "O1":
                self.assertTrue(
                    M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr()
                )  # linear is optimized and used same parameter with original model
                self.assertTrue(M.linear.weight is opt_M.linear.weight)
                self.assertTrue(isinstance(opt_M.linear, _IPEXLinear))
            # un-optimized part should be inplaced
            self.assertTrue(
                M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr()
            )

    def test_optimize_inplace_behavior_training_mode_with_optimizer(self):
        M_ori = TestModule()
        options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
        for dtype, level in options:
            # non-inplace
            M = copy.deepcopy(M_ori).train()
            sgd = torch.optim.SGD(M.parameters(), lr=0.1)
            opt_M, _ = ipex.optimize(
                M, dtype=dtype, optimizer=sgd, level=level, inplace=False
            )
            self.assertTrue(
                M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr()
            )
            self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
            self.assertTrue(
                M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr()
            )
            if level == "O1":
                self.assertEqual(M.linear.weight.dtype, torch.float)
                self.assertEqual(M.conv.weight.dtype, torch.float)
                self.assertEqual(M.embeddingbag.weight.dtype, torch.float)
                self.assertEqual(M.bn.weight.dtype, torch.float)
                self.assertEqual(opt_M.linear.weight.dtype, dtype)
                self.assertEqual(opt_M.conv.weight.dtype, dtype)
                self.assertEqual(opt_M.embeddingbag.weight.dtype, dtype)
                self.assertEqual(opt_M.bn.weight.dtype, torch.float)

            # inplace
            M = copy.deepcopy(M_ori).train()
            sgd = torch.optim.SGD(M.parameters(), lr=0.1)
            opt_M, _ = ipex.optimize(
                M, dtype=dtype, optimizer=sgd, level=level, inplace=True
            )
            self.assertTrue(
                M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr()
            )
            self.assertTrue(M.conv.weight.data_ptr() == opt_M.conv.weight.data_ptr())
            self.assertTrue(
                M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr()
            )
            if level == "O1":
                self.assertEqual(M.linear.weight.dtype, dtype)
                self.assertEqual(M.conv.weight.dtype, dtype)
                self.assertEqual(M.embeddingbag.weight.dtype, dtype)
                self.assertEqual(M.bn.weight.dtype, torch.float)

    def _test_tensor_convert(self, tensor, bf16_tensor):
        top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(tensor)
        # truncated top half should equal with convert fp32 to bf16 by ".bfloat()"
        self.assertEqual(bf16_tensor, top_half)
        # recovery float tensor with top half and bottom half
        float_tensor = torch.ops.torch_ipex.cat_bfloat16_float(top_half, bot_half)
        self.assertEqual(tensor, float_tensor)
        self.assertEqual(tensor.stride(), top_half.stride())
        self.assertEqual(tensor.stride(), float_tensor.stride())

    def test_tensor_convert(self):
        # contiguous case
        tensor = torch.rand(100, 100)
        self._test_tensor_convert(tensor, tensor.bfloat16())
        # transposed case
        self._test_tensor_convert(tensor.t(), tensor.bfloat16().t())
        # sliced-out case
        self._test_tensor_convert(tensor[2:5, 2:5], tensor.bfloat16()[2:5, 2:5])
        # nc11 channel-last case
        tensor = torch.rand(128, 256, 1, 1).to(memory_format=torch.channels_last)
        self._test_tensor_convert(tensor, tensor.bfloat16())

    def test_module_conversion(self):
        M_ori = TestModule()
        options = itertools.product(
            [torch.bfloat16, torch.float32], ["O0", "O1"], [True, False]
        )
        for dtype, level, auto_kernel_selection in options:
            sgd = torch.optim.SGD(M_ori.parameters(), lr=0.1)
            opt_M, _ = ipex.optimize(
                M_ori,
                dtype=dtype,
                optimizer=sgd,
                level=level,
                auto_kernel_selection=auto_kernel_selection,
            )
            if level == "O0":
                self.assertTrue(isinstance(opt_M.linear, torch.nn.Linear))
                self.assertTrue(isinstance(opt_M.conv, torch.nn.Conv2d))
            else:
                if not auto_kernel_selection and dtype == torch.float32:
                    self.assertTrue(isinstance(opt_M.linear, torch.nn.Linear))
                else:
                    self.assertTrue(isinstance(opt_M.linear, _IPEXLinear))
                self.assertTrue(isinstance(opt_M.conv, _IPEXConv2d))

    def test_record_shape(self):
        options = itertools.product([OneLayerMLP, TwoLayerMLP], [True, False])
        for module, inference_only in options:
            M = module()
            input = M.input1
            if isinstance(M, TwoLayerMLP):
                input = (M.input1, M.input2)
            if inference_only:
                M.eval()
                opt_M = ipex.optimize(M, sample_input=input, auto_kernel_selection=True)
            else:
                optimizer = torch.optim.SGD(M.parameters(), lr=0.01)
                opt_M, _ = ipex.optimize(
                    M,
                    optimizer=optimizer,
                    sample_input=input,
                    auto_kernel_selection=True,
                )
            self.assertEqual(opt_M.l1.batch_size_collapsed, 2)
            if isinstance(M, TwoLayerMLP):
                self.assertEqual(opt_M.l2.batch_size_collapsed, 3)

    def test_traced_model_serialization(self):
        for module in [ConvBatchNorm, OneLayerMLP, ConvTranspose2d]:
            for dtype in [torch.float, torch.bfloat16]:
                M = module().eval()
                input = M.input1.to(dtype)
                opt_M = ipex.optimize(M, dtype=dtype, auto_kernel_selection=True)
                with torch.no_grad():
                    traced_M = torch.jit.trace(opt_M, input).eval()
                    traced_M.save("traced_m.pt")
                    loaded_M = torch.jit.load("traced_m.pt")
                    self.assertEqual(traced_M(input), loaded_M(input))
                    os.remove("traced_m.pt")

    def test_optimized_model_with_fx(self):
        for module in [ConvBatchNorm, OneLayerMLP, ConvTranspose2d]:
            for dtype in [torch.float, torch.bfloat16]:
                M = module().eval()
                input = M.input1.to(dtype)
                opt_M = ipex.optimize(M, dtype=dtype, auto_kernel_selection=True)
                ref_out = opt_M(input)
                fx_M = optimization.fuse(opt_M)
                fx_out = fx_M(input)
                self.assertEqual(ref_out, fx_out)
                with torch.no_grad():
                    traced_M = torch.jit.trace(fx_M, input).eval()
                    traced_M = torch.jit.freeze(traced_M)
                    # do graph opt
                    traced_M(input)
                    # get optimized results
                    out = traced_M(input)
                    self.assertEqual(ref_out, out)

    def test_optimized_model_with_sample_input(self):
        for module in [ConvBatchNorm, OneLayerMLP, ConvTranspose2d]:
            model = module().train()
            input = model.input1
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            origin_model_state = copy.deepcopy(model.state_dict())
            ipex_model, _ = ipex.optimize(
                model,
                dtype=torch.float32,
                inplace=False,
                optimizer=optimizer,
                sample_input=input,
            )
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(
                    origin_model_state[var_name], ipex_model_state[var_name]
                )

    def test_partial_model_update(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.L1 = torch.nn.Linear(10, 10)
                self.L2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                return (self.L1(x), self.L2(x))

        model = M()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        model.train()
        model, optimizer = ipex.optimize(
            model, optimizer=optimizer, dtype=torch.bfloat16
        )

        with torch.cpu.amp.autocast():
            loss = model(torch.rand(10, 10))[0].sum()

        loss.backward()
        optimizer.step()

    def _test_load_after_ipex_optimize_inference(
        self, model_class, dtype, optimizer_class, level, inplace
    ):
        model = model_class().train()
        input = model.input
        if optimizer_class == SGD:
            optimizer = optimizer_class(model.parameters(), lr=10.01, momentum=0.1)
        else:
            optimizer = optimizer_class(model.parameters(), lr=10.01)
        ipex_model, ipex_optimizer = ipex.optimize(
            model,
            dtype=dtype,
            optimizer=optimizer,
            sample_input=input,
            level=level,
            inplace=inplace,
        )
        # train 2 iters to save something in optimizer's state
        for _ in range(2):
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y = ipex_model(*input).sum()
            ipex_optimizer.zero_grad()
            y.backward()
            ipex_optimizer.step()

        inf_model = model_class().eval()
        inf_model_state = inf_model.state_dict()
        ipex_inf_model = ipex.optimize(
            inf_model, dtype=dtype, sample_input=input, level=level, inplace=inplace
        )
        # check parameters are not same before load
        ipex_model_state = ipex_model.state_dict()
        for var_name in ipex_model_state:
            self.assertNotEqual(ipex_model_state[var_name], inf_model_state[var_name])
        for p1 in ipex_model.named_parameters():
            prefix, attr = p1[0].split(".")
            sub_m = getattr(ipex_inf_model, prefix)
            param = getattr(sub_m, attr)
            # the empty weight and bias tensor will always be Tensor()
            assert_fn = (
                self.assertEqual
                if p1[0]
                in _empty_weight_bias_parameter_names(
                    prefixes=["conv", "linear", "conv_transpose2d"]
                )
                else self.assertNotEqual
            )
            assert_fn(p1[1], param)

        # check parameters are same after load
        ipex_inf_model.load_state_dict(ipex_model_state)
        inf_model_state = ipex_inf_model.state_dict()
        for var_name in ipex_model_state:
            self.assertEqual(
                ipex_model_state[var_name].to(dtype).float(), inf_model_state[var_name]
            )
        for p1 in ipex_model.named_parameters():
            if p1[0] == "linear.weight":
                # Do not compare linear.weight with block format since
                # linear.weight in ipex_model(training model) is plain
                continue
            prefix, attr = p1[0].split(".")
            sub_m = getattr(ipex_inf_model, prefix)
            param = getattr(sub_m, attr)
            self.assertEqual(p1[1], param)

    def _test_load_after_ipex_optimize_training(
        self, model_class, dtype, optimizer_class, level, inplace
    ):
        model = model_class().train()
        input = model.input
        if optimizer_class == SGD:
            optimizer = optimizer_class(model.parameters(), lr=10.01, momentum=0.1)
        else:
            optimizer = optimizer_class(model.parameters(), lr=10.01)
        ipex_model, ipex_optimizer = ipex.optimize(
            model,
            dtype=dtype,
            optimizer=optimizer,
            sample_input=input,
            level=level,
            inplace=inplace,
        )
        # train 2 iters to save something in optimizer's state
        for _ in range(2):
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y = ipex_model(*input).sum()
            ipex_optimizer.zero_grad()
            y.backward()
            ipex_optimizer.step()
        ref_ipex_model = copy.deepcopy(ipex_model)
        ref_ipex_optimizer = copy.deepcopy(ipex_optimizer)
        ref_ipex_model_state = copy.deepcopy(ipex_model.state_dict())
        ref_ipex_optimizer_state = copy.deepcopy(ipex_optimizer.state_dict())

        # train 2 iters to change model/optimizer state
        for _ in range(2):
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y = ipex_model(*input).sum()
            ipex_optimizer.zero_grad()
            y.backward()
            ipex_optimizer.step()
        # check state changed (with public formt)
        ipex_model_state = ipex_model.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in ipex_model_state:
            self.assertNotEqual(
                ipex_model_state[var_name], ref_ipex_model_state[var_name]
            )
        for var_name in ipex_optimizer_state:
            if var_name == "state":
                self.assertNotEqual(
                    ipex_optimizer_state[var_name], ref_ipex_optimizer_state[var_name]
                )
        # check values before load (with block format)
        for p1, p2 in zip(
            ipex_model.named_parameters(), ref_ipex_model.named_parameters()
        ):
            # the empty weight and bias tensor will always be Tensor()
            assert_fn = (
                self.assertEqual
                if p1[0]
                in _empty_weight_bias_parameter_names(
                    prefixes=["conv", "linear", "conv_transpose2d"]
                )
                else self.assertNotEqual
            )
            assert_fn(p1[1], p2[1])
        for (_, v1), (_, v2) in zip(
            ipex_optimizer.state.items(), ref_ipex_optimizer.state.items()
        ):
            self.assertNotEqual(v1, v2)
        ipex_model.load_state_dict(ref_ipex_model_state)
        ipex_optimizer.load_state_dict(ref_ipex_optimizer_state)
        # check values same after load (with block format)
        for p1, p2 in zip(
            ipex_model.named_parameters(), ref_ipex_model.named_parameters()
        ):
            self.assertEqual(p1[1], p2[1])
        for (_, v1), (_, v2) in zip(
            ipex_optimizer.state.items(), ref_ipex_optimizer.state.items()
        ):
            if "step_size" in v1:
                # For Rprop, there is a "clamp" operation on step_size which will change the "zero"
                # attribute for packed position.
                # The zero pos will be changed after "clamp", and will be zero again after pack and
                # repack it. So in ipex_optimizer, the packed pos of "step_size" will be zero but in
                # ref_ipex_optimizer, the packed pos of "step_size" will not be zero. Thus the
                # assertEqual will be failed.
                #    step_sizes=(1e-6, 50)
                #    step_size_min, step_size_max = group['step_sizes']
                #    step_size.mul_(sign).clamp_(step_size_min, step_size_max)
                #    param.addcmul_(grad.sign(), step_size, value=-1)
                #    (param = param - grad.sign() * step_size)
                # but this step_size will not have impact since grad are zero
                v1 = copy.deepcopy(v1)
                v1.pop("step_size")
                v2 = copy.deepcopy(v2)
                v2.pop("step_size")
                self.assertEqual(v1, v2)

        # check state same after load (with plain format)
        ipex_model_state = ipex_model.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in ipex_model_state:
            self.assertEqual(ipex_model_state[var_name], ref_ipex_model_state[var_name])
        for var_name in ipex_optimizer_state:
            self.assertEqual(
                ipex_optimizer_state[var_name], ref_ipex_optimizer_state[var_name]
            )

    # This test case is to simulate the use case of Stable Diffusion fine-tuning
    def test_eval_backward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=(2, 2))

            def forward(self, x):
                return self.conv(x)

        x = torch.randn(1, 3, 8, 8)
        x_optimized = copy.deepcopy(x)
        x.requires_grad_()
        x_optimized.requires_grad_()

        m = Model().eval()
        optimized_m = ipex.optimize(m)

        y = m(x)
        y.sum().backward()

        y_optimized = optimized_m(x_optimized)
        y_optimized.sum().backward()

        grad = x.grad
        grad_optimized = x_optimized.grad

        self.assertEqual(grad, grad_optimized)

    def test_load_after_optimize(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.input = (
                    torch.randn(1, 3, 224, 224),
                    torch.randn(100, 100),
                    torch.randn(5, 5, 3, 3),
                )
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
                )
                self.linear = torch.nn.Linear(100, 100)
                self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 5, (3, 3))

            def forward(self, x1, x2, x3):
                return (
                    self.conv(x1).sum()
                    + self.linear(x2).sum()
                    + self.conv_transpose2d(x3)
                )

        params_dict = {
            "dtype": [torch.float, torch.bfloat16],
            "optimizer": [
                Lamb,
                Adadelta,
                Adagrad,
                Adam,
                AdamW,
                Adamax,
                ASGD,
                RMSprop,
                Rprop,
                SGD,
            ],
            "level": ["O0", "O1"],
            "inplace": [True, False],
        }
        for dtype, optimizer, level, inplace in list(
            itertools.product(*params_dict.values())
        ):
            self._test_load_after_ipex_optimize_training(
                Model, dtype, optimizer, level, inplace
            )
            self._test_load_after_ipex_optimize_inference(
                Model, dtype, optimizer, level, inplace
            )

    def test_reentrancy_of_ipex_optimize(self):
        CALL_NUM = 3

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.input = (
                    torch.randn(1, 3, 224, 224),
                    torch.randn(100, 100),
                    torch.randn(5, 5, 3, 3),
                )
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
                )
                self.linear = torch.nn.Linear(100, 100)
                self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 5, (3, 3))

            def forward(self, x1, x2, x3):
                return (
                    self.conv(x1).sum()
                    + self.linear(x2).sum()
                    + self.conv_transpose2d(x3)
                )

        def run_and_recursively_call_ipex_optimize(
            model_class,
            dtype,
            level,
            inplace,
            weights_prepack,
            split_master_weight_for_bf16,
            fuse_update_step,
            graph_mode,
        ):
            model = model_class().train()
            input = model.input
            optimizer = torch.optim.SGD(model.parameters(), lr=10.01)
            for _ in range(CALL_NUM):
                # recursively calling ipex.optimize CALL_NUM times
                model, optimizer = ipex.optimize(
                    model,
                    dtype=dtype,
                    optimizer=optimizer,
                    level=level,
                    inplace=inplace,
                    weights_prepack=weights_prepack,
                    split_master_weight_for_bf16=split_master_weight_for_bf16,
                    fuse_update_step=fuse_update_step,
                    graph_mode=graph_mode,
                )
                with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                    y = model(*input).sum()
                optimizer.zero_grad()
                y.backward()
                optimizer.step()

        params_dict = {
            "dtype": [torch.float32, torch.bfloat16],
            "level": ["O1"],
            "inplace": [True, False],
            "weights_prepack": [True, False],
            "split_master_weight_for_bf16": [True, False],
            "fuse_update_step": [True, False],
            "graph_mode": [True, False],
        }

        for (
            dtype,
            level,
            inplace,
            weights_prepack,
            split_master_weight_for_bf16,
            fuse_update_step,
            graph_mode,
        ) in list(itertools.product(*params_dict.values())):
            run_and_recursively_call_ipex_optimize(
                Model,
                dtype,
                level,
                inplace,
                weights_prepack,
                split_master_weight_for_bf16,
                fuse_update_step,
                graph_mode,
            )


if __name__ == "__main__":
    test = unittest.main()
