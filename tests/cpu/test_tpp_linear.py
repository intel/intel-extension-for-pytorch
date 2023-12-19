import unittest
import itertools
import torch
import intel_extension_for_pytorch as ipex
from torch.testing._internal.common_utils import TestCase
import copy
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)


class Linear_with_bias(torch.nn.Module):
    def __init__(self):
        super(Linear_with_bias, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        return self.mlp(x)


class Linear_without_bias(torch.nn.Module):
    def __init__(self):
        super(Linear_without_bias, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return self.mlp(x)


class Linear_gelu(torch.nn.Module):
    def __init__(self):
        super(Linear_gelu, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        return torch.nn.functional.gelu(self.mlp(x))


class Linear_silu(torch.nn.Module):
    def __init__(self):
        super(Linear_silu, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return torch.nn.functional.silu(self.mlp(x))


class Linear_relu(torch.nn.Module):
    def __init__(self):
        super(Linear_relu, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return torch.nn.functional.relu(self.mlp(x))


class Linear_mul(torch.nn.Module):
    def __init__(self):
        super(Linear_mul, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return self.mlp(x) * x


class Linear_add(torch.nn.Module):
    def __init__(self):
        super(Linear_add, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return self.mlp(x) + x


class Linear_add_add(torch.nn.Module):
    def __init__(self):
        super(Linear_add_add, self).__init__()
        self.mlp = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        return self.mlp(x) + x + x


class Linear_tpp_fallback_dnnl(torch.nn.Module):
    def __init__(self):
        super(Linear_tpp_fallback_dnnl, self).__init__()
        self.mlp = torch.nn.Linear(4097, 4097)

    def forward(self, x):
        return self.mlp(x)


class TestTPPlinear(TestCase):
    def test_tpp_linear_fallback(self):
        x1 = torch.rand(1, 1, 4097)
        x2 = copy.deepcopy(x1)
        for dtype in [torch.float, torch.bfloat16]:
            model = Linear_tpp_fallback_dnnl().eval()

            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if dtype is torch.bfloat16 else False
            ):
                ref_out = model(x1)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if dtype is torch.bfloat16 else False
            ):
                out = model(x2)
            self.assertEqual(out, ref_out)
            _disable_tpp()

    def test_tpp_linear(self):
        x1 = torch.rand(1, 1, 4096)
        x2 = copy.deepcopy(x1)
        for dtype in [torch.float, torch.bfloat16]:
            model = Linear_with_bias().eval()
            model_nb = Linear_without_bias().eval()
            if dtype is torch.bfloat16:
                x1 = x1.to(torch.bfloat16)
                x2 = x2.to(torch.bfloat16)
                model = model.to(torch.bfloat16)
                model_nb = model_nb.to(torch.bfloat16)
            ref_out = model(x1)
            ref_out_nb = model_nb(x1)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)
            model_nb = ipex.optimize(model_nb, dtype=dtype)
            out = model(x2)
            out_nb = model_nb(x2)
            self.assertEqual(out, ref_out)
            self.assertEqual(out_nb, ref_out_nb)
            _disable_tpp()

    def test_tpp_linear_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [Linear_with_bias, Linear_without_bias],
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            Model,
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Model().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            with torch.no_grad():
                ref_out = model(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_model(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_gelu(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        with torch.no_grad():
            for dtype in [torch.bfloat16]:
                model = Linear_gelu().eval()
                if dtype is torch.bfloat16:
                    x1 = x1.to(torch.bfloat16)
                    x2 = x2.to(torch.bfloat16)
                    model = model.to(torch.bfloat16)
                ref_out = model(x1)

                _enable_tpp()
                model = ipex.optimize(model, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_linear_gelu(
                    x2, model.mlp.weight, model.mlp.bias
                )
                self.assertEqual(out, ref_out)
                _disable_tpp()

    def test_tpp_linear_gelu_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_gelu().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_gelu(
                    x, model.mlp.weight, model.mlp.bias, model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_silu(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        with torch.no_grad():
            for dtype in [torch.bfloat16]:
                model = Linear_silu().eval()
                if dtype is torch.bfloat16:
                    x1 = x1.to(torch.bfloat16)
                    x2 = x2.to(torch.bfloat16)
                    model = model.to(torch.bfloat16)
                ref_out = model(x1)

                _enable_tpp()
                model = ipex.optimize(model, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_linear_silu(
                    x2, model.mlp.weight, x2.new_empty(0)
                )
                self.assertEqual(out, ref_out)
                _disable_tpp()

    def test_tpp_linear_silu_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_silu().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_silu(
                    x, model.mlp.weight, x.new_empty(0), model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_relu(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        with torch.no_grad():
            for dtype in [torch.bfloat16]:
                model = Linear_relu().eval()
                if dtype is torch.bfloat16:
                    x1 = x1.to(torch.bfloat16)
                    x2 = x2.to(torch.bfloat16)
                    model = model.to(torch.bfloat16)
                ref_out = model(x1)

                _enable_tpp()
                model = ipex.optimize(model, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_linear_relu(
                    x2, model.mlp.weight, x2.new_empty(0)
                )
                self.assertEqual(out, ref_out)
                _disable_tpp()

    def test_tpp_linear_relu_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_relu().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_relu(
                    x, model.mlp.weight, x.new_empty(0), model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_mul(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        with torch.no_grad():
            for dtype in [torch.bfloat16]:
                model = Linear_mul().eval()
                if dtype is torch.bfloat16:
                    x1 = x1.to(torch.bfloat16)
                    x2 = x2.to(torch.bfloat16)
                    model = model.to(torch.bfloat16)
                ref_out = model(x1)

                _enable_tpp()
                model = ipex.optimize(model, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_linear_mul(
                    x2, x2, model.mlp.weight, x2.new_empty(0)
                )
                self.assertEqual(out, ref_out)
                _disable_tpp()

    def test_tpp_linear_mul_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_mul().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_mul(
                    x, x, model.mlp.weight, x.new_empty(0), model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_add(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        with torch.no_grad():
            for dtype in [torch.bfloat16]:
                model = Linear_add().eval()
                if dtype is torch.bfloat16:
                    x1 = x1.to(torch.bfloat16)
                    x2 = x2.to(torch.bfloat16)
                    model = model.to(torch.bfloat16)
                ref_out = model(x1)

                _enable_tpp()
                model = ipex.optimize(model, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_linear_add(
                    x2, x2, model.mlp.weight, x2.new_empty(0), 1.0
                )
                self.assertEqual(out, ref_out)
                _disable_tpp()

    def test_tpp_linear_add_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_add().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_add(
                    x, x, model.mlp.weight, x.new_empty(0), 1.0, model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_add2(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        with torch.no_grad():
            for dtype in [torch.bfloat16]:
                model = Linear_add_add().eval()
                if dtype is torch.bfloat16:
                    x1 = x1.to(torch.bfloat16)
                    x2 = x2.to(torch.bfloat16)
                    model = model.to(torch.bfloat16)
                ref_out = model(x1)

                _enable_tpp()
                model = ipex.optimize(model, dtype=dtype)
                out = torch.ops.torch_ipex.tpp_linear_add_add(
                    x2, x2, x2, model.mlp.weight, model.mlp.bias, 1.0
                )
                self.assertEqual(out, ref_out)
                _disable_tpp()

    def test_tpp_linear_add2_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_add_add().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_add_add(
                    x,
                    x,
                    x,
                    model.mlp.weight,
                    model.mlp.bias,
                    1.0,
                    model.mlp.out_features,
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()


if __name__ == "__main__":
    test = unittest.main()
