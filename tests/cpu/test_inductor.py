import torch
import intel_extension_for_pytorch as ipex
import unittest
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.testing._internal.common_utils import TestCase

# TODO(jgong5): import and pass all inductor tests from stock pytorch


def check_model(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_cuda=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
):
    """Copied and revised from test/inductor/test_torchinductor.py"""

    def compute_grads(args, kwrags, results, grads):
        def gather_leaf_tensors(args, kwargs):
            args, _ = tree_flatten(args)
            kwargs, _ = tree_flatten(kwargs)
            args = args + kwargs
            leaf_tensors = [
                arg
                for arg in args
                if isinstance(arg, torch.Tensor) and arg.requires_grad
            ]
            return leaf_tensors

        flat_results, _ = tree_flatten(results)
        flat_diff_results = [r for r in flat_results if r.requires_grad]
        assert len(flat_diff_results) > 0

        leaf_tensors = gather_leaf_tensors(args, kwrags)
        assert len(leaf_tensors) > 0
        return torch.autograd.grad(
            flat_diff_results,
            leaf_tensors,
            grads,
            allow_unused=True,
            retain_graph=True,
        )

    def clone_preserve_strides(x, device=None):
        if not isinstance(x, torch.Tensor):
            return x
        buffer = torch.as_strided(
            x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
        )
        if not device:
            buffer = buffer.clone()
        else:
            buffer = buffer.to(device, copy=True)
        out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
        return out

    kwargs = kwargs or {}
    torch._dynamo.reset()

    ref_inputs = [clone_preserve_strides(x) for x in example_inputs]
    ref_kwargs = kwargs
    has_lowp_args = False
    original_lowp_dtype = torch.half

    if reference_in_float:
        # check_lowp is ignored here, it's kept just to be able to call `common` with extra arg
        def upcast_fn(x):
            nonlocal has_lowp_args
            if isinstance(x, torch.Tensor) and (
                x.dtype == torch.float16 or x.dtype == torch.bfloat16
            ):
                has_lowp_args = True
                return x.float()
            else:
                return x

        def get_original_lowp_dtype(example_inputs):
            dtypes = [x.dtype for x in example_inputs if isinstance(x, torch.Tensor)]
            dtype_set = set(dtypes)
            return dtype_set.pop() if len(dtype_set) == 1 else torch.half

        ref_inputs = list(map(upcast_fn, example_inputs))
        ref_kwargs = {k: upcast_fn(v) for k, v in kwargs.items()}
        if has_lowp_args:
            original_lowp_dtype = get_original_lowp_dtype(example_inputs)
            if hasattr(model, "to"):
                model = model.to(torch.float)

    torch.manual_seed(0)

    correct = model(*ref_inputs, **ref_kwargs)
    # downcast the model back if needed
    if reference_in_float and has_lowp_args:
        if hasattr(model, "to"):
            model = model.to(original_lowp_dtype)

    torch._inductor.metrics.reset()

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch.compile(run, backend="ipex")

    torch.manual_seed(0)
    actual = run(*example_inputs, **kwargs)
    assert type(actual) == type(correct)

    correct_flat, correct_spec = tree_flatten(correct)
    actual_flat, _ = tree_flatten(actual)
    if reference_in_float:
        correct_flat = tuple(
            y.to(x.dtype)
            if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
            else y
            for x, y in zip(actual_flat, correct_flat)
        )
        correct = tree_unflatten(correct_flat, correct_spec)

    if assert_equal:
        self.assertEqual(
            actual,
            correct,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            exact_dtype=exact_dtype,
        )
        # In case of input mutations, check that inputs are the same
        self.assertEqual(
            ref_inputs,
            example_inputs,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            # our testing sometimes uses higher precision inputs for the reference
            exact_dtype=False,
        )
    else:
        for correct_val, actual_val in zip(correct_flat, actual_flat):
            if isinstance(correct_val, torch.Tensor):
                assert correct_val.device == actual_val.device
                assert correct_val.size() == actual_val.size()
                assert correct_val.stride() == actual_val.stride()
                assert correct_val.layout == actual_val.layout
                if exact_dtype:
                    assert correct_val.dtype == actual_val.dtype

    if check_gradient:
        # generate random unit norm gradients
        grads = [
            torch.rand(r.shape, device=r.device, dtype=r.dtype)
            for r in correct_flat
            if r.requires_grad
        ]
        for g in grads:
            g /= g.norm()

        correct_grad = compute_grads(ref_inputs, ref_kwargs, correct, grads)
        flat_grads, _ = tree_flatten(correct_grad)
        all_none_grads = all(x is None for x in flat_grads)
        if all_none_grads:
            # See Note [Detaching inputs that never need gradients]
            # There are a handful of ops that can return None gradients, into of zero gradients.
            # If all inputs to an AOTAutograd graph are supposed to get None gradients,
            # AOTAutograd will end up forcing all of the outputs of the forward to not require grad.
            # There's no easy fix to this (see the note above), although one option is to
            # force any derivative formulas in core to return tensors of zeros instead of None.
            flat_results, _ = tree_flatten(actual)
            results_that_require_grad = [
                x
                for x in flat_results
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]
            self.assertEqual(len(results_that_require_grad), 0)
        else:
            actual_grad = compute_grads(example_inputs, kwargs, actual, grads)
            self.assertEqual(
                actual_grad,
                correct_grad,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=exact_dtype,
            )

    torch._dynamo.reset()


class TestIpexInductor(TestCase):
    common = check_model

    def setUp(self):
        self.old_backend = ipex._get_compiler_backend()
        ipex._set_compiler_backend("inductor")
        return super().setUp()

    def tearDown(self):
        ipex._set_compiler_backend(self.old_backend)
        return super().tearDown()

    def test_custom_lowering(self):
        """mm lowering overrides"""

        def fn(x: torch.Tensor, y: torch.Tensor):
            return torch.matmul(torch.softmax(x / 10 + 10, -1), y)

        from intel_extension_for_pytorch._inductor.lowering import register_lowering
        from torch._inductor.lowering import aten
        from torch._inductor.ir import TensorBox, Reduction
        from torch._inductor.virtualized import ops, V

        @register_lowering(aten.mm.default)
        def _mm(a: TensorBox, b: TensorBox):
            assert isinstance(a, TensorBox)
            assert isinstance(b, TensorBox)
            a.realize_hint()
            b.realize_hint()

            m, k = a.get_size()
            _k, n = b.get_size()
            assert k == _k
            reduced_sizes = [k]
            new_size = [m, n]

            m = V.graph.sizevars.evaluate_static_shape(m)
            n = V.graph.sizevars.evaluate_static_shape(n)
            k = V.graph.sizevars.evaluate_static_shape(k)

            _a_loader = a.make_loader()
            _b_loader = b.make_loader()

            def a_loader(idx, reduction_idx):
                m, _ = idx
                (k,) = reduction_idx
                return _a_loader([m, k])

            def b_loader(idx, reduction_idx):
                _, n = idx
                (k,) = reduction_idx
                return _b_loader([k, n])

            def fn(idx, reduction_idx):
                return ops.mul(
                    a_loader(idx, reduction_idx), b_loader(idx, reduction_idx)
                )

            result = Reduction.create(
                device=a.get_device(),
                dst_dtype=a.get_dtype(),
                src_dtype=a.get_dtype(),
                inner_fn=fn,
                ranges=new_size,
                reduction_ranges=reduced_sizes,
                reduction_type="sum",
            )

            if isinstance(
                result.data.data, Reduction
            ):  # Only realize if reduction isn't unrolled
                result.realize()

            return result

        x = torch.randn(64, 128)
        y = torch.randn(128, 256).as_strided([128, 256], [1, 128])
        self.common(fn, (x, y))


if __name__ == "__main__":
    test = unittest.main()
