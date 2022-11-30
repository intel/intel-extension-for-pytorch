from functools import partial
from textwrap import dedent
import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, IS_SANDCASTLE, clone_input_helper, first_sample, skipIfSlowGradcheckEnv
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops, OpDTypes
from torch.testing._internal.common_jit import JitCommonTestCase, check_against_reference
from torch.testing._internal.jit_metaprogramming_utils import create_script_fn, create_traced_fn, check_alias_annotation
from torch.testing._internal.jit_utils import disable_autodiff_subgraph_inlining, is_lambda
torch.set_default_dtype(torch.float32)
_variant_ops = partial(ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat))
from common.pytorch_test_base import TestCase, dtypesIfXPU, TEST_XPU, TEST_MULTIGPU, largeTensorTest
from common.common_jit import JitCommonTestCase, check_against_reference
from common.jit_utils import disable_autodiff_subgraph_inlining, is_lambda

@skipIfSlowGradcheckEnv
class TestJit(JitCommonTestCase):
    exact_dtype = True

    @_variant_ops(op_db)
    def test_variant_consistency_jit(self, device, dtype, op):
        _requires_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad, include_conjugated_inputs=include_conjugated_inputs)
        func = op.get_op()
        method = op.get_method()
        variants = {'function': func, 'method': method}
        if isinstance(func, torch._ops.OpOverload):
            self.skipTest("variant consistency doesn't work on torch.ops")
        has_fake_function = op.name in ['resize_', 'resize_as_']
        if has_fake_function:
            variants = {'method': getattr(torch.Tensor, op.name)}
            samples = op.sample_inputs(device, dtype, requires_grad=False)
        tested = False
        for sample in samples:
            for (func_type, variant) in variants.items():
                if variant is None:
                    continue
                if is_lambda(variant):
                    continue
                tested = True
                try:
                    self.indiv_variant_test_jit(device, dtype, op, sample, func_type, variant, has_fake_function)
                except Exception as e:
                    variant_error_info = dedent(f'\n                        Error testing {op.name} {func_type} variant\n                        with dtype: {dtype}\n                        with inputs {sample}:\n                    ')
                    raise Exception(variant_error_info) from e
        assert tested, 'JIT Test does not execute any logic'

    def indiv_variant_test_jit(self, device, dtype, op, sample, func_type, variant, has_fake_function):
        _requires_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        support_script = op.supports_scripting
        name = op.name + '_' if func_type == 'inplace' else op.name
        with disable_autodiff_subgraph_inlining():
            if support_script:
                script_fn = create_script_fn(self, name, func_type)

            def out_fn(output):
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            def get_sample():
                return clone_input_helper(sample.input) if op.name[-1] == '_' else sample.input
            if support_script:
                check_against_reference(self, script_fn, op.get_op(), out_fn, (get_sample(),) + sample.args, sample.kwargs, no_grad=not _requires_grad, no_gradgrad=not op.supports_gradgrad)
            supports_tracing = op.supports_tracing and (not has_fake_function)
            if op.assert_jit_shape_analysis:
                self.assertTrue(supports_tracing)
            if supports_tracing:
                traced_fn = create_traced_fn(self, variant)
                check_against_reference(self, traced_fn, op.get_op(), out_fn, (get_sample(),) + sample.args, sample.kwargs, no_grad=not _requires_grad, no_gradgrad=not op.supports_gradgrad)
            if dtype == torch.float32:
                if support_script and op.name != 'rsub':
                    check_alias_annotation(name, (get_sample(),) + sample.args, sample.kwargs, func_type=func_type, aten_name=op.aten_name)
                checked_shape_analysis = False
                if supports_tracing:
                    out = variant(get_sample(), *sample.args, **sample.kwargs)
                    tuple_of_tensors = isinstance(out, tuple) and all([isinstance(elem, torch.Tensor) for elem in out])
                    if isinstance(out, torch.Tensor) or tuple_of_tensors:
                        if tuple_of_tensors:
                            sizes = [elem.size() for elem in out]
                        else:
                            sizes = out.size()
                        self.checkShapeAnalysis(sizes, traced_fn.graph, op.assert_jit_shape_analysis)
                        checked_shape_analysis = True
                if op.assert_jit_shape_analysis:
                    self.assertTrue(checked_shape_analysis)
            if dtype is torch.float32:
                if IS_SANDCASTLE:
                    nonfusible_nodes = op.autodiff_nonfusible_nodes + op.autodiff_fusible_nodes
                    fusible_nodes = []
                else:
                    nonfusible_nodes = op.autodiff_nonfusible_nodes
                    fusible_nodes = op.autodiff_fusible_nodes
                if supports_tracing:
                    self.assertAutodiffNode(traced_fn.last_graph, op.assert_autodiffed, nonfusible_nodes, fusible_nodes)
                if support_script:
                    self.assertAutodiffNode(script_fn.last_graph, op.assert_autodiffed, nonfusible_nodes, fusible_nodes)
    _alias_ops = partial(ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float,))

    @_alias_ops((op for op in op_db if op.aliases))
    def test_jit_alias_remapping(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        sample = first_sample(self, samples)
        args = ['t0']

        def quote_strs(v):
            if isinstance(v, str):
                return f"'{v}'"
            return str(v)
        args_kw = args + [f'{v}' for v in sample.args] + [f'{k}={quote_strs(v)}' for (k, v) in sample.kwargs.items()]
        sample_args_kwargs = ()
        if len(sample.args) > 0:
            sample_args_kwargs += (sample.args,)
        if len(sample.kwargs) > 0:
            sample_args_kwargs += (sample.kwargs,)
        original_name = op.aten_name
        original_name_inplace = original_name + '_'
        expected_dtype = op(sample.input, *sample.args, **sample.kwargs).dtype
        for a_op in op.aliases:
            inplace = a_op.inplace_variant
            method_or_inplace = [a_op.inplace_variant, a_op.method_variant]
            variants = (v for v in (a_op.op, a_op.method_variant, a_op.inplace_variant) if v is not None)
            for variant in variants:
                variant_name = variant.__name__
                op_name = original_name_inplace if variant is inplace else original_name
                if variant in method_or_inplace:
                    fn_template = '\n                        def _fn(t0{c}):\n                            return t0.{alias_name}({args_kw})\n                    '
                    script = fn_template.format(c=', ' if len(args_kw[1:]) > 1 else '', args_kw=', '.join(args_kw[1:]), alias_name=variant_name)
                else:
                    fn_template = '\n                        def _fn({args}):\n                            return variant({args_kw})\n                    '
                    script = fn_template.format(args=', '.join(args), args_kw=', '.join(args_kw))
                script = script.replace('tensor(', 'torch.tensor(')
                scripted = torch.jit.CompilationUnit(script)._fn
                if variant is inplace and (not torch.can_cast(expected_dtype, dtype)):
                    try:
                        inp = clone_input_helper(sample.input)
                        scripted(inp)
                    except Exception as e:
                        continue
                    self.fail("Inplace operation on integer tensor that should be promoted to float didn't fail!")
                inp = clone_input_helper(sample.input)
                scripted(inp)
                inp = clone_input_helper(sample.input)
                graph = scripted.graph_for(inp)
                FileCheck().check(op.aten_name).check_not(variant_name).run(graph)
            for variant in variants:
                variant_name = variant.__name__
                op_name = original_name_inplace if variant is inplace else original_name

                def _fn(*sample_args, **sample_kwargs):
                    return variant(*sample_args, **sample_kwargs)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                traced = torch.jit.trace(_fn, *inp)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                traced(*inp)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                graph = traced.graph_for(*inp)
                FileCheck().check(op_name).check_not(variant_name).run(graph)
instantiate_device_type_tests(TestJit, globals())
if __name__ == '__main__':
    run_tests()