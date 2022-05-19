import os
import copy
import tempfile
import torch
import torch.fx.experimental.optimization as optimization
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
from functools import wraps
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward, \
    get_execution_plan
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, \
    get_function_arglist, load_tests, TemporaryFileName

from torch.jit._recursive import wrap_cpp_module

import intel_extension_for_pytorch as ipex

LLGA_FUSION_GROUP = 'ipex::LlgaFusionGroup'

default_static_qconfig = QConfig(
        activation= MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
        weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

# For fp32 and bf16 LLGA UT only
def llga_fp32_bf16_test_env(func):
    @wraps(func)
    def wrapTheFunction(*args):
        # make sure that the profiling mode is turned on
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)

        ipex._C.set_llga_fp32_bf16_enabled(True)
        func(*args)
        ipex._C.set_llga_fp32_bf16_enabled(False)
    return wrapTheFunction

def all_backward_graphs(module):
    ge_state = module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    executors = fwd_plan.code.grad_executor_states()
    assert len(executors), 'No backward graph found in the module'
    grad_executor = executors[0]
    bwd_plans = list(grad_executor.execution_plans.values())
    return [p.graph.copy() for p in bwd_plans]


def backward_graph(module):
    graphs = all_backward_graphs(module)
    assert len(graphs), 'Warm up the module before calling backward_graph'
    return graphs[0]


def freeze(model):
    return wrap_cpp_module(torch._C._freeze_module(model._c, preserveParameters=True))


# port from pytorch/test/test_jit_fuser_te.py
def findFusionGroups( graph):
    result = []
    for n in graph.nodes():
        if n.kind() == LLGA_FUSION_GROUP:
            result.append(n.g('Subgraph'))
            continue
        for block in n.blocks():
            result += findFusionGroups(block)
    return result


def warmup_forward(f, *args, profiling_count=2):
    for i in range(profiling_count):
        results = f(*args)
    return results


class JitLlgaTestCase(JitTestCase):
    def checkScript(self, m, x):
        requires_grad = any(t.requires_grad for t in x)
        with torch.set_grad_enabled(requires_grad):
            ref = m(*x)
            scripted = torch.jit.script(m)
            y = scripted(*x)
            self.assertEqual(y, ref)
            graph = scripted.graph_for(*x)
        return scripted, graph

    def checkTrace(self, m, x, freeze=True, *args, **kwargs):
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad(), \
                torch._jit_internal._disable_emit_hooks():
            traced = torch.jit.trace(m, x)
            if isinstance(traced, torch.nn.Module) and freeze:
                traced = torch.jit.freeze(traced)
            warmup_forward(traced, *x)
            fwd_graph = traced.graph_for(*x)

            ref_o = m(*x)
            jit_o = traced(*x)
            self.assertEqual(jit_o, ref_o)
        return fwd_graph, traced

    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def checkQuantizeTrace(self, model, x, atol=1e-3, rtol=1e-2, remove_dropout=False, x_var=None, qconfig=default_static_qconfig, int8_bf16=False):
        graph, traced_model, fp32_model = self.prepareModel(model, x, remove_dropout, qconfig, int8_bf16)
        with torch.no_grad():
            y = fp32_model(*x)
            y = y.to(torch.bfloat16) if int8_bf16 else y
            y_llga = traced_model(*x)
            self.assertEqual(y, y_llga, atol=atol, rtol=rtol)

            # test Fallback when input shape changes:
            if x_var:
                y_var = fp32_model(*x_var)
                y_var = y_var.to(torch.bfloat16) if int8_bf16 else y_var
                y_var_llga = traced_model(*x_var)
                self.assertEqual(y_var, y_var_llga, atol=atol, rtol=rtol)

            return graph

    def prepareModel(self, model, x, remove_dropout=False, qconfig=default_static_qconfig, int8_bf16=False, inplace=False):
        model.eval()
        fp32_model = copy.deepcopy(model)
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            # fold conv bn
            if remove_dropout:
                ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
            model = ipex.quantization.prepare(model, qconfig, x, inplace=inplace)
            # do calibration
            y = model(*x)
            # jit trace to insert quant/dequant
            if int8_bf16:
                with torch.cpu.amp.autocast():
                    convert_model = ipex.quantization.convert(model)
                    traced_model = torch.jit.trace(convert_model, x)
            else:
                convert_model = ipex.quantization.convert(model)
                traced_model = torch.jit.trace(convert_model, x)
            traced_model = torch.jit.freeze(traced_model)

            # warm up run
            y0 = traced_model(*x)
            # get the graph at the second run after freezing
            graph = traced_model.graph_for(*x)
            return graph, traced_model, fp32_model

    def checkPatterns(self, graph, patterns):
        fusion_groups = findFusionGroups(graph)
        assert len(fusion_groups) == len(patterns), "length of subgraphs not equal to length of given patterns"

        for i in range(len(fusion_groups)):
            for pattern in patterns[i]:
                self.assertGraphContains(fusion_groups[i], pattern)
