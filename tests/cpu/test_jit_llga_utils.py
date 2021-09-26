import os
import copy
import tempfile
import torch
import torch.fx.experimental.optimization as optimization

from functools import wraps
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward, \
    get_execution_plan
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, \
    get_function_arglist, load_tests, repeat_test_for_types, TemporaryFileName

from torch.jit._recursive import wrap_cpp_module

import intel_extension_for_pytorch as ipex

LLGA_FUSION_GROUP = 'ipex::LlgaFusionGroup'

# For LLGA UT, disable the PyTorch profiling executor and the IPEX JIT opt
def llga_test_env(func):
    @wraps(func)
    def wrapTheFunction(*args):
        # make sure that the profiling mode is turned on
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)

        ipex.core._jit_set_llga_enabled(True)
        ipex.core.disable_jit_opt()
        func(*args)
        ipex.core.enable_jit_opt()
        ipex.core._jit_set_llga_enabled(False)
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
        m.eval()
        with torch.no_grad(), \
                torch._jit_internal._disable_emit_hooks():
            traced = torch.jit.trace(m, x)
            if freeze:
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

    def checkQuantizeTrace(self, model, x, atol=1e-3, rtol=1e-2, folding=False, remove_dropout=False, config_name="", x_var=None, qscheme=torch.per_tensor_affine, int8_bf16=False):
        graph, traced_model, fp32_model = self.prepareModel(model, x, folding, remove_dropout, config_name, qscheme, int8_bf16)
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

    def prepareModel(self, model, x, folding=False, remove_dropout=False, config_name="", qscheme=torch.per_tensor_affine, int8_bf16=False):
        model.eval()
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            conf = ipex.QuantConf(qscheme=qscheme)
            # fold conv bn
            if folding:
                model = optimization.fuse(model)

            if remove_dropout:
                ipex.utils._replace_dropout_with_identity(model)

            # do calibration
            with ipex.quantization.calibrate(conf):
                y = model(*x)

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'configure_%s.json' % config_name)

                # TODO: remove the serialization and test it in another separate UT once IPEX supported
                # directly using the conf for int8 path
                conf.save(path)
                conf = ipex.QuantConf(path)

                # jit trace to insert quant/dequant
                if int8_bf16:
                    with torch.cpu.amp.autocast():
                        traced_model = ipex.quantization.convert(model, conf, x)
                else:
                    traced_model = ipex.quantization.convert(model, conf, x)

            # warm up run
            y0 = traced_model(*x)

            # get the graph at the second run after freezing
            graph = traced_model.graph_for(*x)

            return graph, traced_model, model

    def checkPatterns(self, graph, patterns):
        fusion_groups = findFusionGroups(graph)
        assert len(fusion_groups) == len(patterns), "length of subgraphs not equal to length of given patterns"

        for i in range(len(fusion_groups)):
            for pattern in patterns[i]:
                self.assertGraphContains(fusion_groups[i], pattern)
