import os
import copy
import tempfile
import torch

from functools import wraps
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward, \
    get_execution_plan
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, \
    get_function_arglist, load_tests, repeat_test_for_types, TemporaryFileName

from torch.jit._recursive import wrap_cpp_module

import intel_pytorch_extension as ipex

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

    def checkTrace(self, m, x, *args, **kwargs):
        grad = any(t.requires_grad for t in x)
        with torch.set_grad_enabled(grad), \
                torch._jit_internal._disable_emit_hooks():
            traced = super().checkTrace(m, x, inputs_require_grads=grad)
            fwd_graph = traced.graph_for(*x)
        if grad:
            warmup_backward(traced(*x).sum())
            return traced, fwd_graph, backward_graph(traced)
        else:
            return traced, fwd_graph

    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def checkQuantizeTrace(self, model, x, atol=1e-3, rtol=1e-2, folding=False, remove_dropout=False, config_name="", x_var=None, qscheme=torch.per_tensor_affine):
        graph, model, fp32_model_with_quant_dequant = self.prepareModel(model, x, folding, remove_dropout, config_name, qscheme)
        with torch.no_grad():
            # calculate after getting the graph
            y_llga = model(*x)

            # disable llga for fp32 path
            ipex.core._jit_set_llga_enabled(False)
            y = fp32_model_with_quant_dequant(*x)
            # test Fallback when input shape changes:
            if x_var:
                y_var = fp32_model_with_quant_dequant(*x_var)
            ipex.core._jit_set_llga_enabled(True)

            self.assertEqual(y, y_llga, atol=atol, rtol=rtol)

            # test Fallback when input shape changes:
            if x_var:
                y_var_llga = model(*x_var)
                self.assertEqual(y_var, y_var_llga, atol=atol, rtol=rtol)

            return graph

    def prepareModel(self, model, x, folding=False, remove_dropout=False, config_name="", qscheme=torch.per_tensor_affine):
        model.eval()
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            # fold conv bn
            if folding:
                model = ipex.fx.conv_bn_fuse(model)

            if remove_dropout:
                ipex.utils._replace_dropout_with_identity(model)

            # do calibration
            conf = ipex.AmpConf(torch.int8, qscheme=qscheme)
            with ipex.amp.calibrate():
                y = model(*x)

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'configure_%s.json' % config_name)

                # TODO: remove the serialization and test it in another separate UT once IPEX supported
                # directly using the conf for int8 path
                conf.save(path)
                conf = ipex.AmpConf(torch.int8, path)

                # jit trace to insert quant/dequant
                with ipex.amp.autocast(enabled=True, configure=conf):
                    model = torch.jit.trace(model, x, check_trace=False)

            fp32_model_with_quant_dequant = copy.deepcopy(model)

            # freeze the module
            model = freeze(model)

            # warm up run
            y0 = model(*x)

            # get the graph at the second run after freezing
            graph = model.graph_for(*x)

            return graph, model, fp32_model_with_quant_dequant

    def checkPatterns(self, graph, patterns):
        fusion_groups = findFusionGroups(graph)
        assert len(fusion_groups) == len(patterns), "length of subgraphs not equal to length of given patterns"

        for i in range(len(fusion_groups)):
            for pattern in patterns[i]:
                self.assertGraphContains(fusion_groups[i], pattern)
