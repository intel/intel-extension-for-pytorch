import torch
import torch.fx as fx
import types


def override_is_leaf_module():
    fx_tracer = fx.Tracer
    orig_is_leaf_module_fn = fx_tracer.is_leaf_module

    def ipex_is_leaf_module_fn(
        self, m: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        is_ipex = m.__module__.startswith("intel_extension_for_pytorch.nn")
        return is_ipex or orig_is_leaf_module_fn(self, m, module_qualified_name)

    fx_tracer.is_leaf_module = types.MethodType(ipex_is_leaf_module_fn, fx_tracer)


override_is_leaf_module()
