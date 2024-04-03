import types
import inspect

import torch
import torch._dynamo.trace_rules as trace_rules

from torch._dynamo.utils import is_safe_constant
from torch._dynamo.config import allowed_functions_module_string_ignorelist

from ...utils.utils import has_xpu

# key: id(obj)
# value: obj(Module/Function)
_torch_object_ids = dict()


def _is_allowed_module_prefix(obj):
    allowed_modules = ("torch", "math", "torch.xpu", "intel_extension_for_pytorch.xpu")
    disallowed_modules = (
        "intel_extension_for_pytorch.xpu.optim",
        "intel_extension_for_pytorch.xpu.FP32MathMode",
        "intel_extension_for_pytorch.xpu.utils",
        "intel_extension_for_pytorch._C",
        "intel_extension_for_pytorch._dynamo.xpu.register",
    )
    allowed_modules_dot = tuple([x + "." for x in allowed_modules])
    module = inspect.getmodule(obj)
    if module is None:
        return False

    mod_name = module.__name__

    # kick out the obj which has prefix in disallowed modules
    if any(mod_name.startswith(m) for m in disallowed_modules):
        return False

    return mod_name in allowed_modules or mod_name.startswith(allowed_modules_dot)


# recursively find the all modules/functions in the given module
def _find_torch_objects(module):
    if any(
        module.__name__.startswith(mod_name)
        for mod_name in allowed_functions_module_string_ignorelist
    ):
        return
    _torch_object_ids[id(module)] = module.__name__
    for name, obj in list(module.__dict__.items()):
        if id(obj) not in _torch_object_ids:
            obj_value = f"{module.__name__}.{name}"
            if isinstance(obj, types.ModuleType):
                if obj.__name__.startswith("torch.xpu") and _is_allowed_module_prefix(
                    obj
                ):
                    _torch_object_ids[id(obj)] = obj_value
                    _find_torch_objects(obj)
            elif _is_allowed_module_prefix(obj):
                _torch_object_ids[id(obj)] = obj_value
            elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                _torch_object_ids[id(obj)] = obj_value


# the container `_allowed_container` and `_disallowed_container` uses the python id to identify
# torch module and function. The register policy here is aligned with torch.cuda.
def _register_module_function_to_dynamo(obj):
    assert has_xpu(), "register modules to dynamo should have xpu compiled"

    # disallowed function for dynamo
    remove = [
        torch.xpu.current_device,
        torch.xpu.set_device,
        torch.xpu.__builtins__,
        torch.xpu.optimize,
    ]
    for entity in remove:
        trace_rules._disallowed_callable_ids.add(id(entity))

    # enumerate all modules and save in _torch_object_ids
    _find_torch_objects(torch.xpu)

    # kick out disallowed
    for idx in trace_rules._disallowed_callable_ids():
        if idx in _torch_object_ids:
            del _torch_object_ids[idx]

    # register to the dynamo legal list
    for index, _ in _torch_object_ids.items():
        if index not in trace_rules._allowed_callable_ids():
            trace_rules._allowed_callable_ids.add(index)
