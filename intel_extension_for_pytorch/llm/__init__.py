from .frontend import optimize

try:
    from .utils import (
        _get_relative_imports,
        _gradient_checkpointing_disable,
        _gradient_checkpointing_enable,
        _get_class_from_dynamic_module,
        _get_cached_module_file,
    )
    import transformers

    transformers.dynamic_module_utils.get_relative_imports = _get_relative_imports
    transformers.dynamic_module_utils.get_cached_module_file = _get_cached_module_file
    transformers.dynamic_module_utils.get_class_from_dynamic_module = (
        _get_class_from_dynamic_module
    )
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_disable = (
        _gradient_checkpointing_disable
    )
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = (
        _gradient_checkpointing_enable
    )
except ImportError:
    pass
