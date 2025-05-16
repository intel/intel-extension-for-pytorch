import re
import functools
import inspect
from functools import partial
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import filecmp
import importlib
import os
import shutil
import typing
from ..utils._logger import logger, WarningType
from typing import Dict, Optional, Union, List
from transformers.dynamic_module_utils import (
    check_imports,
    create_dynamic_module,
    get_class_in_module,
)
import json
from transformers.utils import (
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    cached_file,
    extract_commit_hash,
    is_offline_mode,
    try_to_load_from_cache,
    PaddingStrategy,
    is_tf_tensor,
    is_torch_tensor,
    to_py_obj,
)
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput
from collections.abc import Mapping, Sized
import numpy as np
import torch
import pathlib


def _get_relative_imports(module_file):
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()
    relative_imports = re.findall(
        r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE
    )
    relative_imports += re.findall(
        r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE
    )
    relative_imports = set(relative_imports)
    # For Baichuan2
    if "quantizer" in relative_imports:
        relative_imports.remove("quantizer")
    return list(relative_imports)


def _gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    """
    Activates gradient checkpointing for the current model.

    Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
    activations".

    We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
    the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

    Args:
        gradient_checkpointing_kwargs (dict, *optional*):
            Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
    """
    if not self.supports_gradient_checkpointing:
        raise ValueError(
            f"{self.__class__.__name__} does not support gradient checkpointing."
        )

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    gradient_checkpointing_func = functools.partial(
        checkpoint, **gradient_checkpointing_kwargs
    )

    # For old GC format (transformers < 4.35.0) for models that live on the Hub
    # we will fall back to the overwritten `_set_gradient_checkpointing` methid
    _is_using_old_format = (
        "value" in inspect.signature(self._set_gradient_checkpointing).parameters
    )

    if not _is_using_old_format:
        self._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=gradient_checkpointing_func
        )
    else:
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    if getattr(self, "_hf_peft_config_loaded", False):
        # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
        # we do it also on PEFT:
        # https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        # When training with PEFT, only LoRA layers will have requires grad set to True,
        # but the output of frozen layers need to propagate the gradients to make sure the gradient flows.
        self.enable_input_require_grads()


def _gradient_checkpointing_disable(self):
    """
    Deactivates gradient checkpointing for the current model.

    Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
    activations".
    """
    if self.supports_gradient_checkpointing:
        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` methid
        _is_using_old_format = (
            "value" in inspect.signature(self._set_gradient_checkpointing).parameters
        )
        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=False)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    if getattr(self, "_hf_peft_config_loaded", False):
        self.disable_input_require_grads()


def _get_cached_module_file(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    module_file: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> str:
    """
    Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
    Transformers module.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `str`: The path to the module inside the cache.
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        logger.warning(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            _type=WarningType.DeprecatedArgument,
        )
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Download and cache module_file from the repo `pretrained_model_name_or_path` of grab it if it's a local file.
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
        submodule = os.path.basename(pretrained_model_name_or_path)
    else:
        submodule = pretrained_model_name_or_path.replace("/", os.path.sep)
        cached_module = try_to_load_from_cache(
            pretrained_model_name_or_path,
            module_file,
            cache_dir=cache_dir,
            revision=_commit_hash,
            repo_type=repo_type,
        )

    new_files = []
    try:
        # Load from URL or cache if already cached
        resolved_module_file = cached_file(
            pretrained_model_name_or_path,
            module_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            repo_type=repo_type,
            _commit_hash=_commit_hash,
        )
        if not is_local and cached_module != resolved_module_file:
            new_files.append(module_file)

    except EnvironmentError:
        logger.error(
            f"Could not locate the {module_file} inside {pretrained_model_name_or_path}."
        )
        raise

    # Check we have all the requirements in our environment
    modules_needed = check_imports(resolved_module_file)

    # Now we move the module inside our cached dynamic modules.
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    full_submodule = full_submodule.replace("-", "_").replace("V2.5", "V2_5")
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
    if submodule == os.path.basename(pretrained_model_name_or_path):
        # We copy local files to avoid putting too many folders in sys.path. This copy is done when the file is new or
        # has changed since last copy.
        if not (submodule_path / module_file).exists() or not filecmp.cmp(
            resolved_module_file, str(submodule_path / module_file)
        ):
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        for module_needed in modules_needed:
            module_needed = f"{module_needed}.py"
            module_needed_file = os.path.join(
                pretrained_model_name_or_path, module_needed
            )
            if not (submodule_path / module_needed).exists() or not filecmp.cmp(
                module_needed_file, str(submodule_path / module_needed)
            ):
                shutil.copy(module_needed_file, submodule_path / module_needed)
                importlib.invalidate_caches()
    else:
        # Get the commit hash
        commit_hash = extract_commit_hash(resolved_module_file, _commit_hash)

        commit_hash = "_" + commit_hash[1:]
        # The module file will end up being placed in a subfolder with the git hash of the repo. This way we get the
        # benefit of versioning.
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        full_submodule = full_submodule.replace("-", "_")
        create_dynamic_module(full_submodule)

        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        # Make sure we also have every file with relative
        for module_needed in modules_needed:
            if not (submodule_path / f"{module_needed}.py").exists():
                _get_cached_module_file(
                    pretrained_model_name_or_path,
                    f"{module_needed}.py",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                )
                new_files.append(f"{module_needed}.py")

    if len(new_files) > 0 and revision is None:
        new_files = "\n".join([f"- {f}" for f in new_files])
        repo_type_str = "" if repo_type is None else f"{repo_type}s/"
        url = f"https://huggingface.co/{repo_type_str}{pretrained_model_name_or_path}"
        logger.warning(
            f"A new version of the following files was downloaded from {url}:\n{new_files}"
            + "\n. Make sure to double-check they do not contain any added malicious code. To avoid downloading new "
            + "versions of the code file, you can pin a revision."
        )

    return os.path.join(full_submodule, module_file)


def _get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """
    Extracts all the libraries (not relative imports this time) that are imported in a file.

    Args:
        filename (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `List[str]`: The list of all packages required to use the input module.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # filter out try/except block so in custom code we can have try/except imports
    content = re.sub(
        r"\s*try\s*:\s*.*?\s*except\s*.*?:", "", content, flags=re.MULTILINE | re.DOTALL
    )

    # Imports of the form `import xxx`
    imports = re.findall(r"^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]
    while "flash_attn" in imports:
        imports.remove("flash_attn")
    return list(set(imports))


def _get_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> typing.Type:
    """
    Extracts a class from a module file, present in the local folder or repository of a model.

    <Tip warning={true}>

    Calling this function will execute the code in the module file found locally or downloaded from the Hub. It should
    therefore only be called on trusted repos.

    </Tip>

    Args:
        class_reference (`str`):
            The full name of the class to load, including its module and optionally its repo.
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

            This is used when `class_reference` does not specify another repo.
        module_file (`str`):
            The name of the module file containing the class to look for.
        class_name (`str`):
            The name of the class to import in the module.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).
        code_revision (`str`, *optional*, defaults to `"main"`):
            The specific revision to use for the code on the Hub, if the code leaves in a different repository than the
            rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based system for
            storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `typing.Type`: The class, dynamically imported from the module.

    Examples:

    ```python
    # Download module `modeling.py` from huggingface.co and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("modeling.MyBertModel", "sgugger/my-bert-model")

    # Download module `modeling.py` from a given repo and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("sgugger/my-bert-model--modeling.MyBertModel", "sgugger/another-bert-model")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        logger.warning(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            _type=WarningType.DeprecatedArgument,
        )
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    # Catch the name of the repo if it's specified in `class_reference`
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path
    module_file, class_name = class_reference.split(".")

    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision
    if code_revision is not None and code_revision != "main":
        code_revision = "_" + code_revision[1:]
    # And lastly we get the class inside our newly created module
    final_module = _get_cached_module_file(
        repo_id,
        module_file + ".py",
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=code_revision,
        local_files_only=local_files_only,
        repo_type=repo_type,
    )
    import transformers
    from packaging import version

    trans_version = transformers.__version__
    if version.parse(trans_version) < version.parse("4.39.0"):
        return get_class_in_module(
            class_name, final_module.replace(".py", "").replace("-", "_")
        )
    return get_class_in_module(class_name, final_module.replace("-", "_"))


def _pad(
    self,
    encoded_inputs: Union[
        BatchEncoding,
        List[BatchEncoding],
        Dict[str, EncodedInput],
        Dict[str, List[EncodedInput]],
        List[Dict[str, EncodedInput]],
    ],
    padding=True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_tensors=None,
    verbose: bool = True,
) -> BatchEncoding:
    """
    Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
    in the batch.

    Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
    `self.pad_token_id` and `self.pad_token_type_id`).

    Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
    text followed by a call to the `pad` method to get a padded encoding.

    <Tip>

    If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
    result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
    PyTorch tensors, you will lose the specific device of your tensors however.

    </Tip>

    Args:
        encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]`
            or `List[Dict[str, List[int]]]`):
            Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
            tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
            List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
            collate function.

            Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
            the note above for the return type.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
            `>= 7.5` (Volta).
        padding_side (`str`, *optional*):
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        return_attention_mask (`bool`, *optional*):
            Whether to return the attention mask. If left to the default, will return the attention mask according
            to the specific tokenizer's default, defined by the `return_outputs` attribute.

            [What are attention masks?](../glossary#attention-mask)
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors instead of list of python integers. Acceptable values are:

            - `'tf'`: Return TensorFlow `tf.constant` objects.
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return Numpy `np.ndarray` objects.
        verbose (`bool`, *optional*, defaults to `True`):
            Whether or not to print more information and warnings.
    """
    if self.__class__.__name__.endswith("Fast"):
        if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
            logger.warning_advice(
                f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                " using the `__call__` method is faster than using a method to encode the text followed by a call"
                " to the `pad` method to get a padded encoding."
            )
            self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # If we have a list of dicts, let's convert it in a dict of lists
    # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(
        encoded_inputs[0], Mapping
    ):
        encoded_inputs = {
            key: [example[key] for example in encoded_inputs]
            for key in encoded_inputs[0].keys()
        }

    # The model's main input name, usually `input_ids`, has been passed for padding
    if self.model_input_names[0] not in encoded_inputs:
        raise ValueError(
            "You should supply an encoding or a list of encodings to this method "
            f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
        )

    required_input = encoded_inputs[self.model_input_names[0]]

    if required_input is None or (
        isinstance(required_input, Sized) and len(required_input) == 0
    ):
        if return_attention_mask:
            encoded_inputs["attention_mask"] = []
        return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = required_input[0]
    if isinstance(first_element, (list, tuple)):
        # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
        for item in required_input:
            if len(item) != 0:
                first_element = item[0]
                break
    # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
    if not isinstance(first_element, (int, list, tuple)):
        if is_tf_tensor(first_element):
            return_tensors = "tf" if return_tensors is None else return_tensors
        elif is_torch_tensor(first_element):
            return_tensors = "pt" if return_tensors is None else return_tensors
        elif isinstance(first_element, np.ndarray):
            return_tensors = "np" if return_tensors is None else return_tensors
        else:
            raise ValueError(
                f"type of {first_element} unknown: {type(first_element)}. "
                "Should be one of a python, numpy, pytorch or tensorflow object."
            )

        for key, value in encoded_inputs.items():
            encoded_inputs[key] = to_py_obj(value)

    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
        padding=padding, max_length=max_length, verbose=verbose
    )

    required_input = encoded_inputs[self.model_input_names[0]]
    if required_input and not isinstance(required_input[0], (list, tuple)):
        try:
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )
        except TypeError:
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    batch_size = len(required_input)
    assert all(
        len(v) == batch_size for v in encoded_inputs.values()
    ), "Some items in the output dictionary have a different batch size than others."

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = max(len(inputs) for inputs in required_input)
        padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        inputs = {k: v[i] for k, v in encoded_inputs.items()}
        try:
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )
        except TypeError:
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    return BatchEncoding(batch_outputs, tensor_type=return_tensors)


def _preprocess_deepseek_v3_checkpoint(checkpoint, quant_config):
    block_n = quant_config["weight_block_size"][0]
    for key, data in checkpoint.items():
        if "weight_scale_inv" in key:
            # We don't support quantization block > 1 along N
            # so here we replicate the data along N
            new_data = torch.repeat_interleave(data, block_n, 0)
            # weight shape may not be divisible by block_n
            w_key = ".".join([*key.split(".")[:-1], "weight"])
            if w_key in checkpoint:
                N = checkpoint[w_key].size(0)
                new_data = new_data[:N, :].contiguous()
            checkpoint[key] = new_data
            del data


def load_low_precision_checkpoint(
    pathname: Union[str, os.PathLike],
    rank: int = 0,
    world_size: int = 1,
):
    r"""
    Load low precision checkpoint from a file or a directory containing multiple files.
    Supported file format: .pt, .bin, .pth, .safetensors.
    Args:
        pathname (str or os.PathLike): Path to the checkpoint file or directory containing multiple checkpoint files.
        rank (int, optional): Rank of the current process for Tensor Parallel. Default: 0.
        world_size (int, optional): World size for Tensor Parallel. Default: 1.
    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: A tuple of low precision checkpoint and quantization config.
        The quantization config contains quantization method, group size and desc_act.
    """
    assert os.path.exists(pathname), f"Checkpoint path does not exist: {pathname}"
    assert os.path.isdir(pathname), f"checkpoint path should be a directory: {pathname}"
    file_types = ["*.pt", "*.pth", "*.bin", "*.safetensors"]
    checkpoint_files = []
    file_type = None
    for pattern in file_types:
        checkpoint_files = list(pathlib.Path(pathname).glob(pattern))
        if checkpoint_files:
            file_type = pattern
            break
    assert checkpoint_files, f"Cannot find checkpoint files in path {pathname}."
    config_file = pathname + "/config.json"
    assert os.path.exists(config_file), f"Cannot find config.json in path: {pathname}."

    load_fn = partial(torch.load, weights_only=True)
    if file_type == "*.safetensors":
        try:
            import safetensors
        except ImportError:
            print("Please install safetensors package to load safetensors checkpoint.")
            exit(1)
        load_fn = safetensors.torch.load_file

    # load config.json and find quantization_config
    model_config = None
    quantization_config = None
    with open(config_file, "r", encoding="utf-8") as file:
        model_config = json.load(file)
    if "quantization_config" in model_config:
        quantization_config = model_config["quantization_config"]
    else:
        for file in ["quant_config.json", "quantize_config.json"]:
            config_file = pathname + "/" + file
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as file:
                    quantization_config = json.load(file)
                break
    assert (
        quantization_config is not None
    ), f"Cannot find quantization config in path: {pathname}."

    # read quantization config
    quant_method = quantization_config.get("quant_method", None)
    group_size = quantization_config.get("group_size", None)
    desc_act = quantization_config.get("desc_act", None)
    backend = quantization_config.get("backend", None)
    bits = quantization_config.get("bits", None)
    weight_block_size = quantization_config.get("weight_block_size", None)
    assert quant_method is not None, "Cannot find quant_method in quantization config."
    if quant_method == "gptq":
        assert desc_act is not None, "group_size should not be None for GPTQ"
        bits = 4 if bits is None else bits
    elif quant_method == "awq":
        desc_act = False
        bits = 4 if bits is None else bits
    elif quant_method == "intel/auto-round":
        if backend is None or "gptq" in backend:
            quant_method = "gptq"
        elif "awq" in backend:
            quant_method = "awq"
        else:
            raise (
                NotImplementedError,
                f"Unsupported backend: {backend} of intel/auto-round",
            )
        desc_act = False
        bits = 4 if bits is None else bits
    elif quant_method == "fp8":
        arch = model_config["architectures"][0]
        assert arch == "DeepseekV3ForCausalLM", (
            "DeepseekV3ForCausalLM is the only supported architecture for fp8 quantization."
            " Found: " + arch
        )
        assert (
            weight_block_size is not None
        ), "weight_block_size should not be None for Deepseek-V3/R1 fp8"
        assert (
            isinstance(weight_block_size, list) and len(weight_block_size) == 2
        ), "weight_block_size should be a list of two integers." " Found: " + str(
            weight_block_size
        )
        group_size = weight_block_size[1]
        desc_act = False
        bits = 8 if bits is None else bits
    elif quant_method == "rtn":
        desc_act = False
        bits = 8 if bits is None else bits
    elif quant_method == "int8":
        desc_act = False
        bits = 8 if bits is None else bits
    else:
        raise (NotImplementedError, f"Unsupported quantization method: {quant_method}")
    quant_config = {
        "quant_method": quant_method,
        "group_size": group_size,
        "desc_act": desc_act,
        "weight_block_size": weight_block_size,
        "bits": bits,
    }

    # load checkpoint files and shard if necessary
    low_precision_checkpoint = {}
    tp_grain_size = group_size if group_size > 0 else 64
    logger.debug(
        f"Loading {len(checkpoint_files)} checkpoint files on rank {rank}/{world_size}"
    )
    for ckpt in checkpoint_files:
        data_f = load_fn(ckpt)
        if quant_method == "fp8":
            _preprocess_deepseek_v3_checkpoint(data_f, quant_config)
        if world_size > 1:
            data_f_shard = shard_low_precision_checkpoint(
                data_f,
                model_config,
                rank,
                world_size,
                quant_method,
                tp_grain_size,
                desc_act,
                bits,
            )
            low_precision_checkpoint.update(data_f_shard)
        else:
            low_precision_checkpoint.update(data_f)
    logger.debug(f"loading checkpoint files done on rank {rank}/{world_size}")

    return low_precision_checkpoint, quant_config


def shard_low_precision_checkpoint(
    low_precision_checkpoint,
    model_config,
    rank,
    world_size,
    quantization_method,
    tp_grain_size,
    desc_act,
    bits,
):
    r"""
    Shard low-precision checkpoint for autoTP.
    Sharding policy:
        Attention:
            Divide N or K by num_k_v_heads if available or num_attention_heads
            Assign each rank a chunk of N or K evenly or by the 2:1:1 policy
            - 2:1:1 policy: 4 heads for 3 ranks, the first rank gets 2 and others get 1.
        MLP:
            Divide N or K by tp_grain_size
            Assign each rank a chunk of N or K evenly or by the 2:1:1 policy
    Tensor shapes for different quantization methods:
        AWQ:
            qweight shape: [K, N // 8]
            scales shape: [K // G, N]
            qzeros shape: [K // G, N // 8]
        GPTQ:
            qweight shape: [K // 8, N]
            scales shape: [K // G, N]
            qzeros shape: [K // G, N // 8]
            g_idx shape: [K]
            Similar to AWQ, but with different paths if desc_act is True or False.
            If desc_act is True:
            - If sharding along N
                - g_idx is not sharded
                - Others are sharded as usual
            - If sharding along K
                - g_idx is shared the same way as qweight
                - Scales and zero points are not sharded
            If desc_act is False:
            - g_idx is ignored
            - Others are sharded as usual
        FP8:
            qweight shape: [N, K]
            weight_scale_inv shape: [N // block_n, K // block_k], block_n = block_k = 128 for DeepSeek-V3/R1
        RTN (INC):
            qweight shape: [K // 4, N]
            scales shape: [K // G, N] or [1, N]
            qzeros shape: [K // G, N // 4] or [1, N]
        INT8 (meituan/DeepSeek-R1-Channel-INT8):
            qweight shape: [N, K]
            scales shape: [N, 1]
            qzeros shape: None

    Args:
        low_precision_checkpoint (dict): Model's low_precision_checkpoint as a state dict.
        model_config (dict): HuggingFace model config as a dict.
        rank (int): current rank for tp.
        world_size (int): total number of ranks.
        quantization_method (str): "awq" or "gptq".
        tp_grain_size (int): Grain size for MLP sharding. Usually equal to group size for
            quantization. Must be a multiple of 8.
        desc_act (bool): desc_act (a.k.a. act_order) for GPTQ. False for others.
        bits: Number of bits for quantization.

    Returns:
        A sharded checkpoint dict.

    """
    assert tp_grain_size % 8 == 0, "tp_grain_size must be a multiple of 8"
    num_heads = model_config["num_attention_heads"]
    if "num_key_value_heads" in model_config:
        num_heads = model_config["num_key_value_heads"]
    local_rank = rank

    mha_layers_split_by_N = [
        "q_proj",
        "k_proj",
        "v_proj",
        "q_b_proj",
        "kv_b_proj",
    ]
    # mlp is split with grain size = tp_grain_size
    mlp_layers_split_by_N = [
        "gate_proj",
        "up_proj",
        "fc_in",
        "fc1",
        "query_key_value",
        "w1",
        "w3",
    ]
    mha_layers_split_by_K = [
        "o_proj",
        "out_proj",
    ]
    # mlp is split with grain size = tp_grain_size
    mlp_layers_split_by_K = [
        "down_proj",
        "fc_out",
        "fc2",
        "dense",
        "dense_4h_to_h",
        "w2",
    ]
    lm_head_layers = ["lm_head"]  # split by K but not quantized
    low_precision_checkpoint_dict = low_precision_checkpoint.copy()
    head_range = [0]
    head_per_rank = num_heads // world_size
    for i in range(0, world_size):
        head_this_rank = head_per_rank
        if i < num_heads % world_size:
            head_this_rank += 1
        head_range.append(head_range[-1] + head_this_rank)
    for key in low_precision_checkpoint.keys():
        q_head_start = head_range[rank]
        q_head_end = q_head_start + (head_range[rank + 1] - head_range[rank])
        if "bias" in key:
            continue
        if any(substring in key for substring in mha_layers_split_by_N):
            data = low_precision_checkpoint_dict[key]
            if quantization_method == "awq":
                # qweight shape: [K, N // 8]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                if data.shape[-1] % head_range[-1] == 0:
                    dim = data.shape[-1] // head_range[-1]
                else:
                    assert data.shape[-1] % world_size == 0
                    dim = data.shape[-1] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    :, q_head_start * dim : q_head_end * dim
                ].contiguous()
            elif quantization_method == "gptq" or (
                quantization_method == "rtn" and bits == 4
            ):
                # qweight shape: [K // 8, N]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                # g_idx shape: [K]
                if data.shape[-1] % head_range[-1] == 0:
                    dim = data.shape[-1] // head_range[-1]
                else:
                    assert data.shape[-1] % world_size == 0
                    dim = data.shape[-1] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                if "g_idx" in key:
                    if not desc_act:
                        low_precision_checkpoint_dict.pop(key)
                else:
                    low_precision_checkpoint_dict[key] = data[
                        :, q_head_start * dim : q_head_end * dim
                    ].contiguous()
            elif quantization_method == "fp8":
                # weight shape: [N, K]
                # weight_scale_inv shape: [N // block_n, K // block_k]
                if data.shape[0] % head_range[-1] == 0:
                    dim = data.shape[0] // head_range[-1]
                else:
                    assert data.shape[0] % world_size == 0
                    dim = data.shape[0] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    q_head_start * dim : q_head_end * dim, :
                ].contiguous()
            elif quantization_method == "rtn" and bits == 8:
                # from INC, using GPTQ-like format
                # qweight shape: [K // 4, N]
                # scales shape: [K // G, N] or [1, N]
                # qzeros shape: [K // G, N // 4] or [1, N // 4]
                if data.shape[-1] % head_range[-1] == 0:
                    dim = data.shape[-1] // head_range[-1]
                else:
                    assert data.shape[-1] % world_size == 0
                    dim = data.shape[-1] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    :, q_head_start * dim : q_head_end * dim
                ].contiguous()
            elif quantization_method == "int8":
                # qweight shape: [N, K]
                # scales shape: [N, 1]
                # qzeros shape: None
                if data.shape[0] % head_range[-1] == 0:
                    dim = data.shape[0] // head_range[-1]
                else:
                    assert data.shape[0] % world_size == 0
                    dim = data.shape[0] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    q_head_start * dim : q_head_end * dim, :
                ].contiguous()
            else:
                raise AssertionError(f"{quantization_method} is not supported yet.")
        elif any(substring in key for substring in mlp_layers_split_by_N):
            data = low_precision_checkpoint_dict[key]
            if quantization_method == "awq":
                # qweight shape: [K, N // 8]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                if "scales" in key:
                    assert (
                        data.shape[1] % tp_grain_size == 0
                    ), "N must be divisible by tp_grain_size"
                    grains = data.shape[1] // tp_grain_size
                    dim = tp_grain_size
                else:
                    assert (
                        data.shape[1] * 8
                    ) % tp_grain_size == 0, "N must be divisible by tp_grain_size"
                    grains = data.shape[1] // (tp_grain_size // 8)
                    dim = tp_grain_size // 8
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    :, grains_start * dim : grains_end * dim
                ].contiguous()
            elif quantization_method == "gptq" or (
                quantization_method == "rtn" and bits == 4
            ):
                # qweight shape: [K // 8, N]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                # g_idx shape: [K]
                if "qzeros" in key:
                    assert (
                        data.shape[-1] * 8
                    ) % tp_grain_size == 0, "N must be divisible by tp_grain_size"
                    grains = data.shape[-1] // (tp_grain_size // 8)
                    dim = tp_grain_size // 8
                elif "g_idx" not in key:  # qweight, scales
                    assert (
                        data.shape[-1] % tp_grain_size == 0
                    ), "N must be divisible by tp_grain_size"
                    grains = data.shape[-1] // tp_grain_size
                    dim = tp_grain_size
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                if "g_idx" in key:
                    if not desc_act:
                        low_precision_checkpoint_dict.pop(key)
                else:
                    low_precision_checkpoint_dict[key] = data[
                        :, grains_start * dim : grains_end * dim
                    ].contiguous()
            elif quantization_method == "fp8":
                # weight shape: [N, K]
                # weight_scale_inv shape: [N // block_n, K // block_k]
                assert (
                    data.shape[0] % tp_grain_size == 0
                ), "N must be divisible by tp_grain_size"
                grains = data.shape[0] // tp_grain_size
                dim = tp_grain_size
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    grains_start * dim : grains_end * dim, :
                ].contiguous()
            elif quantization_method == "rtn" and bits == 8:
                # from INC, using GPTQ-like format
                # qweight shape: [K // 4, N]
                # scales shape: [K // G, N] or [1, N]
                # qzeros shape: [K // G, N // 4] or [1, N // 4]
                comp_ratio = 4
                if "qzeros" in key:
                    assert (
                        data.shape[-1] * comp_ratio
                    ) % tp_grain_size == 0, "N must be divisible by tp_grain_size"
                    grains = data.shape[-1] // (tp_grain_size // comp_ratio)
                    dim = tp_grain_size // comp_ratio
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    :, grains_start * dim : grains_end * dim
                ].contiguous()
            elif quantization_method == "int8":
                # qweight shape: [N, K]
                # scales shape: [N, 1]
                # qzeros shape: None
                assert (
                    data.shape[0] % tp_grain_size == 0
                ), "N must be divisible by tp_grain_size"
                grains = data.shape[0] // tp_grain_size
                dim = tp_grain_size
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    grains_start * dim : grains_end * dim, :
                ].contiguous()
            else:
                raise AssertionError(f"{quantization_method} is not supported yet.")
        elif any(substring in key for substring in mha_layers_split_by_K):
            data = low_precision_checkpoint_dict[key]
            if ("scales" in key or "qzeros" in key) and data.shape[0] == 1:
                continue
            if quantization_method == "awq":
                # qweight shape: [K, N // 8]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                if data.shape[0] % head_range[-1] == 0:
                    dim = data.shape[0] // head_range[-1]
                else:
                    assert data.shape[0] % world_size == 0
                    dim = data.shape[0] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    q_head_start * dim : q_head_end * dim
                ].contiguous()
            elif quantization_method == "gptq" or (
                quantization_method == "rtn" and bits == 4
            ):
                # qweight shape: [K // 8, N]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                # g_idx shape: [K]
                if data.shape[0] % head_range[-1] == 0:
                    dim = data.shape[0] // head_range[-1]
                else:
                    assert data.shape[0] % world_size == 0
                    dim = data.shape[0] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                if desc_act is False:
                    if "g_idx" in key:
                        low_precision_checkpoint_dict.pop(key)
                    else:
                        low_precision_checkpoint_dict[key] = data[
                            q_head_start * dim : q_head_end * dim
                        ].contiguous()
                elif "g_idx" in key or "qweight" in key:
                    low_precision_checkpoint_dict[key] = data[
                        q_head_start * dim : q_head_end * dim
                    ].contiguous()
            elif quantization_method == "fp8":
                # weight shape: [N, K]
                # weight_scale_inv shape: [N // block_n, K // block_k]
                if data.shape[-1] % head_range[-1] == 0:
                    dim = data.shape[-1] // head_range[-1]
                else:
                    assert data.shape[-1] % world_size == 0
                    dim = data.shape[-1] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    :, q_head_start * dim : q_head_end * dim
                ].contiguous()
            elif quantization_method == "rtn" and bits == 8:
                # from INC, using GPTQ-like format
                # qweight shape: [K // 4, N]
                # scales shape: [K // G, N] or [1, N]
                # qzeros shape: [K // G, N // 4] or [1, N // 4]
                if data.shape[0] == 1:
                    continue
                if data.shape[0] % head_range[-1] == 0:
                    dim = data.shape[0] // head_range[-1]
                else:
                    assert data.shape[0] % world_size == 0
                    dim = data.shape[0] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    q_head_start * dim : q_head_end * dim
                ].contiguous()
            elif quantization_method == "int8":
                # qweight shape: [N, K]
                # scales shape: [N, 1]
                # qzeros shape: None
                if data.shape[-1] == 1:
                    continue
                if data.shape[-1] % head_range[-1] == 0:
                    dim = data.shape[-1] // head_range[-1]
                else:
                    assert data.shape[-1] % world_size == 0
                    dim = data.shape[-1] // world_size
                    q_head_start = local_rank
                    q_head_end = local_rank + 1
                low_precision_checkpoint_dict[key] = data[
                    :, q_head_start * dim : q_head_end * dim
                ].contiguous()
            else:
                raise AssertionError(f"{quantization_method} is not supported yet.")
        elif any(substring in key for substring in mlp_layers_split_by_K):
            data = low_precision_checkpoint_dict[key]
            if ("scales" in key or "qzeros" in key) and data.shape[0] == 1:
                continue
            if quantization_method == "awq":
                # qweight shape: [K, N // 8]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                if "qweight" in key:
                    assert (
                        data.shape[0] % tp_grain_size == 0
                    ), "K must be divisible by tp_grain_size"
                    grains = data.shape[0] // tp_grain_size
                    dim = tp_grain_size
                else:
                    grains = data.shape[0]
                    dim = 1
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    grains_start * dim : grains_end * dim
                ].contiguous()
            elif quantization_method == "gptq" or (
                quantization_method == "rtn" and bits == 4
            ):
                # qweight shape: [K // 8, N]
                # scales shape: [K // G, N]
                # qzeros shape: [K // G, N // 8]
                # g_idx shape: [K]
                if "qweight" in key:
                    assert (
                        data.shape[0] * 8 % tp_grain_size == 0
                    ), "K must be divisible by tp_grain_size"
                    grains = data.shape[0] // (tp_grain_size // 8)
                    dim = tp_grain_size // 8
                elif "g_idx" in key:
                    assert data.shape[0] % tp_grain_size == 0
                    grains = data.shape[0] // tp_grain_size
                    dim = tp_grain_size
                else:
                    grains = data.shape[0]
                    dim = 1
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                if desc_act is False:
                    if "g_idx" in key:
                        low_precision_checkpoint_dict.pop(key)
                    else:
                        low_precision_checkpoint_dict[key] = data[
                            grains_start * dim : grains_end * dim
                        ].contiguous()
                elif "g_idx" in key or "qweight" in key:
                    low_precision_checkpoint_dict[key] = data[
                        grains_start * dim : grains_end * dim
                    ].contiguous()
            elif quantization_method == "fp8":
                # weight shape: [N, K]
                # weight_scale_inv shape: [N // block_n, K // block_k]
                if "weight" in key and "scale" not in key:
                    assert (
                        data.shape[-1] % tp_grain_size == 0
                    ), "K must be divisible by tp_grain_size"
                    grains = data.shape[-1] // tp_grain_size
                    dim = tp_grain_size
                else:
                    grains = data.shape[-1]
                    dim = 1
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    :, grains_start * dim : grains_end * dim
                ].contiguous()
            elif quantization_method == "rtn" and bits == 8:
                # from INC, using GPTQ-like format
                # qweight shape: [K // 4, N]
                # scales shape: [K // G, N] or [1, N]
                # qzeros shape: [K // G, N // 4] or [1, N // 4]
                if data.shape[0] == 1:
                    continue
                comp_ratio = 4
                if "qweight" in key:
                    assert (
                        data.shape[0] * comp_ratio % tp_grain_size == 0
                    ), "K must be divisible by tp_grain_size"
                    grains = data.shape[0] // (tp_grain_size // comp_ratio)
                    dim = tp_grain_size // comp_ratio
                else:
                    grains = data.shape[0]
                    dim = 1
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    grains_start * dim : grains_end * dim
                ].contiguous()
            elif quantization_method == "int8":
                # qweight shape: [N, K]
                # scales shape: [N, 1]
                # qzeros shape: None
                if data.shape[-1] == 1:
                    continue
                if "scales" in key:
                    grains = data.shape[-1]
                    dim = 1
                else:
                    grains = data.shape[-1] // tp_grain_size
                    dim = tp_grain_size
                grains_per_rank = grains // world_size
                grains_rem = grains % world_size
                grains_start = grains_per_rank * local_rank + min(
                    local_rank, grains_rem
                )
                grains_end = (
                    grains_start
                    + grains_per_rank
                    + (1 if local_rank < grains_rem else 0)
                )
                low_precision_checkpoint_dict[key] = data[
                    :, grains_start * dim : grains_end * dim
                ].contiguous()
            else:
                raise AssertionError(f"{quantization_method} is not supported yet.")
        elif any(substring in key for substring in lm_head_layers):
            # lm_head: [N, K] (not quantized)
            # Same for all quantization methods
            data = low_precision_checkpoint_dict[key]
            assert (
                data.shape[1] % tp_grain_size == 0
            ), "K must be divisible by tp_grain_size"
            grains = data.shape[1] // tp_grain_size
            dim = tp_grain_size
            grains_per_rank = grains // world_size
            grains_rem = grains % world_size
            grains_start = grains_per_rank * local_rank + min(local_rank, grains_rem)
            grains_end = (
                grains_start + grains_per_rank + (1 if local_rank < grains_rem else 0)
            )
            low_precision_checkpoint_dict[key] = data[
                :, grains_start * dim : grains_end * dim
            ].contiguous()

    return low_precision_checkpoint_dict
