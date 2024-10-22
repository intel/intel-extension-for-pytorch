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
    full_submodule = full_submodule.replace("-", "_")
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
    if code_revision is not None:
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
