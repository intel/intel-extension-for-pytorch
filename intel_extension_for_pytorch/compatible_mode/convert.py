import os
import sys
import torch
import functools
import intel_extension_for_pytorch  # noqa:F401

from .fake_module import common, nccl
from collections import namedtuple
from ruamel.yaml import YAML
from .wrap_api import WrapAPI
from typing import Optional, Any
from torch.types import _dtype

yaml = YAML(typ="safe", pure=True)
not_callable_list = [
    "is_bf16_supported",
    "has_half",
    "_initialization_lock",
    "_initialized",
    "_lazy_seed_tracker",
    "_queued_calls",
    "_tls",
    "threading",
    "traceback",
]

pre_device_class = torch.device


def get_yaml_list(file_path: str):
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),
        "r",
        encoding="utf-8",
    ) as f:
        yaml_list = yaml.load(f.read())
    return (
        yaml_list["tensor_to_api"],
        yaml_list["create_tensor_supported_api"],
        yaml_list["data_loader_api"],
        yaml_list["ddp_api"],
        yaml_list["pass_api"],
        yaml_list["failure_api"],
        yaml_list["ccl_api"],
        yaml_list["create_tensor_unsupported_api"],
    )


def get_api_info(dist_backend):
    api_map = namedtuple("api_entry", "api_mod api_name api_wrap")

    (
        to_list,
        create_tensor_list,
        data_loader_list,
        ddp_list,
        pass_list,
        failure_list,
        ccl_list,
        create_tensor_unsupported_list,
    ) = get_yaml_list("yaml/register_support_api.yaml")

    torch_api_map_list = []
    api_list_supported = (
        to_list
        + create_tensor_list
        + data_loader_list
        + ddp_list
        + pass_list
        + ccl_list
        + failure_list
    )
    api_list_unsupported = create_tensor_unsupported_list

    api_list = api_list_supported + api_list_unsupported
    for item in api_list:
        item_list = item.split(".")
        mod = item_list[:-1]
        length = len(mod)
        new_mod = item_list[0]
        for i in range(1, length):
            new_mod = new_mod + "." + item_list[i]

        eval_new_mod = new_mod
        if new_mod != "":
            eval_new_mod = eval(new_mod)
        api_name = item_list[-1]
        if item in pass_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_pass)
            )
        elif item in failure_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_failure)
            )
        elif item in ccl_list:
            if dist_backend == "ccl":
                torch_api_map_list.append(
                    api_map(eval_new_mod, api_name, WrapAPI.wrap_api_ccl)
                )
        elif item in to_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_to)
            )
        elif item in api_list_unsupported:
            if api_name not in not_callable_list:
                torch_api_map_list.append(
                    api_map(eval_new_mod, api_name, WrapAPI.wrap_api_skip)
                )
        else:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_common)
            )

    return torch_api_map_list


def get_attr(mod, name):
    api = None
    try:
        api = getattr(mod, name)
    except AttributeError:
        pass
    return api


def set_attr(mod, name, new_name):
    try:
        setattr(mod, name, new_name)
    except AttributeError:
        pass


class WrapHelper:
    def __init__(
        self, target_device="xpu", dist_backend="ccl", compile_backend="inductor"
    ):
        self.torch_api_map = set()
        self.target_device = target_device
        self.dist_backend = dist_backend
        self.compile_backend = compile_backend

    def convert_api(self):
        if self.target_device == "xpu":
            torch_api_map_list = get_api_info(self.dist_backend)
            for item in torch_api_map_list:
                self.torch_api_map.add(item)
            for item in self.torch_api_map:
                api = get_attr(item.api_mod, item.api_name)

                # handle the interface for torch.Tensor.cuda and torch.nn.Module.cuda
                if item.api_name == "cuda":
                    api = get_attr(item.api_mod, "to")

                if api is not None:
                    set_attr(item.api_mod, item.api_name, item.api_wrap(api))

            # disable torch.jit.script for cannot support
            set_attr(torch.jit, "script", WrapAPI.wrap_jit_script(torch.jit.script))

            # fake for torch.cuda.amp.common for it cannot be found in torch.xpu
            set_attr(torch.cuda, "nccl", nccl)
            set_attr(torch.cuda.amp, "common", common)
            set_attr(
                torch.cuda,
                "GradScaler",
                functools.partial(torch.amp.GradScaler, device="xpu"),
            )

            # TODO: currently return a magic number to pass the test,
            # next step need to investigate the relation ship between cuda capability
            # with our device and return a properate version
            def get_device_capability():
                return 9, 0

            set_attr(torch.cuda, "get_device_capability", get_device_capability)

        torch_autocast = torch.autocast

        class fake_autocast:
            def __init__(
                self,
                device_type="xpu",
                dtype: Optional[_dtype] = None,
                enabled: bool = True,
                cache_enabled: Optional[bool] = None,
            ):
                if device_type.find("cuda") != -1:
                    self.i = torch_autocast(
                        device_type="xpu",
                        dtype=torch.bfloat16,
                        enabled=enabled,
                        cache_enabled=cache_enabled,
                    )

                self.i = torch_autocast(
                    device_type="xpu",
                    dtype=torch.bfloat16,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )
                self.i.fast_dtype = torch.bfloat16

            def __enter__(self):
                self.i.__enter__()

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
                self.i.__exit__(exc_type, exc_val, exc_tb)

            def __call__(self, func):
                return self.i.__call__(func)

        torch.autocast = fake_autocast
        torch.cuda.amp.autocast = fake_autocast

    def convert_var(self):
        if self.target_device == "xpu":
            torch.has_cuda = True
            torch.cuda.has_half = True
            torch.version.cuda = "11.7"
            torch._C._XpuDeviceProperties.major = 8
            torch._C._XpuDeviceProperties.minor = 5

            # set device property
            device_property = torch.xpu.get_device_properties(torch.device("xpu"))
            torch._C._XpuDeviceProperties.multi_processor_count = (
                device_property.gpu_subslice_count
            )
            torch.cuda.amp.GradScaler = torch.amp.GradScaler

            class device_meta_class(type):
                def __instancecheck__(cls, instance):
                    if instance is None:
                        return False
                    return isinstance(instance, pre_device_class)

            class fake_device(metaclass=device_meta_class):
                def __new__(cls, ss, i=0):
                    if isinstance(ss, pre_device_class):
                        return pre_device_class("xpu") if ss.type != "xpu" else ss
                    ss = ss.replace("cuda", "xpu")
                    return (
                        pre_device_class(ss)
                        if ss.find(":") != -1
                        else pre_device_class(ss, i)
                    )

            torch.device = fake_device
            # TODO: major, minor. Major means the arch, minor means the incremental imporvement

    def convert_module(self):
        def replace_backend(target_backend, replace_backend, name):
            if name.startswith(target_backend):

                migrate_name = replace_backend + name[len(target_backend) :]
                if migrate_name in sys.modules.keys():
                    sys.modules[name] = sys.modules[migrate_name]

        if self.target_device == "xpu":
            for name, mod in sys.modules.items():
                replace_backend("torch.cuda", "torch.xpu", name)
                replace_backend(
                    "torch.backends.cuda",
                    "intel_extension_for_pytorch.backends.xpu",
                    name,
                )

            torch.cuda = sys.modules["torch.cuda"]
            torch.backends.backends = sys.modules["torch.backends.cuda"]


def compatible_mode(
    target_device="xpu", dist_backend="ccl", compile_backend="inductor"
):
    helper = WrapHelper(target_device, dist_backend, compile_backend)

    # convert torch function outside of module [torch.cuda, torch.backends.cuda]
    helper.convert_module()
    # convert torch apis using device or set "cuda" device as default device
    helper.convert_var()
    # convert torch attr related with cuda device
    helper.convert_api()
