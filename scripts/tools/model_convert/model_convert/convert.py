import os
from collections import namedtuple
import torch
import intel_extension_for_pytorch
from ruamel.yaml import YAML
from .wrap_api import WrapAPI

yaml = YAML(typ="safe", pure=True)
lazy_init_list = ["_is_in_bad_fork",
                  "_lazy_call",
                  "_lazy_init",
                  "is_initialized",
                  "DeferredCudaCallError",
                  "_LazySeedTracker"]
not_callable_list = ["is_bf16_supported",
                     "has_half",
                     "_initialization_lock",
                     "_initialized",
                     "_lazy_seed_tracker",
                     "_queued_calls",
                     "_tls",
                     "threading",
                     "traceback"]


def get_api_info():
    api_map = namedtuple("api_entry", "api_mod api_name api_wrap")

    torch_to_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yaml/register_torch_to_api.yaml"
    )
    torch_create_tensor_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "yaml/register_torch_create_tensor_api.yaml",
    )
    torch_data_loader_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "yaml/register_torch_data_loader_api.yaml",
    )
    torch_ddp_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yaml/register_torch_ddp_api.yaml"
    )
    torch_pass_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yaml/register_torch_pass_api.yaml"
    )
    torch_failure_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "yaml/register_torch_failure_api.yaml",
    )
    torch_ccl_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yaml/register_torch_ccl_api.yaml"
    )

    torch_create_tensor_unsupported_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "yaml/register_torch_create_tensor_api_unsupported.yaml",
    )

    with open(torch_to_file_path, "r", encoding="utf-8") as f:
        to_list = yaml.load(f.read())
    with open(torch_create_tensor_file_path, "r", encoding="utf-8") as f:
        create_tensor_list = yaml.load(f.read())
    with open(torch_data_loader_file_path, "r", encoding="utf-8") as f:
        data_loader_list = yaml.load(f.read())
    with open(torch_ddp_file_path, "r", encoding="utf-8") as f:
        ddp_list = yaml.load(f.read())
    with open(torch_pass_file_path, "r", encoding="utf-8") as f:
        pass_list = yaml.load(f.read())
    with open(torch_failure_file_path, "r", encoding="utf-8") as f:
        failure_list = yaml.load(f.read())
    with open(torch_ccl_file_path, "r", encoding="utf-8") as f:
        ccl_list = yaml.load(f.read())

    with open(torch_create_tensor_unsupported_file_path, "r", encoding="utf-8") as f:
        create_tensor_unsupported_list = yaml.load(f.read())

    cuda_list = []
    for item in dir(torch.cuda):
        cuda_list.append("torch.cuda." + item)

    tmp_list = []
    for item in dir(torch.xpu):
        tmp_list.append("torch.cuda." + item)

    for item in lazy_init_list:
        tmp_list.append("torch.cuda." + item)
    tmp_list.append("torch.cuda._CudaBase")
    cuda_xpu_common_list = list(set(cuda_list).intersection(set(tmp_list)))
    cuda_support_xpu_not_list = list(
        set(cuda_list).difference(set(tmp_list)).difference(set(failure_list))
    )

    torch_api_map_list = []
    api_list_supported = (
        to_list
        + create_tensor_list
        + data_loader_list
        + ddp_list
        + cuda_xpu_common_list
        + pass_list
        + ccl_list
        + failure_list
    )
    api_list_unsupported = create_tensor_unsupported_list + cuda_support_xpu_not_list

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

    if os.getenv("VERBOSE_MODEL_CONVERT") == "1":
        print("#### Torch {} Device API Comparison ####".format(torch.__version__))
        print("#### Following torch api are  supported by xpu:")
        print(api_list_supported)
        print(
            "#### Warning: following torch api cuda supports while xpu does not support:"
        )
        print(api_list_unsupported)
        print("###########################################")

    if os.getenv("UPDATE_SUPPORT_LIST") == "1":
        with open("api_supported_by_xpu.yaml", "w", encoding="utf-8") as f:
            yaml.dump(api_list_supported, f)
        with open("api_unsupported_by_xpu.yaml", "w", encoding="utf-8") as f:
            yaml.dump(api_list_unsupported, f)

    return torch_api_map_list, cuda_xpu_common_list, cuda_support_xpu_not_list


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
    def __init__(self):
        self.torch_api_map = set()

    def convert_api(self):
        torch_api_map_list, cuda_xpu_common_list, _ = get_api_info()
        for item in torch_api_map_list:
            self.torch_api_map.add(item)
        for item in self.torch_api_map:
            api = get_attr(item.api_mod, item.api_name)
            if item.api_name == "cuda":
                api = get_attr(item.api_mod, "to")
            full_api_name = "torch.cuda." + item.api_name
            if full_api_name in cuda_xpu_common_list:
                if item.api_name in lazy_init_list:
                    if item.api_name == "DeferredCudaCallError":
                        api = get_attr(torch.xpu.lazy_init, "DeferredXPUCallError")
                    else:
                        api = get_attr(torch.xpu.lazy_init, item.api_name)
                elif item.api_name == "_CudaBase":
                    api = get_attr(torch.xpu, "_XPUBase")
                else:
                    api = get_attr(torch.xpu, item.api_name)
            if api is not None:
                set_attr(item.api_mod, item.api_name, item.api_wrap(api))

    def convert_var(self):
        torch.has_cuda = True
        torch.version.cuda = "11.7"
        torch.cuda.has_half = True
        torch.cuda._initialization_lock = torch.xpu.lazy_init._initialization_lock
        torch.cuda._initialized = torch.xpu.lazy_init._initialized
        torch.cuda._lazy_seed_tracker = torch.xpu.lazy_init._lazy_seed_tracker
        torch.cuda._queued_calls = torch.xpu.lazy_init._queued_calls
        torch.cuda._tls = torch.xpu.lazy_init._tls
        torch.cuda.threading = torch.xpu.lazy_init.threading
        torch.cuda.traceback = torch.xpu.lazy_init.traceback


def convert():
    helper.convert_var()
    helper.convert_api()


helper = WrapHelper()
