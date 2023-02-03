import os
from collections import namedtuple
import torch
import intel_extension_for_pytorch
import ruamel.yaml as yaml
from .wrap_api import WrapAPI


def get_api_info(save_yaml=False):
    api_map = namedtuple('api_entry', 'api_mod api_name api_wrap')

    torch_to_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_to_api.yaml")
    torch_create_tensor_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_create_tensor_api.yaml")
    torch_data_loader_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_data_loader_api.yaml")
    torch_ddp_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_ddp_api.yaml")
    torch_cuda_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_cuda_api.yaml")
    torch_pass_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_pass_api.yaml")
    torch_failure_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_failure_api.yaml")
    torch_ccl_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_ccl_api.yaml")

    torch_create_tensor_unsupported_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "yaml/register_torch_create_tensor_api_unsupported.yaml")

    with open(torch_to_file_path, 'r', encoding='utf-8') as f:
        to_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_create_tensor_file_path, 'r', encoding='utf-8') as f:
        create_tensor_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_data_loader_file_path, 'r', encoding='utf-8') as f:
        data_loader_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_ddp_file_path, 'r', encoding='utf-8') as f:
        ddp_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_cuda_file_path, 'r', encoding='utf-8') as f:
        cuda_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_pass_file_path, 'r', encoding='utf-8') as f:
        pass_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_failure_file_path, 'r', encoding=yaml.Loader) as f:
        failure_list = yaml.load(f.read(), Loader=yaml.Loader)
    with open(torch_ccl_file_path, 'r', encoding='utf-8') as f:
        ccl_list = yaml.load(f.read(), Loader=yaml.Loader)

    with open(torch_create_tensor_unsupported_file_path, 'r', encoding='utf-8') as f:
        create_tensor_unsupported_list = yaml.load(
            f.read(), Loader=yaml.Loader)

    tmp_list = []
    xpu_list = dir(torch.xpu)
    for item in xpu_list:
        tmp_list.append('torch.cuda.' + item)

    cuda_xpu_common_list = list(
        set(cuda_list).intersection(set(tmp_list)))
    cuda_support_xpu_not_list = list(
        set(cuda_list).difference(set(tmp_list)).difference(set(failure_list)))

    torch_api_map_list = []
    api_list_supported = to_list + create_tensor_list + \
        data_loader_list + ddp_list + cuda_xpu_common_list + pass_list + ccl_list + failure_list
    api_list_unsupported = create_tensor_unsupported_list + cuda_support_xpu_not_list

    api_list = api_list_supported + api_list_unsupported
    for item in api_list:
        item_list = item.split('.')
        mod = item_list[:-1]
        length = len(mod)
        new_mod = item_list[0]
        for i in range(1, length):
            new_mod = new_mod + '.' + item_list[i]

        eval_new_mod = new_mod
        if new_mod != '':
            eval_new_mod = eval(new_mod)
        api_name = item_list[-1]
        if item in api_list_unsupported:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_skip))
        elif item in pass_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_pass))
        elif item in failure_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_failure))
        elif item in ccl_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_ccl))
        elif item in to_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_to))
        else:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_common))

    if os.getenv("VERBOSE_MODEL_CONVERT") == "1":
        print('#### Torch {} Device API Comparison ####'.format(torch.__version__))
        print('#### Following torch api are  supported by xpu:')
        print(api_list_supported)
        print('#### Warning: following torch api cuda supports while xpu does not support:')
        print(api_list_unsupported)
        print('###########################################')

    if save_yaml:
        with open('api_list_supported.yaml', "w", encoding="utf-8") as f:
            yaml.dump(api_list_supported, f, Dumper=yaml.RoundTripDumper)
        with open('api_list_unsupported.yaml', "w", encoding="utf-8") as f:
            yaml.dump(api_list_unsupported, f, Dumper=yaml.RoundTripDumper)

    return torch_api_map_list, cuda_xpu_common_list, cuda_support_xpu_not_list


class WrapHelper():
    def __init__(self):
        self.torch_api_map = set()

    def convert_api(self):
        torch_api_map_list, cuda_xpu_common_list, _ = get_api_info()
        for item in torch_api_map_list:
            self.torch_api_map.add(item)
        for item in self.torch_api_map:
            api = getattr(item.api_mod, item.api_name)
            if item.api_name == 'cuda':
                api = getattr(item.api_mod, 'to')
            full_api_name = 'torch.cuda.' + item.api_name
            if full_api_name in cuda_xpu_common_list:
                api = getattr(torch.xpu, item.api_name)
            setattr(item.api_mod, item.api_name, item.api_wrap(api))
    def convert_var(self):
        torch.has_cuda = True
        torch.version.cuda = '11.7'

def convert():
    helper.convert_var()
    helper.convert_api()


helper = WrapHelper()
