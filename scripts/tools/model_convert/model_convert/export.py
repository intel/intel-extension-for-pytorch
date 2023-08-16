import os
import torch
import intel_extension_for_pytorch
from search_and_replace import read_file, search_and_replace, search_and_replace_via_ast
import argparse
import subprocess
from ruamel.yaml import YAML
yaml = YAML(typ="safe", pure=True)


def walkdir(path=".", aggressive=False):
    """walk through all files contain specific content in specific directory"""
    file_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            abspath = os.path.abspath(os.path.join(dirpath, filename))
            if "cuda" in read_file(abspath):
                if aggressive:
                    file_list.append(abspath)
                else:
                    if ".py" in abspath or ".ipynb" in abspath:
                        file_list.append(abspath)
    return file_list


def model_script_convert(file_list, in_place, aggressive, verbose):
    """convert model script from cuda for xpu."""
    torch_create_tensor_unsupported_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "yaml/register_torch_create_tensor_api_unsupported.yaml",
    )

    with open(torch_create_tensor_unsupported_file_path, "r", encoding="utf-8") as f:
        create_tensor_unsupported_list = yaml.load(f.read())

    cuda_list = []
    for item in dir(torch.cuda):
        cuda_list.append("torch.cuda." + item)

    tmp_list = []
    for item in dir(torch.xpu):
        tmp_list.append("torch.cuda." + item)
    cuda_xpu_common_list = list(set(cuda_list).intersection(set(tmp_list)))
    cuda_support_xpu_not_list = list(set(cuda_list).difference(set(tmp_list)))

    api_list_unsupported = create_tensor_unsupported_list + cuda_support_xpu_not_list
    if verbose:
        print("#### Torch {} Device API Comparison ####".format(torch.__version__))
        print(
            "#### Warning: following torch api cuda supports while xpu does not support:"
        )
        print(api_list_unsupported)
        print("###########################################")
    search_and_replace_via_ast(file_list)
    corner_cases_dic = {"torch.has_cuda": "True",
                        "torch.version.cuda": "11.7",
                        "torch.cuda.has_half": "True",
                        "torch.cuda._CudaBase": "torch.xpu._XPUBase",
                        "torch.cuda._initialization_lock": "torch.xpu.lazy_init._initialization_lock",
                        "torch.cuda._initialized": "torch.xpu.lazy_init._initialized",
                        "torch.cuda._lazy_seed_tracker": "torch.xpu.lazy_init._lazy_seed_tracker",
                        "torch.cuda._queued_calls": "torch.xpu.lazy_init._queued_calls",
                        "torch.cuda._tls": "torch.xpu.lazy_init._tls",
                        "torch.cuda.threading": "torch.xpu.lazy_init.threading",
                        "torch.cuda.traceback": "torch.xpu.lazy_init.traceback",
                        "torch.cuda._is_in_bad_fork": "torch.xpu.lazy_init._is_in_bad_fork",
                        "torch.cuda._lazy_call": "torch.xpu.lazy_init._lazy_call",
                        "torch.cuda._lazy_init": "torch.xpu.lazy_init._lazy_init",
                        "torch.cuda.is_initialized": "torch.xpu.lazy_init.is_initialized",
                        "torch.cuda.DeferredCudaCallError": "torch.xpu.lazy_init.DeferredXPUCallError",
                        "torch.cuda._LazySeedTracker": "torch.xpu.lazy_init._LazySeedTracker"}
    if not aggressive:
        # roll back the definately unsupported xpu api
        for api_name in cuda_support_xpu_not_list:
            old_api_name = api_name.replace("cuda", "xpu")
            new_api_name = api_name
            is_corner_case = False
            if new_api_name in corner_cases_dic.keys():
                is_corner_case = True
                new_api_name = corner_cases_dic[new_api_name]
            # TODO: use regrex match
            regrex_match = False
            revert, revert_file_list = search_and_replace(
                old_api_name, new_api_name, file_list, regrex_match, verbose
            )
            if revert:
                print("###########################################")
                if is_corner_case:
                    print("Warning: {0} will be mapped to {1}".format(api_name, new_api_name))
                else:
                    print(
                        "Warning: {0} is not supported by torch.xpu, will not change it in following files".format(
                            api_name
                        )
                    )
                for revert_file in revert_file_list:
                    print(revert_file)
                print("###########################################")
    for file_name in file_list:
        dirname, fname = os.path.split(file_name)
        split_fname = list(os.path.splitext(fname))
        new_fname = split_fname[0] + ".xpu" + split_fname[1]
        new_file_name = os.path.join(dirname, new_fname)
        if os.path.exists(new_file_name):
            org = read_file(file_name)
            new = read_file(new_file_name)
            if org == new:
                os.remove(new_file_name)
            elif in_place:
                os.rename(new_file_name, file_name)
            else:
                print("diff {0} {1}".format(file_name, new_file_name))
                ret = subprocess.call(["diff", file_name, new_file_name])
                os.remove(new_file_name)


def main():
    """
    Main function of model script convert parser.
    """
    parser = argparse.ArgumentParser(description=f"model script convert parser")
    parser.add_argument(
        "--path", "-p", required=True, help="search the string from this path"
    )
    parser.add_argument(
        "--in-place", "-i", action="store_true", help="change files in-place"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="turn on verbose mode"
    )
    parser.add_argument(
        "--aggressive",
        "-a",
        action="store_true",
        help="change files in aggressive mode",
    )
    args = parser.parse_args()
    file_list = walkdir(args.path, args.aggressive)
    model_script_convert(file_list, args.in_place, args.aggressive, args.verbose)


if __name__ == "__main__":
    main()
