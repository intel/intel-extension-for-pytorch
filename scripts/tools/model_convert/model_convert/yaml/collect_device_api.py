import torch
import argparse
import ruamel.yaml as yaml


def collect_device_api(device, save, file_path):
    """Collect cuda or xpu device api list."""
    if device == "cuda":
        cuda_api_list = dir(torch.cuda)
        torch_cuda_api_list = []
        for item in cuda_api_list:
            torch_cuda_api_list.append("torch.cuda." + item)
        print("torch cuda api list: {}".format(cuda_api_list))
        if save:
            with open(file_path, "w", encoding="utf-8") as f:
              yaml.dump(torch_cuda_api_list, f)
    else:
        import intel_extension_for_pytorch

        if torch.xpu.is_available():
            xpu_api_list = dir(torch.xpu)
            torch_xpu_api_list = []
            for item in xpu_api_list:
                torch_xpu_api_list.append("torch.xpu." + item)
            print("torch xpu api list: {}".format(xpu_api_list))
            if save:
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(torch_xpu_api_list, f)
        else:
            print("xpu is not available, check your env first")


def main():
    """
    Main function of torch device api collection parser.
    """
    parser = argparse.ArgumentParser(description=f"torch device api collection parser")
    parser.add_argument(
        "--device", "-d", default="cuda", help="change files in aggressive mode"
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="save device api list in yaml file"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="torch_cuda_api_list.yaml",
        help="save device api list in this yaml file",
    )
    args = parser.parse_args()
    collect_device_api(args.device, args.save, args.file)


if __name__ == "__main__":
    main()
