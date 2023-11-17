import torch
import torch.nn as nn


class IPEXTransformerBlock(nn.Module):
    def __init__(self, module, config, dtype, device, module_name) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.device = device
        self.module_name = module_name

    def contruct_module_from_config(self):
        pass

    def print_all_paramter_with_name(self):
        for name, param in self.ipex_optimized_module.named_parameters():
            name = self.module_name + "." + name
            print("module name: ", name)
            print("module param: ", param)

    def transpose_parameter(self):
        pass

    def port_attn_parameter(self):
        raise NotImplementedError

    def port_mlp_parameter(self):
        raise NotImplementedError

    def port_norm_parameter(self):
        raise NotImplementedError

    def port_module_specific_parameter(self):
        pass

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameter()
        self.port_mlp_parameter()
        self.port_norm_parameter()
        self.port_module_specific_parameter()
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        # for debug
        # self.print_all_paramter_with_name

    def get_optimized_module(self):
        return self

    def release_resources(self):
        for module in self.ipex_optimized_module.children():
            if hasattr(module, "release_resources"):
                module.release_resources()
