from torch.testing._internal.common_utils import TestCase


class TestVerbose(TestCase):
    def test_ipex_log_level(self):
        cmd = """
import torch
import intel_extension_for_pytorch
torch.xpu.set_log_level(2)
res["res"] = torch.xpu.get_log_level()"""

        global res
        res = {}
        exec(cmd)
        assert "INFO" in str(res["res"]), "Required info log level not found"

    def test_ipex_log_component(self):
        cmd = """
import torch
import intel_extension_for_pytorch
res["default_log_component"] = torch.xpu.get_log_component()"""

        global res
        res = {}
        exec(cmd)
        print(str(res["default_log_component"]))
        # default log_component should be all
        assert "ALL" in str(
            res["default_log_component"]
        ), "Default ipex_log_component should be ALL"

        cmd = """
import torch
import intel_extension_for_pytorch
torch.xpu.set_log_component("OPS;MEMORY/")
res["log_component"] = torch.xpu.get_log_component()"""
        exec(cmd)
        assert "OPS;MEMORY" in str(
            res["log_component"]
        ), "Default ipex_log_component should be ALL"

    def test_ipex_split_size(self):
        cmd = """
import torch
import intel_extension_for_pytorch
res["default_split_size"] = torch.xpu.get_log_split_file_size()
"""
        global res
        res = {}
        exec(cmd)
        assert -1 == int(
            res["default_split_size"]
        ), "default ipex log split size should be -1"

        cmd = """
import torch
import intel_extension_for_pytorch
torch.xpu.set_log_split_file_size(10)
res["set_split_size"] = torch.xpu.get_log_split_file_size()
"""
        exec(cmd)
        print(res["set_split_size"])
        assert 10 == int(
            res["set_split_size"]
        ), "setting ipex log split size should be 10"

    def test_ipex_rotate_size(self):
        cmd = """
import torch
import intel_extension_for_pytorch
res["default_rotate_size"] = torch.xpu.get_log_rotate_file_size()
"""
        global res
        res = {}
        exec(cmd)
        assert -1 == int(
            res["default_rotate_size"]
        ), "default ipex log split size should be -1"

        cmd = """
import torch
import intel_extension_for_pytorch
torch.xpu.set_log_rotate_file_size(50)
res["set_rotate_size"] = torch.xpu.get_log_rotate_file_size()
"""
        exec(cmd)
        assert 50 == int(
            res["set_rotate_size"]
        ), "setting ipex log split size should be 50"
