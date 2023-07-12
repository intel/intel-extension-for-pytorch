import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest
from torch.xpu.cpp_extension import IS_WINDOWS 


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestDPCPPExtensionMethod(TestCase):
    def test_add_ninja(self):
        import mod_test_add_ninja
        a = torch.rand(2, 3).to(dpcpp_device)
        b = torch.rand(2, 3).to(dpcpp_device)
        c = torch.empty_like(a)
        d = a + b
        mod_test_add_ninja.add(a, b, c)
        self.assertEqual(d.to(cpu_device), c.to(cpu_device))

    def test_add_non_ninja(self):
        import mod_test_add_non_ninja
        a = torch.rand(2, 3).to(dpcpp_device)
        b = torch.rand(2, 3).to(dpcpp_device)
        c = torch.empty_like(a)
        d = a + b
        mod_test_add_non_ninja.add(a, b, c)
        self.assertEqual(d.to(cpu_device), c.to(cpu_device))

    @pytest.mark.skipif(IS_WINDOWS, reason="ldd is not supported in Windows.")
    def test_dpcpp_extension_link_libraries(self):
        import subprocess
        import os
        path = os.path.join(os.path.dirname(__file__), "build")
        so_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if "mod_test_add_non_ninja" in file:
                    so_file = os.path.join(root, file)
                    break
        self.assertTrue(so_file, "mod_test_add_non_ninja.so not found.")
        results = subprocess.check_output(['ldd', so_file])
        link_libraries = results.split(b'\n')
        for lib_link in link_libraries:
            if b'libze_loader.so' in lib_link and b'not found' not in lib_link:
                self.assertTrue(lib_link)
                return
        self.assertTrue(False, "ze_loader not linked or libze_loader.so not found.")
