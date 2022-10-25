import unittest
import os
import subprocess

import intel_extension_for_pytorch._C as core

supported_isa_set = ["default", "avx2", "avx2_vnni", "avx512", "avx512_vnni", "avx512_bf16", "amx"]

def get_isa_val(isa_name):
    if isa_name == "default":
        return 0
    elif isa_name == "avx2":
        return 1
    elif isa_name == "avx2_vnni":
        return 2
    elif isa_name == "avx512":
        return 3
    elif isa_name == "avx512_vnni":
        return 4
    elif isa_name == "avx512_bf16":
        return 5
    elif isa_name == "amx":
        return 6
    else:
        return 100

def get_ipex_isa_env_setting():
    env_isa = os.getenv('ATEN_CPU_CAPABILITY')
    return env_isa

def get_currnet_isa_level():
    return core._get_current_isa_level().lower()

def get_highest_binary_support_isa_level():
    return core._get_highest_binary_support_isa_level().lower()

def get_highest_cpu_support_isa_level():
    return core._get_highest_cpu_support_isa_level().lower()

def check_not_sync_onednn_isa_level():
    return core._check_not_sync_onednn_isa_level()

class TestDynDisp(unittest.TestCase):

    def test_manual_select_kernel(self):
        env_isa = get_ipex_isa_env_setting()
        cur_isa = get_currnet_isa_level()
        max_bin_isa = get_highest_binary_support_isa_level()
        max_cpu_isa = get_highest_cpu_support_isa_level()

        expected_isa_val = min(get_isa_val(max_bin_isa), get_isa_val(max_cpu_isa))
        
        if (env_isa != None):
            expected_isa_val = min(get_isa_val(env_isa), expected_isa_val)
        
        actural_isa_val = get_isa_val(cur_isa)

        # Isa level and compiler version are not linear relationship.
        # gcc 9.4 can build avx512_vnni.
        # gcc 11.3 start to support avx2_vnni.
        self.assertTrue(actural_isa_val <= expected_isa_val)
        return    

    def test_dyndisp_in_supported_set(self):
        env_isa = get_ipex_isa_env_setting()

        if (env_isa != None):
            return

        cur_isa = get_currnet_isa_level()
        expected_isa = cur_isa in supported_isa_set

        self.assertTrue(expected_isa)
        return        

    @unittest.skipIf(check_not_sync_onednn_isa_level(), 'skip this if not sync onednn isa level')
    def test_ipex_set_onednn_isa_level(self):
        command = 'ATEN_CPU_CAPABILITY=avx2 python -c "import torch; import intel_extension_for_pytorch._C as core; print(core._get_current_onednn_isa_level())" '
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
          out = p.stdout.readlines()
          onednn_isa_level = str(out[-1], 'utf-8').strip()
          self.assertTrue(onednn_isa_level == 'AVX2')

    @unittest.skipIf(check_not_sync_onednn_isa_level(), 'skip this if not sync onednn isa level')
    def test_onednn_do_not_set_isa_level(self):
        command = 'ONEDNN_MAX_CPU_ISA=avx2 python -c "import torch; import intel_extension_for_pytorch._C as core; print(core._get_current_isa_level().lower())" '
        cur_ipex_isa = get_currnet_isa_level()
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
          out = p.stdout.readlines()
          cur_ipex_isa_1 = str(out[-1], 'utf-8').strip()
          self.assertTrue(cur_ipex_isa == cur_ipex_isa_1)

if __name__ == '__main__':
    unittest.main()
