import unittest
import os

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

if __name__ == '__main__':
    unittest.main()
