import unittest
import os

import intel_extension_for_pytorch._C as core

def get_ipex_isa_env_setting():
    env_isa = os.getenv('ATEN_CPU_CAPABILITY')
    return env_isa

def get_currnet_isa_level():
    return core._get_current_isa_level().lower()

class TestDynDisp(unittest.TestCase):

    def test_env_setting(self):
        env_isa = get_ipex_isa_env_setting()
        cur_isa = get_currnet_isa_level()

        if (env_isa == None):
            return

        self.assertEqual(env_isa.lower(), cur_isa.lower())

    def test_dyndisp_in_supported_set(self):
        env_isa = get_ipex_isa_env_setting()

        if (env_isa != None):
            return

        cur_isa = get_currnet_isa_level()
        supported_isa_set = ["default", "avx2", "avx512"]

        expected_isa = cur_isa in supported_isa_set

        self.assertTrue(expected_isa)

if __name__ == '__main__':
    unittest.main()
