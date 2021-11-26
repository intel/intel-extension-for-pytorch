import unittest
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
import time, sys
from intel_extension_for_pytorch.cpu.launch import *
import os
import glob

class TestLauncher(TestCase):
    def del_env(self, env_name):

        if env_name in os.environ:
            del os.environ[env_name]

    def find_lib(self, lib_type):
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(os.environ["CONDA_PREFIX"] + "/lib/")
        elif "VIRTUAL_ENV" in os.environ:
            library_paths.append(os.environ["VIRTUAL_ENV"] + "/lib/")

        library_paths += ["{}/.local/lib/".format(expanduser("~")), "/usr/local/lib/",
                         "/usr/local/lib64/", "/usr/lib/", "/usr/lib64/"]
        lib_find = False
        for lib_path in library_paths:
            library_file = lib_path + "lib" + lib_type + ".so"
            matches = glob.glob(library_file)
            if len(matches) > 0:
                lib_find = True
                break
        return lib_find

    def test_iomp_memory_allocator_setup(self):
       cpuinfo = CPUinfo()
       launcher = Launcher()
       self.del_env("OMP_NUM_THREADS")
       self.del_env("LD_PRELOAD")
       self.del_env("KMP_AFFINITY")
       self.del_env("KMP_BLOCKTIME")
       launcher.set_multi_thread_and_allocator(10, disable_iomp=False, enable_tcmalloc=True)
       find_iomp5 = self.find_lib("iomp5")
       find_tcmalloc = self.find_lib("tcmalloc")
       ld_preload_in_os = "LD_PRELOAD" in os.environ
       iomp5_enabled = "libiomp5.so" in os.environ["LD_PRELOAD"] if ld_preload_in_os else False
       tcmalloc_enabled = "libtcmalloc.so" in os.environ["LD_PRELOAD"] if ld_preload_in_os else False
       self.assertEqual(find_iomp5, iomp5_enabled)
       self.assertEqual(find_tcmalloc, tcmalloc_enabled)
       launcher.set_multi_thread_and_allocator(10, disable_iomp=False, enable_tcmalloc=False, enable_jemalloc=True)
       find_jemalloc = self.find_lib("jemalloc")
       jemalloc_enabled = "libjemalloc.so" in os.environ["LD_PRELOAD"] if ld_preload_in_os else False
       self.assertEqual(find_jemalloc, jemalloc_enabled)
       kmp_affinity_enabled = "KMP_AFFINITY" in os.environ and os.environ["KMP_AFFINITY"] == "granularity=fine,compact,1,0"
       block_time_enabled = "KMP_BLOCKTIME" in os.environ and os.environ["KMP_BLOCKTIME"] == "1"
       self.assertEqual(kmp_affinity_enabled, True)
       self.assertEqual(block_time_enabled, True)
       if jemalloc_enabled:
           self.assertEqual(jemalloc_enabled, "MALLOC_CONF" in os.environ)

    def test_mpi_pin_domain_and_ccl_worker_affinity(self):
       launcher = DistributedTrainingLauncher()
       total_cores = 56
       proc_per_node = 2
       ccl_worker_count = 4
       pin_doamin = launcher.get_mpi_pin_domain(proc_per_node, ccl_worker_count, total_cores)
       expect_pin_domain = "[0xffffff0,0xffffff00000000,]"
       self.assertEqual(pin_doamin, expect_pin_domain)
       ccl_worker_affinity = launcher.get_ccl_worker_affinity(proc_per_node, ccl_worker_count, total_cores)
       expect_ccl_worker_affinity = "0,1,2,3,28,29,30,31"
       self.assertEqual(ccl_worker_affinity, expect_ccl_worker_affinity)


if __name__ == '__main__':
    test = unittest.main()
