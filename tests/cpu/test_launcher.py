import unittest
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
import time, sys
from intel_extension_for_pytorch.cpu.launch import *
import os
import glob
import subprocess 

class TestLauncher(TestCase):
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

    def test_memory_allocator_setup(self):
       launcher = Launcher()

       # tcmalloc
       find_tcmalloc = self.find_lib("tcmalloc")
       cmd = ["python", "-m", "intel_extension_for_pytorch.cpu.launch", "--enable_tcmalloc", "--no_python", "ls"]
       r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
       tcmalloc_enabled = "libtcmalloc.so" in str(r.stdout, "utf-8").split("INFO - LD_PRELOAD=", 1)[1] 
       self.assertEqual(find_tcmalloc, tcmalloc_enabled)

       # jemalloc
       find_jemalloc = self.find_lib("jemalloc")
       cmd = ["python", "-m", "intel_extension_for_pytorch.cpu.launch", "--enable_jemalloc", "--no_python", "ls"]
       r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
       jemalloc_enabled = "libjemalloc.so" in str(r.stdout, "utf-8").split("INFO - LD_PRELOAD=", 1)[1] 
       self.assertEqual(find_jemalloc, jemalloc_enabled)
       if jemalloc_enabled:
           self.assertEqual(jemalloc_enabled, "MALLOC_CONF" in str(r.stdout, "utf-8"))

    def test_mpi_pin_domain_and_ccl_worker_affinity(self):
       launcher = DistributedTrainingLauncher()
       total_cores = 56
       proc_per_node = 2
       ccl_worker_count = 4
       pin_doamin = launcher.get_mpi_pin_domain(proc_per_node, ccl_worker_count, total_cores)
       expect_pin_domain = "[0xffffff0,0xffffff00000000,]"
       self.assertEqual(pin_doamin, expect_pin_domain)
       ccl_worker_affinity = launcher.get_ccl_worker_affinity(proc_per_node, ccl_worker_count, total_cores)
       expected_ccl_worker_affinity = "0,1,2,3,28,29,30,31"
       self.assertEqual(ccl_worker_affinity, expected_ccl_worker_affinity)
    
    def test_numactl_core_affinity(self):
        cpuinfo = CPUinfo()
        num_physical_cores = cpuinfo.physical_core_nums()
        
        launcher = MultiInstanceLauncher()
        numactl_available = launcher.is_numactl_available()
        
        if numactl_available:
            expected_core_affinity = "numactl -C {}-{}".format(str(0), str(num_physical_cores-1))
            cmd = ["python", "-m", "intel_extension_for_pytorch.cpu.launch", "--no_python", "ls"]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert r.returncode == 0
            assert expected_core_affinity in str(r.stdout, "utf-8")
    
    def test_taskset_core_affinity(self):
        cpuinfo = CPUinfo()
        num_physical_cores = cpuinfo.physical_core_nums()
    
        expected_core_affinity = "taskset -c {}-{}".format(str(0), str(num_physical_cores-1))
        cmd = ["python", "-m", "intel_extension_for_pytorch.cpu.launch", "--disable_numactl", "--no_python", "ls"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert r.returncode == 0
        assert expected_core_affinity in str(r.stdout, "utf-8")

    def test_core_affinity_with_skip_cross_node_cores(self):
        cpuinfo = CPUinfo()
        num_nodes = cpuinfo.node_nums()
        num_cores_per_node = len(cpuinfo.node_physical_cores[0])
        
        if num_nodes > 1:
            # ncore_per_instance that guarantees cross-node cores binding without --skip_cross_node_cores
            ncore_per_instance = num_cores_per_node -1
            
            cmd = "python -m intel_extension_for_pytorch.cpu.launch --ncore_per_instance {} --skip_cross_node_cores --no_python ls".format(ncore_per_instance)
            r = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert r.returncode == 0
            
            for i in range(num_nodes):
                node_i_start_core = i*num_cores_per_node
                expected_node_i_core_affinity = "-c {}-{}".format(str(node_i_start_core), str(node_i_start_core + ncore_per_instance -1))
                assert expected_node_i_core_affinity in str(r.stdout, "utf-8").lower()
    
    def test_core_affinity_with_skip_cross_node_cores_and_use_logical_core(self):
        cpuinfo = CPUinfo()
        num_nodes = cpuinfo.node_nums()
        num_cores_per_node = len(cpuinfo.node_physical_cores[0])
        num_threads_per_core = int(cpuinfo.logical_core_nums()/cpuinfo.physical_core_nums())

        if num_nodes > 1 and num_threads_per_core > 1:
            # ncore_per_instance that guarantees cross-node cores binding without --skip_cross_node_cores
            ncore_per_instance = num_cores_per_node -1
            
            cmd = "python -m intel_extension_for_pytorch.cpu.launch --ncore_per_instance {} --use_logical_core --skip_cross_node_cores --no_python ls".format(ncore_per_instance)
            r = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert r.returncode == 0
            
            for i in range(num_nodes):
                node_i_physical_start_core = i*num_cores_per_node
                node_i_logical_start_core = (i+num_nodes)*num_cores_per_node
                expected_node_i_physical_core_affinity = "-c {}-{}".format(str(node_i_physical_start_core), str(node_i_physical_start_core + ncore_per_instance -1))
                expected_node_i_logical_core_affinity = "-c {}-{}".format(str(node_i_logical_start_core), str(node_i_logical_start_core + ncore_per_instance -1))
                assert expected_node_i_physical_core_affinity in str(r.stdout, "utf-8").lower()
                assert expected_node_i_logical_core_affinity in str(r.stdout, "utf-8").lower()
    
    def test_core_affinity_with_skip_cross_node_cores_and_node_id_use_logical_core(self):
        cpuinfo = CPUinfo()
        num_nodes = cpuinfo.node_nums()
        num_cores_per_node = len(cpuinfo.node_physical_cores[0])
        num_threads_per_core = int(cpuinfo.logical_core_nums()/cpuinfo.physical_core_nums())
        
        if num_nodes > 1 and num_threads_per_core > 1:
            # ncore_per_instance that guarantees cross-node cores binding without --skip_cross_node_cores
            ncore_per_instance = num_cores_per_node -1
            
            cmd = "python -m intel_extension_for_pytorch.cpu.launch --ncore_per_instance {} --node_id 0 --use_logical_core --skip_cross_node_cores --no_python ls".format(ncore_per_instance)
            r = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert r.returncode == 0
            
            node_0_physical_start_core = 0*num_cores_per_node 
            node_0_logical_start_core = (0+num_nodes)*num_cores_per_node
            
            expected_node_0_physical_core_affinity = "-c {}-{}".format(str(node_0_physical_start_core), str(node_0_physical_start_core + ncore_per_instance -1))
            expected_node_0_logical_core_affinity = "-c {}-{}".format(str(node_0_logical_start_core), str(node_0_logical_start_core + ncore_per_instance -1))
            
            assert expected_node_0_physical_core_affinity in str(r.stdout, "utf-8").lower()
            assert expected_node_0_logical_core_affinity in str(r.stdout, "utf-8").lower()
    
    def test_skip_cross_node_cores_with_too_many_ncore_per_instance(self):
        cpuinfo = CPUinfo()
        num_nodes = cpuinfo.node_nums()
        num_cores_per_node = len(cpuinfo.node_physical_cores[0])
        
        if num_nodes > 1:
            # ncore_per_instance that is too many to skip cross-node cores 
            ncore_per_instance = num_cores_per_node +1
            
            expected_msg = "Please make sure --ncore_per_instance < core(s) per socket"
            
            cmd = "python -m intel_extension_for_pytorch.cpu.launch --ncore_per_instance {} --skip_cross_node_cores --no_python ls".format(ncore_per_instance)
            r = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert r.returncode != 0
            assert expected_msg in str(r.stdout, "utf-8")
    
    def test_skip_cross_node_cores_with_divisible_ncore_per_instance(self):
        cpuinfo = CPUinfo()
        num_nodes = cpuinfo.node_nums()
        num_cores_per_node = len(cpuinfo.node_physical_cores[0])
        
        if num_nodes > 1:
            # ncore_per_instance that guarantees no cross-node cores binding 
            ncore_per_instance = num_cores_per_node
            
            expected_msg = "--skip_cross_node_cores is set, but there are no cross-node cores"
            
            cmd = "python -m intel_extension_for_pytorch.cpu.launch --ncore_per_instance {} --skip_cross_node_cores --no_python ls".format(ncore_per_instance)
            r = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert r.returncode == 0
            assert expected_msg in str(r.stdout, "utf-8")

if __name__ == '__main__':
    test = unittest.main()
