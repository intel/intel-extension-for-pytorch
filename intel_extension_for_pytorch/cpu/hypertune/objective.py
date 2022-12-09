#reference: https://github.com/intel/neural-compressor/blob/15477100cef756e430c8ef8ef79729f0c80c8ce6/neural_compressor/objective.py
import subprocess
import sys  

class MultiObjective(object):
    def __init__(self, program, program_args, tune_launcher):
        self.program = program 
        self.program_args = program_args 
        self.tune_launcher = tune_launcher
            
    def evaluate(self, cfg):
        python = sys.executable
        cmd = [python]
        cmd.append("-m")
        cmd.append("intel_extension_for_pytorch.cpu.launch")
        
        if self.tune_launcher:
            launcher_args = self.decode_launcer_cfg(cfg)
            cmd += launcher_args 
        
        cmd += [self.program]
        cmd += self.program_args
        
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # todo: r.returncode != 0 
        
        output = str(r.stdout, "utf-8") 
        usr_objective_vals = self.extract_usr_objectives(output)
        return usr_objective_vals 
    
    def decode_launcer_cfg(self, cfg):
        ncore_per_instance = cfg["ncore_per_instance"]
        ninstances = cfg["ninstances"]
        use_all_nodes = cfg["use_all_nodes"]
        use_logical_core = cfg["use_logical_core"]
        disable_numactl = cfg["disable_numactl"]
        disable_iomp = cfg["disable_iomp"]
        malloc = cfg["malloc"]
            
        launcher_args = []
        
        if ncore_per_instance != -1:
            launcher_args.append("--ncore_per_instance")
            launcher_args.append(str(ncore_per_instance))
        
        if ninstances != -1:
            launcher_args.append("--ninstances")
            launcher_args.append(str(ninstances))
        
        if use_all_nodes == False:
            launcher_args.append("--node_id")
            launcher_args.append("0")
        
        if use_logical_core == True:
            launcher_args.append("--use_logical_core")

        if disable_numactl == True:
            launcher_args.append("--disable_numactl")

        if disable_iomp == True:
            launcher_args.append("--disable_iomp")

        if malloc == "tc":
            launcher_args.append("--enable_tcmalloc")
        elif malloc == "je":
            launcher_args.append("--enable_jemalloc")
        elif malloc == "default":
            launcher_args.append("--use_default_allocator")
            
        return launcher_args
    
    def extract_usr_objectives(self, output):
        HYPERTUNE_TOKEN = "@hypertune"
        output = output.strip().splitlines()
        
        objectives = []
        for i, s in enumerate(output):
            if HYPERTUNE_TOKEN in s:
                try:
                    objectives.append(float(output[i+1]))
                except:
                    raise RuntimeError("Extracting objective {} failed for {} file. Make sure to print an int/float value after the @hypertune token as the objective value to be minimized or maximized.".format(output[i], self.program))
        return objectives