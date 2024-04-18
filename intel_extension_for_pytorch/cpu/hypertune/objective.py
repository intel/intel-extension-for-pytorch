# reference: https://github.com/intel/neural-compressor/blob/\
#            15477100cef756e430c8ef8ef79729f0c80c8ce6/neural_compressor/objective.py
import subprocess
from ...utils._logger import logger, WarningType


class MultiObjective(object):
    def __init__(self, program, program_args, tune_launcher):
        self.program = program
        self.program_args = program_args
        self.tune_launcher = tune_launcher

    def evaluate(self, cfg):
        cmd = ["ipexrun"]

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

    def deprecate_config(self, cfg, deprecated, new, default):
        v_deprecated = default
        v_new = default
        if deprecated in cfg.keys():
            v_deprecated = cfg[deprecated]
        if new in cfg.keys():
            v_new = cfg[new]
        assert (
            v_deprecated == default or v_new == default
        ), f"Configurations {deprecated} and {new} cannot be set at the same time."
        ret = default
        if v_deprecated != default:
            logger.warn(
                f"[**Warning**] Configuration {deprecated} is deprecated by {new}.",
                _type=WarningType.DeprecatedArgument,
            )
            ret = v_deprecated
        if v_new != default:
            ret = v_new
        return ret

    def decode_launcer_cfg(self, cfg):
        ncores_per_instance = self.deprecate_config(
            cfg, "ncore_per_instance", "ncores_per_instance", -1
        )
        ninstances = cfg["ninstances"]
        use_all_nodes = cfg["use_all_nodes"]
        use_logical_cores = self.deprecate_config(
            cfg, "use_logical_core", "use_logical_cores", False
        )
        disable_numactl = cfg["disable_numactl"]
        disable_iomp = cfg["disable_iomp"]
        malloc = cfg["malloc"]

        launcher_args = []

        if ncores_per_instance != -1:
            launcher_args.append("--ncores_per_instance")
            launcher_args.append(str(ncores_per_instance))

        if ninstances != -1:
            launcher_args.append("--ninstances")
            launcher_args.append(str(ninstances))

        if use_all_nodes is False:
            launcher_args.append("--nodes-list")
            launcher_args.append("0")

        if use_logical_cores is True:
            launcher_args.append("--use-logical-cores")

        if disable_numactl is True:
            launcher_args.append("--multi-task-manager")
            launcher_args.append("taskset")

        if disable_iomp is True:
            launcher_args.append("--omp-runtime")
            launcher_args.append("default")

        if malloc == "tc":
            launcher_args.append("--memory-allocator")
            launcher_args.append("tcmalloc")
        elif malloc == "je":
            launcher_args.append("--memory-allocator")
            launcher_args.append("jemalloc")
        elif malloc == "default":
            launcher_args.append("--memory-allocator")
            launcher_args.append("default")

        return launcher_args

    def extract_usr_objectives(self, output):
        HYPERTUNE_TOKEN = "@hypertune"
        output = output.strip().splitlines()

        objectives = []
        for i, s in enumerate(output):
            if HYPERTUNE_TOKEN in s:
                try:
                    objectives.append(float(output[i + 1]))
                except BaseException:
                    raise RuntimeError(
                        f"Extracting objective {output[i]} failed for {self.program} file. \
                            Make sure to print an int/float value after the @hypertune token as \
                            the objective value to be minimized or maximized."
                    )
        return objectives
