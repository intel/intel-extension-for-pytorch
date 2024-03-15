import copy
import os
from pathlib import Path
import ast
import re
import yaml
from schema import Schema, And, Use, Optional, Or, Hook
from .dotdict import DotDict
from ..strategy import STRATEGIES
from intel_extension_for_pytorch.cpu.launch import CPUPoolList

# ### tuning ####
tuning_default = {"strategy": "grid", "max_trials": 100}


def _valid_strategy(data):
    data = data.lower()
    assert data in STRATEGIES, f"Tuning strategy {data} is NOT supported"
    return data


tuning_schema = Schema(
    {
        Optional("strategy", default="grid"): And(str, Use(_valid_strategy)),
        Optional("max_trials", default=100): int,
    }
)

# ### output_dir ###
output_dir_default = os.getcwd() + "/"
output_dir_schema = Schema(str)

# ### objective ###
objective_schema = Schema(
    {
        "name": str,
        Optional("higher_is_better", default=False): bool,
        Optional("target_val", default=-float("inf")): And(Or(int, float)),
    }
)

# ### hyperparams ###
# ### launcher ###

# default values if not tuning
launcher_hyperparam_default_val = {
    "ncore_per_instance": [-1],
    "ncores_per_instance": [-1],
    "ninstances": [-1],
    "use_all_nodes": [True],
    "use_logical_core": [False],
    "use_logical_cores": [False],
    "disable_numactl": [False],
    "disable_iomp": [False],
    "malloc": ["tc"],
}

# default search spaces if not user-specified
cpuinfo = CPUPoolList().pool_all
is_hyperthreading_enabled = len([c for c in cpuinfo if not c.is_physical_core]) > 0

launcher_hyperparam_default_search_space = {
    "hp": [
        "ncore_per_instance",
        "ncores_per_instance",
        "ninstances",
        "use_all_nodes",
        "use_logical_core",
        "use_logical_cores",
        "disable_numactl",
        "disable_iomp",
        "malloc",
    ],
    "ncore_per_instance": "all_logical_cores",
    "ncores_per_instance": "all_logical_cores",
    "ninstances": "all_logical_cores",
    "use_all_nodes": [True, False],
    "use_logical_core": [True, False],
    "use_logical_cores": [True, False],
    "disable_numactl": [True, False],
    "disable_iomp": [True, False],
    "malloc": ["pt", "tc", "je"],
}


def _valid_launcher_schema(key, scope, error):
    if isinstance(scope[key], str):
        assert scope[key] == "all_physical_cores" or scope[key] == "all_logical_cores"


def input_str_to_list_int(data):
    if isinstance(data, str):
        if data == "all_physical_cores":
            return [c.cpu + 1 for c in cpuinfo if c.is_physical_core]
        elif data == "all_logical_cores":
            return [c.cpu + 1 for c in cpuinfo]

    assert isinstance(data, list)
    return data


launcher_schema = Schema(
    {
        "hp": And(list, lambda s: all(isinstance(i, str) for i in s)),
        Hook("ncore_per_instance", handler=_valid_launcher_schema): object,
        Optional("ncore_per_instance", default="all_logical_cores"): And(
            Or(str, list),
            Use(input_str_to_list_int),
            lambda s: all(isinstance(i, int) for i in s),
        ),
        Hook("ncores_per_instance", handler=_valid_launcher_schema): object,
        Optional("ncores_per_instance", default="all_logical_cores"): And(
            Or(str, list),
            Use(input_str_to_list_int),
            lambda s: all(isinstance(i, int) for i in s),
        ),
        Hook("ninstances", handler=_valid_launcher_schema): object,
        Optional("ninstances", default="all_logical_cores"): And(
            Or(str, list),
            Use(input_str_to_list_int),
            lambda s: all(isinstance(i, int) for i in s),
        ),
        Optional(
            "use_all_nodes",
            default=(
                [True, False] if len(set([c.node for c in cpuinfo])) > 1 else [True]
            ),
        ): And(list, lambda s: all(isinstance(i, bool) for i in s)),
        Optional(
            "use_logical_core",
            default=[True, False] if is_hyperthreading_enabled else [False],
        ): And(list, lambda s: all(isinstance(i, bool) for i in s)),
        Optional(
            "use_logical_cores",
            default=[True, False] if is_hyperthreading_enabled else [False],
        ): And(list, lambda s: all(isinstance(i, bool) for i in s)),
        Optional("disable_numactl", default=[True, False]): And(
            list, lambda s: all(isinstance(i, bool) for i in s)
        ),
        Optional("disable_iomp", default=[True, False]): And(
            list, lambda s: all(isinstance(i, bool) for i in s)
        ),
        Optional("malloc", default=["pt", "tc", "je"]): And(
            list, lambda s: all(isinstance(i, str) for i in s)
        ),
    }
)

hyperparams_default = {"launcher": launcher_hyperparam_default_search_space}
hyperparams_schema = Schema(
    {
        Optional("launcher"): launcher_schema,
    }
)

schema = Schema(
    {
        # tuning
        Optional("tuning", default=tuning_default): tuning_schema,
        # hyperparams
        Optional("hyperparams", default=hyperparams_default): hyperparams_schema,
        # output_dir
        Optional("output_dir", default=output_dir_default): output_dir_schema,
    }
)


# reference: https://github.com/intel/neural-compressor/blob/15477100cef756\
#            e430c8ef8ef79729f0c80c8ce6/neural_compressor/conf/config.py
class Conf(object):
    def __init__(self, conf_fpath, program_fpath, program_args):
        assert Path(conf_fpath).exists(), f"{conf_fpath} does not exist"
        self.execution_conf = DotDict(
            schema.validate(
                self._convert_conf(
                    self._read_conf(conf_fpath), copy.deepcopy(schema.validate(dict()))
                )
            )
        )

        assert Path(program_fpath).exists(), f"{program_fpath} does not exist"
        self.program = program_fpath
        self.program_args = program_args
        self.usr_objectives = self._extract_usr_objectives(self.program)

    def _read_conf(self, conf_fpath):
        try:
            with open(conf_fpath, "r") as f:
                content = f.read()
                conf = yaml.safe_load(content)
                validated_conf = schema.validate(conf)
            return validated_conf

        except BaseException:
            raise RuntimeError(
                "The yaml file format is not correct. Please refer to document."
            )

    def _convert_conf(self, src, dst):
        hyperparam_default_val = {"launcher": launcher_hyperparam_default_val}

        for k in dst:
            if k == "hyperparams":
                dst_hps = set(dst["hyperparams"])
                for tune_x in dst_hps:
                    # case 1: tune {launcher}
                    if tune_x in src["hyperparams"]:
                        for hp in dst["hyperparams"][tune_x]["hp"]:
                            # case 1.1: not tune hp, use hp default val
                            if hp not in src["hyperparams"][tune_x]["hp"]:
                                dst["hyperparams"][tune_x][hp] = hyperparam_default_val[
                                    tune_x
                                ][hp]
                            # case 1.2: tune hp, use default or user defined search space
                            else:
                                dst["hyperparams"][tune_x][hp] = src["hyperparams"][
                                    tune_x
                                ][hp]
                    # case 2: not tune {launcher}
                    else:
                        del dst["hyperparams"][tune_x]

            elif k == "output_dir":
                if src[k] != dst[k]:
                    path = os.path.dirname(
                        src[k] if src[k].endswith("/") else src[k] + "/"
                    )
                    if not os.path.exists(path):
                        os.makedirs(path)
                    dst[k] = path

            else:
                dst[k] = src[k]
        return dst

    def _extract_usr_objectives(self, program_fpath):
        # e.g. [{'name': 'latency', 'higher_is_better': False, 'target_val': 0},
        #       {'name': 'throughput', 'higher_is_better':True, 'target_val': 100}]

        HYPERTUNE_TOKEN = "@hypertune"

        def _parse_hypertune_token(line):
            pattern = r'print\("@hypertune (.*?)"\)'
            lineseg = re.search(pattern, line)
            try:
                line = lineseg.group(1)
                objective = ast.literal_eval(line)
                objective = objective_schema.validate(objective)
            except BaseException:
                raise RuntimeError(
                    f"Parsing @hypertune failed for line {line} of {program_fpath} file"
                )
            return objective

        with Path(program_fpath).open("r") as f:
            text = f.read()
        lines = text.splitlines()

        return [_parse_hypertune_token(l) for l in lines if HYPERTUNE_TOKEN in l]
