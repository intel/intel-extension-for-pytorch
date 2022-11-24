HyperTune (Experimental)
========================

![HyperTune](../../../images/hypertune/hypertune.png)

HyperTune is an experimental feature to perform hyperparameter/execution configuration searching. The searching is used in various areas such as optimization of hyperparameters of deep learning models. The searching is extremely useful in real situations when the number of hyperparameters, including configuration of script execution, and their search spaces are huge that manually tuning these hyperparameters/configuration is impractical and time consuming. Hypertune automates this process of execution configuration searching for the [launcher](../performance_tuning/launch_script.md) and Intel® Extension for PyTorch\*.

## Usage of Hypertune
```
python -m intel_extension_for_pytorch.cpu.hypertune --conf_file <your_conf_file> <your_python_script> [args]
```

There are two things to provide Hypertune (1) `<your_conf_file>` .yaml file to define the hyperparameters and their search spaces (2) `<your_python_script>` as an optimization function.

### `your_conf_file`
The .yaml file is used to define configuration of Hypertune. There are two main information needed: (1) hyperparameters to tune and their search spaces (2) tuning strategy. See comments below together with a sample .yaml file.

```
tuning:                                                        # optional.
  strategy: grid                                               # optional. The tuning strategy. Default is grid. Must be one of {grid, random}.
  max_trials: 100                                              # optional. Allowed number of trials. Default is 100. If given time, set max_trials to product of length of all search spaces to try all possible combinations of hyperparameters.

output_dir: /path/to/saving/directory                          # optional. Directory to which the tuning history will be saved in record.csv file. Default is current working directory.

hyperparams:                                                   # mandatory.
  launcher:                                                    # optional.
    hp: ['ncore_per_instance', 'ninstances']                   # mandatory. Mandatory if hyperparams.launcher is specified. Specify the launcher hyperparameters to tune.
    ncore_per_instance: all_physical_cores                     # optional.  Search space of ncore_per_instance if chosen to tune. If not defined, default search space of ncore_per_instance is used.
    ninstances:  [1]                                           # optional.  Search space of ninstances if chosen to tune. If not defined, default search space of ninstances is used.
```

### Hyperparameters
#### Launcher Hyperparameters
Currently hypertune tunes for the following launcher hyperparameters:

| hyperparameter | default value | default search space | search space format |
| :-- | :--: | :--: | :--: |
| ```ncore_per_instance``` | -1 | `all_logical_cores` | `str or list of int. str must be one of {'all_logical_cores', 'all_physical_cores'}` |
| ```ninstances``` | -1 | `all_logical_cores` | `str or list of int. str must be one of {'all_logical_cores', 'all_physical_cores'}` |
| ```use_all_nodes``` | True | `[True, False] if num_nodes > 1 else [True]` | `list of bool` |
| ```use_logical_core``` | False | `[True, False] if is_hyperthreading_enabled else [False]` | `list of bool` |
| ```disable_numactl``` | False | `[True, False]` | `list of bool` |
| ```disable_iomp``` | False | `[True, False]` | `list of bool` |
| ```malloc``` | tc | `['tc', 'je', 'pt']` | `list of str. str must be in {'tc', 'je', 'pt'}` |

### Defining hyperparameters and their search spaces
#### 1. Defining hyperparameters to tune:

List the hyperparameters to tune in `hp`. For example, to tune all launcher hyperparameters:
```
hyperparams:
  launcher:
    hp: ['ncore_per_instance', 'ninstances', 'use_all_nodes', 'use_logical_core', 'disable_numactl', 'disable_iomp', 'malloc']
```

For example, to tune only launcher `ncore_per_instance`:
```
hyperparams:
  launcher:
    hp: ['ncore_per_instance']                    .
```
All other launcher hyperparameters (`ninstances`, `use_all_nodes`, `use_logical_core`, `disable_numactl`, `disable_iomp`, `malloc`) will not be tuned and instead will use the default value defined in the previous section.

#### 2. Defining the search spaces of the hyperparameters:

#### Default search space

If you don't specify the search space of a hyperparamter, then the default search space defined in the previous section will be used for the hyperparameters defined in `hp`. For example,
```
hyperparams:
  launcher:
    hp: ['malloc']                    .
```
`malloc` will be tuned using its default search space, `['tc', 'je', 'pt']`. All other launcher hyperparamters (`ncore_per_instance`, `ninstances`, `use_all_nodes`, `use_logical_core`, `disable_numactl`, `disable_iomp`) will not be tuned and instead will use their default values.

#### User defined search space

Specify the search space of a hyperparameter. For example,
```
hyperparams:
  launcher:
    hp: ['ncore_per_instance', 'ninstances', 'malloc']
    ninstances: [1]
    ncore_per_instance: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
`ninstances` and `ncore_per_instance` will use user defined spaces `[1]` and `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` respectively. `malloc` will use its default search space, `['tc', 'je', 'pt']`.

### `<your_python_script>`
This is the script as an optimization function.
- Step 1. Print the objective(s) you want to optimize. Make sure this is just an int or float to be minimized or maximized.
- Step 2. Just before the objective(s), add print statement(s) of the `@hypertune {'name': str, 'higher_is_better': bool, 'target_val': int or float}`.
```
'name'                                     # mandatory. The name of your objective function.
'higher_is_better'                         # optional. True if objective function is to be maximized, False if to be minimized. Default is False.
'target_val'                               # optional. Target value of the objective function. Default is -float('inf')
```

Have a look at the [example script](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.0/intel_extension_for_pytorch/cpu/hypertune/example/resnet50.py).

## Usage Examples

**Tuning `ncore_per_instance` for minimum `latency`**

Suppose we want to tune `ncore_per_instance` for a single instance to minimize latency for resnet50 on a machine with two Intel(R) Xeon(R) Platinum 8180M CPUs. Each socket has 28 physical cores and another 28 logical cores.

Run the following command with [example.yaml](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.0/intel_extension_for_pytorch/cpu/example/example.yaml) and [resnet50.py](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.0/intel_extension_for_pytorch/cpu/hypertune/example/resnet50.py):
```
python -m intel_extension_for_pytorch.cpu.hypertune --conf_file <hypertune_directory>/example/example.yaml <hypertune_directory>/example/resnet50.py
```

Once search completes, it will print to terminal the best tune result and best tune configuration found. Below is an output for this example:
```
Best configuration found is: {'ncore_per_instance': 15, 'ninstances': 1, 'use_all_nodes': True, 'use_logical_core': False, 'disable_numactl': False, 'disable_iomp': False, 'malloc': 'tc'}
latency: 12.339081764221191
```
15 `ncore_per_instance` gave the minimum latency.

You will also find the tuning history in `<output_dir>/record.csv`. You can take [a sample csv file](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.0/intel_extension_for_pytorch/cpu/hypertune/example/record.csv) as a reference.

Hypertune can also optimize multi-objective function. Add as many objectives as you would like to your script.
