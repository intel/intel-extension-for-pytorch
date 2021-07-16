# Overview

As introduced in [Performance Tuning Guide](Performance_Tuning.md), there are several factors that influence performance very much. Setting those configurations properly contributes to performance boost. However, there is no unified configuration that is optimal to all topologies. Users need to try different combinations by themselves. A *launch* script is provided to automate these configuration settings to free users from this complicated work. This guide helps you to learn some most frequent usage examples.

The configurations are mainly around the following perspectives. Italic values are default if applicable.
1. OpenMP library: [*Intel OpenMP library* | GNU OpenMP library]
2. Memory allocator: [PyTorch default memory allocator | Jemalloc | *TCMalloc*]
3. Number of instances: [*Single instance* | Multiple instances]

# Usage of launch script

The *launch* script is provided as a module of torch_ipex. You can take advantage of it with the following command:
```
python -m torch_ipex.launch [knobs] <your_pytorch_script> [args]
```

Available knobs are listed below:

| knob | type | default value | help |
| :-- | :--: | :--: | :-- |
| ```-m```, ```--module``` | - | False | Changes each process to interpret the launch script  as a python module, executing with the same behavior as 'python -m'. |
| ```--no_python``` | - | False | Do not prepend the --program script with \python\ - just exec  it directly. Useful when the script is not a Python script. |
| ```--ninstances``` | int | -1 | For multi-instance, you should give the cores number you used for per instance. |
| ```--ncore_per_instance``` | int | -1 | Cores per instance |
| ```--latency_mode``` | - | False | By default 4 core per instance and use all physical cores |
| ```--throughput_mode``` | - | False | By default one instance per socket and use all physical cores |
| ```--socket_id``` | int | -1 | Socket id for multi-instance, by default all sockets will be used |
| ```--use_logical_core``` | - | False | Whether only use physical cores |
| ```--disable_numactl``` | - | False | Disable numactl |
| ```--core_list``` | str | None | Specify the core list as 'core_id, core_id, ....', otherwise, all the cores will be used. |
| ```--log_path``` | str | './logs' | The log file path, default path is './logs/' |
| ```--log_file_prefix``` | str | 'run' | log file prefix |
| ```--disable_iomp``` | - | False | By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD |
| ```--enable_tcmalloc``` | - | False | Enable tcmalloc allocator |
| ```--enable_jemalloc``` | - | False | Enable jemalloc allocator |
| ```--use_default_allocator``` | - |  False | Use default memory allocator |

The *launch* script respects existing environment variables when it get launched, expect for *LD_PRELOAD*. If you have your favorite values for certain environment variables, you can set them before running the *launch* script. A typical usage scenario is as the following. Intel OpenMP library uses an environment variable *KMP_AFFINITY* to control its behavior. Different settings result in different performance numbers. By default, if you enable Intel OpenMP library, the *launch* script will set *KMP_AFFINITY* to "granularity=fine,compact,1,0". If you want to try with other values, you can use *export* command on Linux to set *KMP_AFFINITY* before you run the *launch* script. In this case, the script will not set the default value but take the existing value of *KMP_AFFINITY*, and print a message to stdout.

Execution via the *launch* script dumps logs into files under a designated log directory. By default, it is ```log``` under your current directory. 2 types of files will be generated. One file (```<prefix>_timestamp_instances.log```) contains command and information when the script was launched. Another type of files (```<prefix>_timestamp_instance_N_cores_....log```) contain stdout print of each instance.

E.g.
```
run_20210712212258_instances.log
run_20210712212258_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

# Usage Examples

Example script [launch_example.py](../examples/launch_example.py) will be used in this guide.

- Single instance for inference
  - [I. Use all physical cores](#i-use-all-physical-cores)
  - [II. Use all cores including logical cores](#ii-use-all-cores-including-logical-cores)
  - [III. Use physical cores on 1 socket](#iii-use-physical-cores-on-1-socket)
  - [IV. Use your designated number of cores](#iv-use-your-designated-number-of-cores)
- Multiple instances for inference
  - [V. Throughput mode (i.e. number of socket instances, each instance runs on 1 socket)](#v-throughput-mode)
  - [VI. Latency mode (Use 4 cores for each instance)](#vi-latency-mode)
  - [VII. Your designated number of instances](#vii-your-designated-number-of-instances)
- Usage of Jemalloc/TCMalloc/Default memory allocator
  - [Jemalloc](#jemalloc)
  - [TCMalloc](#tcmalloc)
  - [Default memory allocator](#default-memory-allocator)
- Usage of GNU OpenMP library
  - [Intel OpenMP library](#intel-openmp-library)
  - [GNU OpenMP library](#gnu-openmp-library)

__Note:__ GIF files below intend to show CPU usage ONLY. Please do NOT induct performance numbers.

## Single instance for inference

### I. Use all physical cores

```
python -m torch_ipex.launch launch_example.py
```

CPU usage is shown as below. 1 main worker thread was launched, then it launched physical core number of threads on all physical cores.

![Single instance all physical cores](../images/1ins_phy.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
    ├── run_20210712212258_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
    └── run_20210712212258_instances.log
```

The ```run_20210712212258_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712212258_instances.log
2021-07-12 21:22:58,764 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 21:22:58,764 - __main__ - INFO - OMP_NUM_THREADS=44
2021-07-12 21:22:58,764 - __main__ - INFO - Using Intel OpenMP
2021-07-12 21:22:58,764 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 21:22:58,764 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 21:22:58,764 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 21:22:58,764 - __main__ - WARNING - Numa Aware: cores:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'] in different NUMA node
2021-07-12 21:22:58,764 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712212258_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

### II. Use all cores including logical cores

```
python -m torch_ipex.launch --use_logical_core launch_example.py
```

CPU usage is shown as below. 1 main worker thread was launched, then it launched threads on all cores, including logical cores.

![Single instance logical cores](../images/1ins_log.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
    ├── run_20210712223308_instances.log
    └── run_20210712223308_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_44_45_46_47_48_49_50_51_52_53_54_55_56_57_58_59_60_61_62_63_64_65_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_66_67_68_69_70_71_72_73_74_75_76_77_78_79_80_81_82_83_84_85_86_87.log
```

The ```run_20210712223308_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712223308_instances.log
2021-07-12 22:33:08,117 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 22:33:08,117 - __main__ - INFO - OMP_NUM_THREADS=88
2021-07-12 22:33:08,117 - __main__ - INFO - Using Intel OpenMP
2021-07-12 22:33:08,118 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 22:33:08,118 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 22:33:08,118 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 22:33:08,118 - __main__ - WARNING - Numa Aware: cores:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87'] in different NUMA node
2021-07-12 22:33:08,118 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712223308_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_44_45_46_47_48_49_50_51_52_53_54_55_56_57_58_59_60_61_62_63_64_65_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_66_67_68_69_70_71_72_73_74_75_76_77_78_79_80_81_82_83_84_85_86_87.log
```

__Note:__ A known issue is that if there are too many cores used, name of the log file will exceed length limit and thus not able to generate the log file. It will be fixed later showing range of cores, rather than enumerate all cores out.

### III. Use physical cores on 1 socket

```
python -m torch_ipex.launch --socket_id 1 launch_example.py
```

CPU usage is shown as below. 1 main worker thread was launched, then it launched threads on all other cores on the same socket.

![Single instance all physical cores](../images/1ins_soc.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
	├── run_20210712214504_instances.log
    └── run_20210712214504_instance_0_cores_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
    
```

The ```run_20210712214504_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712214504_instances.log
2021-07-12 21:45:04,512 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 21:45:04,513 - __main__ - INFO - OMP_NUM_THREADS=22
2021-07-12 21:45:04,513 - __main__ - INFO - Using Intel OpenMP
2021-07-12 21:45:04,513 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 21:45:04,513 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 21:45:04,513 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 21:45:04,513 - __main__ - INFO - numactl -C 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712214504_instance_0_cores_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

### IV. Use your designated number of cores

```
python -m torch_ipex.launch --ninstances 1 --ncore_per_instance 10 launch_example.py
```

CPU usage is shown as below. 1 main worker thread was launched, then it launched threads on other 9 physical cores.

![Single instance designated number of cores](../images/1ins_cus.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
	├── run_20210712220928_instances.log
    └── run_20210712220928_instance_0_cores_0_1_2_3_4_5_6_7_8_9.log
```

The ```run_20210712220928_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712220928_instances.log
2021-07-12 22:09:28,355 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 22:09:28,355 - __main__ - INFO - OMP_NUM_THREADS=10
2021-07-12 22:09:28,355 - __main__ - INFO - Using Intel OpenMP
2021-07-12 22:09:28,355 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 22:09:28,356 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 22:09:28,356 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 22:09:28,356 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712220928_instance_0_cores_0_1_2_3_4_5_6_7_8_9.log
```

## Multiple instances for inference

### V. Throughput mode

```
python -m torch_ipex.launch --throughput_mode launch_example.py
```

CPU usage is shown as below. 2 main worker threads were launched on 2 socket respectively, then they launched threads on other physical cores.

![Multiple instance throughput mode](../images/nins_thr.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
	├── run_20210712221150_instances.log
    ├── run_20210712221150_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21.log
	└── run_20210712221150_instance_1_cores_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

The ```run_20210712221150_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712221150_instances.log
2021-07-12 22:11:50,233 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 22:11:50,233 - __main__ - INFO - OMP_NUM_THREADS=22
2021-07-12 22:11:50,233 - __main__ - INFO - Using Intel OpenMP
2021-07-12 22:11:50,233 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 22:11:50,233 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 22:11:50,233 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 22:11:50,233 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221150_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21.log
2021-07-12 22:11:50,236 - __main__ - INFO - numactl -C 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221150_instance_1_cores_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

### VI. Latency mode

```
python -m torch_ipex.launch --latency_mode launch_example.py
```

CPU usage is shown as below. 4 cores are used for each instance.

![Multiple instances latency mode](../images/nins_lat.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
	├── run_20210712221415_instances.log
    ├── run_20210712221415_instance_0_cores_0_1_2_3.log
	├── run_20210712221415_instance_1_cores_4_5_6_7.log
	├── run_20210712221415_instance_2_cores_8_9_10_11.log
	├── run_20210712221415_instance_3_cores_12_13_14_15.log
	├── run_20210712221415_instance_4_cores_16_17_18_19.log
	├── run_20210712221415_instance_5_cores_20_21_22_23.log
	├── run_20210712221415_instance_6_cores_24_25_26_27.log
	├── run_20210712221415_instance_7_cores_28_29_30_31.log
	├── run_20210712221415_instance_8_cores_32_33_34_35.log
	├── run_20210712221415_instance_9_cores_36_37_38_39.log
	└── run_20210712221415_instance_10_cores_40_41_42_43.log
```

The ```run_20210712221415_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712221415_instances.log
2021-07-12 22:14:15,140 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 22:14:15,140 - __main__ - INFO - OMP_NUM_THREADS=4
2021-07-12 22:14:15,140 - __main__ - INFO - Using Intel OpenMP
2021-07-12 22:14:15,140 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 22:14:15,140 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 22:14:15,140 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 22:14:15,140 - __main__ - INFO - numactl -C 0,1,2,3 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_0_cores_0_1_2_3.log
2021-07-12 22:14:15,143 - __main__ - INFO - numactl -C 4,5,6,7 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_1_cores_4_5_6_7.log
2021-07-12 22:14:15,146 - __main__ - INFO - numactl -C 8,9,10,11 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_2_cores_8_9_10_11.log
2021-07-12 22:14:15,149 - __main__ - INFO - numactl -C 12,13,14,15 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_3_cores_12_13_14_15.log
2021-07-12 22:14:15,151 - __main__ - INFO - numactl -C 16,17,18,19 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_4_cores_16_17_18_19.log
2021-07-12 22:14:15,154 - __main__ - WARNING - Numa Aware: cores:['20', '21', '22', '23'] in different NUMA node
2021-07-12 22:14:15,154 - __main__ - INFO - numactl -C 20,21,22,23 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_5_cores_20_21_22_23.log
2021-07-12 22:14:15,157 - __main__ - INFO - numactl -C 24,25,26,27 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_6_cores_24_25_26_27.log
2021-07-12 22:14:15,159 - __main__ - INFO - numactl -C 28,29,30,31 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_7_cores_28_29_30_31.log
2021-07-12 22:14:15,162 - __main__ - INFO - numactl -C 32,33,34,35 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_8_cores_32_33_34_35.log
2021-07-12 22:14:15,164 - __main__ - INFO - numactl -C 36,37,38,39 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_9_cores_36_37_38_39.log
2021-07-12 22:14:15,167 - __main__ - INFO - numactl -C 40,41,42,43 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221415_instance_10_cores_40_41_42_43.log
```

### VII. Your designated number of instances

```
python -m torch_ipex.launch --ninstances 4 launch_example.py
```

CPU usage is shown as below. 4 main worker thread were launched, then they launched threads on all other physical cores.

![Multiple instances designated number of instances](../images/nins_cus.gif)

If you check your log directory, you will find directory structure as below.

```
.
├── launch_example.py
└── logs
	├── run_20210712221305_instances.log
    ├── run_20210712221305_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10.log
	├── run_20210712221305_instance_1_cores_11_12_13_14_15_16_17_18_19_20_21.log
	├── run_20210712221305_instance_2_cores_22_23_24_25_26_27_28_29_30_31_32.log
	└── run_20210712221305_instance_3_cores_33_34_35_36_37_38_39_40_41_42_43.log
```

The ```run_20210712221305_instances.log``` contains information and command that were used for this execution launch.
```
$ cat logs/run_20210712221305_instances.log
2021-07-12 22:13:05,470 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-12 22:13:05,470 - __main__ - INFO - OMP_NUM_THREADS=11
2021-07-12 22:13:05,470 - __main__ - INFO - Using Intel OpenMP
2021-07-12 22:13:05,470 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-12 22:13:05,470 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-12 22:13:05,470 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-12 22:13:05,471 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221305_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10.log
2021-07-12 22:13:05,473 - __main__ - INFO - numactl -C 11,12,13,14,15,16,17,18,19,20,21 -m 0 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221305_instance_1_cores_11_12_13_14_15_16_17_18_19_20_21.log
2021-07-12 22:13:05,476 - __main__ - INFO - numactl -C 22,23,24,25,26,27,28,29,30,31,32 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221305_instance_2_cores_22_23_24_25_26_27_28_29_30_31_32.log
2021-07-12 22:13:05,479 - __main__ - INFO - numactl -C 33,34,35,36,37,38,39,40,41,42,43 -m 1 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210712221305_instance_3_cores_33_34_35_36_37_38_39_40_41_42_43.log
```

## Usage of Jemalloc/TCMalloc/Default memory allocator

Memory allocator influences performance sometime. If users do not designate desired memory allocator, the *launch* script searches them in the order of TCMalloc > Jemalloc > PyTorch default memory allocator, and takes the first matched one.

### Jemalloc

__Note:__ You can set your favorite value to *MALLOC_CONF* before running the *launch* script if you do not want to use its default setting.

```
python -m torch_ipex.launch --enable_jemalloc launch_example.py
```

you can confirm usage in log file:

```
2021-07-13 15:30:48,235 - __main__ - INFO - Use JeMallocl memory allocator
2021-07-13 15:30:48,235 - __main__ - INFO - MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000
2021-07-13 15:30:48,235 - __main__ - INFO - OMP_NUM_THREADS=44
2021-07-13 15:30:48,235 - __main__ - INFO - Using Intel OpenMP
2021-07-13 15:30:48,235 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-13 15:30:48,235 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-13 15:30:48,235 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so:<VIRTUAL_ENV>/lib/libjemalloc.so
2021-07-13 15:30:48,236 - __main__ - WARNING - Numa Aware: cores:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'] in different NUMA node
2021-07-13 15:30:48,236 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210713153048_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

### TCMalloc

```
python -m torch_ipex.launch --enable_tcmalloc launch_example.py
```

you can confirm usage in log file:

```
2021-07-13 15:33:33,654 - __main__ - INFO - Use TCMalloc memory allocator
2021-07-13 15:33:33,654 - __main__ - INFO - OMP_NUM_THREADS=44
2021-07-13 15:33:33,654 - __main__ - INFO - Using Intel OpenMP
2021-07-13 15:33:33,654 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-13 15:33:33,654 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-13 15:33:33,654 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so:<VIRTUAL_ENV>/lib/libtcmalloc.so
2021-07-13 15:33:33,654 - __main__ - WARNING - Numa Aware: cores:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'] in different NUMA node
2021-07-13 15:33:33,655 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210713153333_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

### Default memory allocator

```
python -m torch_ipex.launch --use_default_allocator launch_example.py
```

you can confirm usage in log file:

```
2021-07-13 15:36:59,784 - __main__ - INFO - OMP_NUM_THREADS=44
2021-07-13 15:36:59,784 - __main__ - INFO - Using Intel OpenMP
2021-07-13 15:36:59,784 - __main__ - INFO - KMP_AFFINITY=granularity=fine,compact,1,0
2021-07-13 15:36:59,784 - __main__ - INFO - KMP_BLOCKTIME=1
2021-07-13 15:36:59,784 - __main__ - INFO - LD_PRELOAD=<VIRTUAL_ENV>/lib/libiomp5.so
2021-07-13 15:36:59,784 - __main__ - WARNING - Numa Aware: cores:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'] in different NUMA node
2021-07-13 15:36:59,784 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210713153659_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```

## Usage of OpenMP library

### Intel OpenMP Library

Generally, Intel OpenMP library brings better performance. Thus, in the *launch* script, Intel OpenMP library is used by default, if it is available. Intel OpenMP library takes environment variables like *KMP_AFFINITY* and *KMP_BLOCKTIME* to control its behavior. You can set your favorite values to them before running the *launch* script if you do not want to use the default settings.

### GNU OpenMP Library

It is, however, not always that Intel OpenMP library brings better performance comparing to GNU OpenMP library. In this case, you can use knob ```--disable_iomp``` to switch active OpenMP library to the GNU one.

```
python -m torch_ipex.launch --disable_iomp launch_example.py
```

you can confirm usage in log file:

```
2021-07-13 15:25:00,760 - __main__ - WARNING - Both TCMalloc and JeMalloc are not found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/<user>/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance
2021-07-13 15:25:00,761 - __main__ - INFO - OMP_NUM_THREADS=44
2021-07-13 15:25:00,761 - __main__ - WARNING - Numa Aware: cores:['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'] in different NUMA node
2021-07-13 15:25:00,761 - __main__ - INFO - numactl -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 <VIRTUAL_ENV>/bin/python launch_example.py 2>&1 | tee ./logs/run_20210713152500_instance_0_cores_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43.log
```
