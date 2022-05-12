# Get Operator Registrations for All Kinds of Backends

> This script is for get operators registrated in all kinds of backends (Pytorch 1.10+)

## Run python script to get operators in other backends
query_op.py provides the script to get the operators registration information, which output a csv file.

1. Query Ops in CPU or CUDA:

```bash
# This command output a single file of CPU or CUDA ops
python query_op.py --key CPU --output cpu_ops.csv
python query_op.py --key CUDA --output cuda_ops.csv
# since CPU and CUDA queries are one time effort, so also add the above two generated files in git repo
```

2. Query Ops in XPU:

```bash
# This command output a merged file with CPU, CUDA and XPU ops
python query_op.py --key XPU --cpu_queried_file cpu_ops.csv --cuda_queried_file cuda_ops.csv --output xpu_ops.csv
# xpu_ops.csv will contain all what we need (cpu, cuda, xpu)
```