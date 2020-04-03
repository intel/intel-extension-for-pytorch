#!/bin/bash

cpu_op_path=$1

python "./gen-cpu-sparse-dispatch.py"                 \
  --output_folder="$cpu_op_path"                      \
  "./pytorch_headers/RegistrationDeclarations.h"      \
  "./pytorch_headers/SparseCPUType.h"                 \
  "./gen-check/aten_ipex_sparse_type_default.h"
