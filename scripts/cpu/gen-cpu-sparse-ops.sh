#!/bin/bash

cpu_op_path=$1
python "./gen-cpu-sparse-ops.py"                      \
  --output_folder="$cpu_op_path"                      \
  "./pytorch_headers/RegistrationDeclarations.h"      \
  "./pytorch_headers/Functions.h"                     \
  "./pytorch_headers/SparseCPUType.h"                 \
  "./sparse_spec/SparseAttrs.h"                       \
  "gen_check/SparseOPs.h"
