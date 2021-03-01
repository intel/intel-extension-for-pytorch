#!/bin/bash

cpu_op_path=$1
pytorch_dir=$2
sparse_ops_dec_dir=$3
python "./gen-sparse-cpu-ops.py" \
  "$cpu_op_path/SparseOPs.h" \
  "$cpu_op_path/SparseOPs.cpp" \
  "$pytorch_dir/include/ATen/RegistrationDeclarations.h" \
  "$pytorch_dir/include/ATen/Functions.h" \
  "$sparse_ops_dec_dir/SparseCPUType.h" \
  "./sparse_spec/SparseAttrs.h"
