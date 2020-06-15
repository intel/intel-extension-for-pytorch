#!/bin/bash

cpu_op_path=$1
pytorch_dir=$2
python "./gen-dense-cpu-ops.py" \
  "$cpu_op_path/DenseOPs.h" \
  "$cpu_op_path/DenseOPs.cpp" \
  "$pytorch_dir/include/torch/csrc/autograd/generated/RegistrationDeclarations.h"      \
  "$pytorch_dir/include/ATen/Functions.h"
