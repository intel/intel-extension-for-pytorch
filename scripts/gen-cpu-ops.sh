#!/bin/bash

cpu_op_path=$1
python "./gen-cpu-ops.py" \
  --output_folder="$cpu_op_path" \
  "$cpu_op_path/OPs.h" \
  "./RegistrationDeclarations.h" \
  "./Functions.h"                \
  "./SparseCPUType.h"            \
  "./SparseAttrs.h"
