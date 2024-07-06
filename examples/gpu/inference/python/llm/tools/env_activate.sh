#!/bin/bash

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh)
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_ROOT=${CONDA_PREFIX}
export TORCH_LLM_ALLREDUCE=1
