#!/bin/bash

python "./gen-cpu-ops.py" \
  --output_folder="./" \
  "./OPs.h" \
  "./RegistrationDeclarations.h" \
  "./Functions.h"

if [ ! -f 'OPs.cpp' ]; then
  echo "Cannot find OPs.cpp!"
  exit 1
else
  cp OPs.cpp ../torch_ipex/csrc/cpu/
fi

if [ ! -f 'OPs.h' ]; then
  echo "Cannot find OPs.h!"
  exit 1
else
  cp OPs.h ../torch_ipex/csrc/cpu/
fi
