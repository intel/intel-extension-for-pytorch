#!/bin/bash

python "./gen-cpu-ops.py" \
  --gen_class_mode 1\
  --output_folder="./" \
  "./OPs.h" \
  "./RegistrationDeclarations.h" \
  "./Functions.h"
