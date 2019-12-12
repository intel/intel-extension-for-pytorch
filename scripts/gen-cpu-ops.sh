#!/bin/bash

python "./gen-cpu-ops.py" \
  --gen_class_mode \
  --output_folder="./" \
  "./OPs.h" \
  "./RegistrationDeclarations.h" \
  "./Functions.h"
