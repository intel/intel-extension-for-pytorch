#!/bin/bash

python "./gen-common-ops.py" \
  --gen_class_mode \
  --output_folder="./" \
  "./aten_ipex_type_default.h" \
  "./RegistrationDeclarations.h" \
  "./Functions.h"
