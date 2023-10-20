#!/bin/sh
python -c 'import intel_extension_for_pytorch._C as core;print(core._get_current_isa_level())'
ATEN_CPU_CAPABILITY=avx2 python -c 'import intel_extension_for_pytorch._C as core;print(core._get_current_isa_level())'
