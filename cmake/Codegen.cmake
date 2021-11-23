Function(GEN_BACKEND_ONECPP file_cpp file_h file_yaml)
add_custom_command(OUTPUT
        ${IPEX_GPU_ATEN_GENERATED}/ATen/${file_cpp}
        ${IPEX_GPU_ATEN_GENERATED}/ATen/${file_h}
        COMMAND
        mkdir -p ${IPEX_GPU_ATEN_GENERATED} && mkdir -p ${IPEX_GPU_ATEN_GENERATED}/ATen
        COMMAND
        "${PYTHON_EXECUTABLE}" -m tools.codegen.gen_backend_stubs
        --output_dir ${IPEX_GPU_ATEN_GENERATED}/ATen
        --source_yaml ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/yaml/${file_yaml}
        WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts
        DEPENDS
        ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/gen_backend_stubs.py
        ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/yaml/${file_yaml})
endfunction(GEN_BACKEND_ONECPP)

Function(GEN_BACKEND file_cpp file_autograd_cpp file_h file_yaml)
add_custom_command(OUTPUT
        ${IPEX_GPU_ATEN_GENERATED}/ATen/${file_cpp}
        ${IPEX_GPU_ATEN_GENERATED}/ATen/${file_autograd_cpp}
        ${IPEX_GPU_ATEN_GENERATED}/ATen/${file_h}
        COMMAND
        mkdir -p ${IPEX_GPU_ATEN_GENERATED} && mkdir -p ${IPEX_GPU_ATEN_GENERATED}/ATen
        COMMAND
        "${PYTHON_EXECUTABLE}" -m tools.codegen.gen_backend_stubs
        --output_dir ${IPEX_GPU_ATEN_GENERATED}/ATen
        --source_yaml ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/yaml/${file_yaml}
        WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts
        DEPENDS
        ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/gen_backend_stubs.py
        ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/yaml/${file_yaml})
endfunction(GEN_BACKEND)

GEN_BACKEND(RegisterXPU.cpp RegisterAutogradXPU.cpp XPUNativeFunctions.h xpu_functions.yaml)
GEN_BACKEND_ONECPP(RegisterQuantizedXPU.cpp QuantizedXPUNativeFunctions.h quantizedxpu_functions.yaml)
GEN_BACKEND_ONECPP(RegisterSparseXPU.cpp SparseXPUNativeFunctions.h sparsexpu_functions.yaml)

list(APPEND gpu_generated_src ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterAutogradXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterSparseXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeSparseXPU.cpp)

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
