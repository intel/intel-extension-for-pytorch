
set(IPEX_GPU_ATEN_GENERATED "${PROJECT_SOURCE_DIR}/csrc/aten/generated")

add_custom_command(OUTPUT
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.h
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.h
        COMMAND
        mkdir -p ${IPEX_GPU_ATEN_GENERATED} && mkdir -p ${IPEX_GPU_ATEN_GENERATED}/ATen
        COMMAND
        "${PYTHON_EXECUTABLE}" -m scripts.gpu.gen_code --declarations-path
        ${PROJECT_SOURCE_DIR}/scripts/declarations/Declarations.yaml
        --out ${IPEX_GPU_ATEN_GENERATED}/ATen/
        --source-path ${PROJECT_SOURCE_DIR}
        WORKING_DIRECTORY
        ${IPEX_ROOT_DIR}/
        DEPENDS
        ${PROJECT_SOURCE_DIR}/scripts/declarations/Declarations.yaml
        ${PROJECT_SOURCE_DIR}/scripts/gpu/gen_code.py
        ${PROJECT_SOURCE_DIR}/scripts/gpu/DPCPPGPUType.h
        ${PROJECT_SOURCE_DIR}/scripts/gpu/QUANTIZEDDPCPPGPUType.h)

list(APPEND gpu_generated_src ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp)

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.h
                                              ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.h)
add_library(IPEX_GPU_FILES_GEN_LIB INTERFACE)
add_dependencies(IPEX_GPU_FILES_GEN_LIB IPEX_GPU_GEN_TARGET)