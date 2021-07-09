add_custom_command(OUTPUT
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeSparseXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.h
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.h
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeSparseXPU.h
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
        ${PROJECT_SOURCE_DIR}/scripts/gpu/QUANTIZEDDPCPPGPUType.h
        ${PROJECT_SOURCE_DIR}/scripts/gpu/SPARSEDPCPPGPUType.h)

list(APPEND gpu_generated_src ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeSparseXPU.cpp)

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
