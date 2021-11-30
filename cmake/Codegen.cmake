list(APPEND gpu_generated_src ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeSparseXPU.cpp)

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
