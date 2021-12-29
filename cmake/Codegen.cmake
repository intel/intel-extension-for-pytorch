set(SIMPLE_TRACE)
if(BUILD_SIMPLE_TRACE_IPEX_ENTRY)
        set(SIMPLE_TRACE "--simple_trace")
endif()

Function(GEN_BACKEND file_yaml)
        SET(generated_files "")
        FOREACH(f ${ARGN})
                LIST(APPEND generated_files "${IPEX_GPU_ATEN_GENERATED}/ATen/${f}")
        ENDFOREACH()
        file(GLOB_RECURSE depended_files
                ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/*.py
                ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/templates/*)
        add_custom_command(OUTPUT
                ${generated_files}
                COMMAND
                mkdir -p ${IPEX_GPU_ATEN_GENERATED}/ATen
                COMMAND
                "${PYTHON_EXECUTABLE}" -m tools.codegen.gen_backend_stubs
                --output_dir ${IPEX_GPU_ATEN_GENERATED}/ATen
                --source_yaml ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/yaml/${file_yaml}
                ${SIMPLE_TRACE}
                WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts
                DEPENDS
                ${depended_files}
                ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/yaml/${file_yaml})
endfunction(GEN_BACKEND)

GEN_BACKEND(xpu_functions.yaml XPUNativeFunctions.h RegisterXPU.cpp RegisterAutogradXPU.cpp)
GEN_BACKEND(quantizedxpu_functions.yaml QuantizedXPUNativeFunctions.h RegisterQuantizedXPU.cpp)
GEN_BACKEND(sparsexpu_functions.yaml SparseXPUNativeFunctions.h RegisterSparseXPU.cpp)

list(APPEND gpu_generated_src ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterAutogradXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/RegisterSparseXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        ${IPEX_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeSparseXPU.cpp)

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
