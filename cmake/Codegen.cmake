set(SIMPLE_TRACE)
if(BUILD_SIMPLE_TRACE)
        set(SIMPLE_TRACE "--simple_trace")
endif()

set(BUILD_IPEX_GPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/csrc/aten/generated/ATen")

Function(GEN_BACKEND file_yaml)
        SET(generated_files "")
        FOREACH(f ${ARGN})
                LIST(APPEND generated_files "${BUILD_IPEX_GPU_ATEN_GENERATED}/${f}")
        ENDFOREACH()
        file(GLOB_RECURSE depended_files
                ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/*.py
                ${PROJECT_SOURCE_DIR}/scripts/tools/codegen/templates/*)
        add_custom_command(OUTPUT
                ${generated_files}
                COMMAND
                mkdir -p ${BUILD_IPEX_GPU_ATEN_GENERATED}
                COMMAND
                "${PYTHON_EXECUTABLE}" -m tools.codegen.gen_backend_stubs
                --output_dir ${BUILD_IPEX_GPU_ATEN_GENERATED}
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

#list(APPEND gpu_generated_src ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterXPU.cpp
#        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterAutogradXPU.cpp
#        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterQuantizedXPU.cpp
#        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterSparseXPU.cpp)

set(gpu_generated_src "")

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
