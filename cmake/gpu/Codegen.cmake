if(Codegen_GPU_cmake_included)
    return()
endif()
set(Codegen_GPU_cmake_included true)

set(SIMPLE_TRACE)
if(BUILD_SIMPLE_TRACE)
        set(SIMPLE_TRACE "--simple_trace")
endif()

set(BUILD_IPEX_GPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/csrc/aten/generated/ATen")
file(MAKE_DIRECTORY ${BUILD_IPEX_GPU_ATEN_GENERATED})

Function(GEN_BACKEND file_yaml)
        SET(generated_files "")
        FOREACH(f ${ARGN})
                LIST(APPEND generated_files "${BUILD_IPEX_GPU_ATEN_GENERATED}/${f}")
        ENDFOREACH()
        file(GLOB_RECURSE depended_files
                ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/${file_yaml})
        add_custom_command(OUTPUT
                ${generated_files}
                COMMAND
                "${PYTHON_EXECUTABLE}" -m torchgen.gen_backend_stubs
                --output_dir ${BUILD_IPEX_GPU_ATEN_GENERATED}
                --source_yaml ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/${file_yaml}
                ${SIMPLE_TRACE}
                WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts/tools
                DEPENDS
                ${depended_files}
                ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/${file_yaml})
endfunction(GEN_BACKEND)

GEN_BACKEND(xpu_functions.yaml XPUNativeFunctions.h RegisterXPU.cpp RegisterAutogradXPU.cpp)
GEN_BACKEND(quantizedxpu_functions.yaml QuantizedXPUNativeFunctions.h RegisterQuantizedXPU.cpp)
GEN_BACKEND(sparsexpu_functions.yaml SparseXPUNativeFunctions.h RegisterSparseXPU.cpp)
GEN_BACKEND(nestedtensorxpu_functions.yaml NestedTensorXPUNativeFunctions.h RegisterNestedTensorXPU.cpp)

Function(GEN_NATIVE_IMPL file_yaml file_impl file_head)
        set(source_file ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/${file_yaml})
        set(output_file ${BUILD_IPEX_GPU_ATEN_GENERATED}/${file_impl})
        set(header_file ${file_head})
        set(GENERATE_IMPL "--generate_impl")
        add_custom_command(OUTPUT
                ${output_file}
                COMMAND
                "${PYTHON_EXECUTABLE}" -m torchgen.gen_backend_stubs
                --source_yaml ${source_file}
                --header_file ${header_file}
                --output_impl_file ${output_file}
                ${GENERATE_IMPL}
                ${SIMPLE_TRACE}
                WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts/tools
                DEPENDS
                ${source_file})
endfunction(GEN_NATIVE_IMPL)


list(APPEND gpu_generated_src ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterXPU.cpp
        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterAutogradXPU.cpp
        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterQuantizedXPU.cpp
        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterSparseXPU.cpp
        ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterNestedTensorXPU.cpp)

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
