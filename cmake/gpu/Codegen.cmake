if(Codegen_GPU_cmake_included)
    return()
endif()
set(Codegen_GPU_cmake_included true)
set(BUILD_IPEX_GPU_ATEN_GENERATED_DIR "${CMAKE_BINARY_DIR}/csrc/gpu/")
set(BUILD_IPEX_GPU_ATEN_GENERATED "${BUILD_IPEX_GPU_ATEN_GENERATED_DIR}/xpu/ATen")
set(BUILD_IPEX_GPU_ATOI_GENERATED "${BUILD_IPEX_GPU_ATEN_GENERATED_DIR}/xpu/atoi")
file(MAKE_DIRECTORY ${BUILD_IPEX_GPU_ATEN_GENERATED})
file(MAKE_DIRECTORY ${BUILD_IPEX_GPU_ATOI_GENERATED})
set(RegisterXPU_PATH ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterXPU_0.cpp)
set(RegisterSparseXPU_PATH ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterSparseXPU_0.cpp)
set(RegisterNestedTensorXPU_PATH ${BUILD_IPEX_GPU_ATEN_GENERATED}/RegisterNestedTensorXPU_0.cpp)

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
                WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts/tools
                DEPENDS
                ${depended_files}
                ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/${file_yaml})
endfunction(GEN_BACKEND)


function(GEN_XPU file_yaml)
  set(generated_files "")
  foreach(f ${ARGN})
    list(APPEND generated_files "${BUILD_IPEX_GPU_ATEN_GENERATED}/${f}")
  endforeach()
  file(GLOB_RECURSE depend_files ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/${file_yaml})

  add_custom_command(
    OUTPUT ${generated_files}
    COMMAND
    "${PYTHON_EXECUTABLE}" -m torchgen.gen
    --source-path ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/
    --install-dir ${BUILD_IPEX_GPU_ATEN_GENERATED}
    --per-operator-headers
    --static-dispatch-backend
    --backend-whitelist XPU SparseXPU NestedTensorXPU
    --xpu
    --update-aoti-c-shim
    --extend-aoti-c-shim
    --aoti-install-dir=${BUILD_IPEX_GPU_ATOI_GENERATED}
    # Codegen post-process
    COMMAND "${PYTHON_EXECUTABLE}" ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/remove_headers.py --register_xpu_path ${RegisterXPU_PATH}
    COMMAND "${PYTHON_EXECUTABLE}" ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/remove_headers.py --register_xpu_path ${RegisterSparseXPU_PATH}
    COMMAND "${PYTHON_EXECUTABLE}" ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/remove_headers.py --register_xpu_path ${RegisterNestedTensorXPU_PATH}
    WORKING_DIRECTORY ${IPEX_ROOT_DIR}
    DEPENDS
    ${depended_files}
    ${PROJECT_SOURCE_DIR}/scripts/tools/torchgen/yaml/native/${file_yaml})

endfunction(GEN_XPU)

GEN_XPU(
  native_functions.yaml
  ${BUILD_IPEX_XPU_ATEN_GENERATED}/XPUFunctions.h
  ${BUILD_IPEX_XPU_ATEN_GENERATED}/RegisterXPU_0.cpp
  ${BUILD_IPEX_XPU_ATEN_GENERATED}/RegisterSparseXPU_0.cpp
  ${BUILD_IPEX_XPU_ATEN_GENERATED}/RegisterNestedTensorXPU_0.cpp
  ${BUILD_IPEX_XPU_ATOI_GENERATED}//c_shim_xpu.h
  ${BUILD_IPEX_XPU_ATOI_GENERATED}/c_shim_xpu.cpp
)

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
                WORKING_DIRECTORY ${IPEX_ROOT_DIR}/scripts/tools
                DEPENDS
                ${source_file})
endfunction(GEN_NATIVE_IMPL)


list(APPEND gpu_generated_src ${RegisterXPU_PATH} ${RegisterSparseXPU_PATH} ${RegisterNestedTensorXPU_PATH})

add_custom_target(IPEX_GPU_GEN_TARGET DEPENDS ${gpu_generated_src})
set(IPEX_GPU_GEN_FILES ${gpu_generated_src})
