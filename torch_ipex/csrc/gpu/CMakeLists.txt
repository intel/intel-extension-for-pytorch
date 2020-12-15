add_subdirectory(./aten)
add_subdirectory(./jit)

FILE(GLOB aten_core_src "Module.cpp" "Stream.cpp" "Storage.cpp" )

set(DPCPP_GPU_ATEN_GENERATED "${CMAKE_CURRENT_SOURCE_DIR}/aten/generated")

add_custom_command(OUTPUT
        ${DPCPP_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${DPCPP_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp
        COMMAND
        mkdir -p ${DPCPP_GPU_ATEN_GENERATED} && mkdir -p ${DPCPP_GPU_ATEN_GENERATED}/ATen
        COMMAND
        "${PYTHON_EXECUTABLE}" ${PROJECT_SOURCE_DIR}/scripts/gpu/gen_code.py --declarations-path
        ${PROJECT_SOURCE_DIR}/scripts/declarations/Declarations.yaml
        --out ${DPCPP_GPU_ATEN_GENERATED}/ATen/
        --source-path ${PROJECT_SOURCE_DIR}
        DEPENDS
        ${PROJECT_SOURCE_DIR}/scripts/gpu/DPCPPGPUType.h
        ${PROJECT_SOURCE_DIR}/scripts/gpu/QUANTIZEDDPCPPGPUType.h)

set(DPCPP_SRCS ${aten_core_src} ${DPCPP_ATEN_SRCS} ${DPCPP_JIT_SRCS}
        ${DPCPP_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${DPCPP_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp)

include_directories(${DPCPP_GPU_ATEN_GENERATED})
MESSAGE(STATUS "include generated ${DPCPP_GPU_ATEN_GENERATED}")

# A workaround to disable assert conflict in PyTorch proper
# Should be removed if DPC++ compiler provides its owner macro
add_definitions(-DUSE_DPCPP)

if(USE_PERSIST_STREAM)
  add_definitions(-DUSE_PERSIST_STREAM)
endif()

if(BUILD_INTERNAL_DEBUG)
  add_definitions(-DBUILD_INTERNAL_DEBUG)
endif()

if(BUILD_DOUBLE_KERNEL)
  add_definitions(-DBUILD_DOUBLE_KERNEL)
endif()

if (USE_GEN12HP_ONEDNN)
  add_definitions(-DUSE_GEN12HP_ONEDNN)
endif()

if (USE_PRIMITIVE_CACHE)
  add_definitions(-DUSE_PRIMITIVE_CACHE)
endif()

# Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -D__STRICT_ANSI__")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl-unnamed-lambda")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fno-sycl-early-optimizations")
set_source_files_properties(${DPCPP_SRCS} COMPILE_FLAGS "${IPEX_COMPILE_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")
if(BUILD_BY_PER_KERNEL)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_kernel")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl, -T ${PROJECT_SOURCE_DIR}/cmake/per_ker.ld")
else()
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_source")
endif()
message(STATUS "DPCPP found. Compiling with SYCL support")

if(USE_USM)
  add_definitions(-DUSE_USM)
  message(STATUS "USM is enabled as device memory management!")
endif()

if (USE_MULTI_CONTEXT)
  add_definitions(-DUSE_MULTI_CONTEXT)
  message(STATUS "multi context is enabled!")
endif()

add_library(torch_ipex_gpu SHARED ${DPCPP_SRCS})

if(USE_ONEDPL)
  find_package(oneDPL)
  target_link_libraries(torch_ipex_gpu PUBLIC oneDPL)
  add_definitions(-DUSE_ONEDPL)
  add_definitions(-DONEDPL_USE_TBB_BACKEND=0)
endif()

if (USE_ONEMKL)
  find_package(MKLDPCPP QUIET)
  if (MKLDPCPP_FOUND)
    target_link_libraries(torch_ipex_gpu PUBLIC ${ONEMKL_SHARED_LIBS})
    include_directories(${ONEMKL_INCLUDE_DIR})
    add_definitions(-DUSE_ONEMKL)
  else()
    set(USE_ONEMKL OFF)
    message(WARNING "Cannot find oneMKL.")
  endif()
endif()

set(ONEDNN_USE_SYCL ON)
find_package(oneDNN QUIET)
if(ONEDNN_FOUND)
  target_link_libraries(torch_ipex_gpu PUBLIC ${ONEDNN_LIBRARIES})
  include_directories(BEFORE SYSTEM ${ONEDNN_INCLUDE_DIR})
  add_dependencies(torch_ipex_gpu dnnl)
else()
  message(FATAL_ERROR "Cannot find oneDNN")
endif()

target_include_directories(torch_ipex_gpu PUBLIC
        ${PYTHON_INCLUDE_DIR}
        ${IPEX_C_SOURCE_DIR}
        ${DPCPP_GPU_ROOT}
        ${DPCPP_GPU_ATEN_SRC_ROOT}
        ${PYTORCH_INCLUDE_DIR})
target_include_directories(torch_ipex_gpu PUBLIC ${ONEDNN_INCLUDE_DIR})
target_link_libraries(torch_ipex_gpu PUBLIC ${ONEDNN_LIBRARIES})
add_dependencies(torch_ipex_gpu dnnl)

install(TARGETS torch_ipex_gpu LIBRARY DESTINATION lib)
