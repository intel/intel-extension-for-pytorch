## Included by CMakeLists
if(GPU_cmake_included)
    return()
endif()
set(GPU_cmake_included true)

# ---[ Build flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
# Eigen fails to build with some versions, so convert this to a warning
# Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
if (CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=old-style-cast")
# These flags are not available in GCC-4.8.5. Set only when using clang.
# Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-infinite-recursion")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-typedef-redefinition")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-aligned-allocation-unavailable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-writable-strings")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++14-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++17-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments")
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  endif()
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()
if ((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
  OR (CMAKE_COMPILER_IS_GNUCXX
  AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()
if (WERROR)
  check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
  if (NOT COMPILER_SUPPORT_WERROR)
    set(WERROR FALSE)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif(WERROR)
if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
endif()
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math")

#link option
set(CMAKE_SKIP_RPATH TRUE)

# ---[ Main build
set(IPEX_C_SOURCE_DIR "${PROJECT_SOURCE_DIR}/torch_ipex/csrc")
set(DPCPP_GPU_ROOT "${IPEX_C_SOURCE_DIR}/gpu")
set(DPCPP_GPU_ATEN_SRC_ROOT "${DPCPP_GPU_ROOT}/aten")
set(DPCPP_GPU_ATEN_GENERATED "${DPCPP_GPU_ROOT}/aten/generated")

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
        ${PROJECT_SOURCE_DIR}/scripts/declarations/Declarations.yaml
        ${PROJECT_SOURCE_DIR}/scripts/gpu/gen_code.py
        ${PROJECT_SOURCE_DIR}/scripts/gpu/DPCPPGPUType.h
        ${PROJECT_SOURCE_DIR}/scripts/gpu/QUANTIZEDDPCPPGPUType.h)

# sources
set(DPCPP_ATEN_SRCS)
add_subdirectory(torch_ipex/csrc/gpu/aten)

set(DPCPP_JIT_SRCS)
add_subdirectory(torch_ipex/csrc/gpu/jit)

set(TORCH_IPEX_SRCS)
FILE(GLOB IPEX_UTIL_SRC "${IPEX_C_SOURCE_DIR}/*.cpp")
FILE(GLOB GPU_UTIL_SRC "${DPCPP_GPU_ROOT}/*.cpp")
list(APPEND TORCH_IPEX_SRCS ${DPCPP_ATEN_SRCS} ${DPCPP_JIT_SRCS} ${IPEX_UTIL_SRC} ${GPU_UTIL_SRC})

add_library(torch_ipex SHARED ${TORCH_IPEX_SRCS}
        ${DPCPP_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeXPU.cpp
        ${DPCPP_GPU_ATEN_GENERATED}/ATen/AtenIpexTypeQuantizedXPU.cpp)

set_target_properties(torch_ipex PROPERTIES PREFIX "")
set_target_properties(torch_ipex PROPERTIES OUTPUT_NAME ${LIB_NAME})

# includes
if(DEFINED PYTORCH_INCLUDE_DIR)
  include_directories(${PYTORCH_INCLUDE_DIR})
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${IPEX_C_SOURCE_DIR})
include_directories(${DPCPP_GPU_ROOT})
include_directories(${DPCPP_GPU_ATEN_SRC_ROOT})
include_directories(${DPCPP_GPU_ATEN_GENERATED})

# pytorch library
if(DEFINED PYTORCH_LIBRARY_DIR)
  link_directories(${PYTORCH_LIBRARY_DIR})
  target_link_libraries(torch_ipex PUBLIC ${PYTORCH_LIBRARY_DIR}/libtorch_cpu.so)
  target_link_libraries(torch_ipex PUBLIC ${PYTORCH_LIBRARY_DIR}/libtorch_python.so)
  target_link_libraries(torch_ipex PUBLIC ${PYTORCH_LIBRARY_DIR}/libc10.so)
else()
  message(FATAL_ERROR, "Cannot find PyTorch library directory")
endif()

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
  # Enable FRAMEWORK primitive cache
  add_definitions(-DUSE_PRIMITIVE_CACHE)
endif()

# Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
if(USE_ONEDPL)
  find_package(oneDPL)
  target_link_libraries(torch_ipex PUBLIC oneDPL)
  add_definitions(-DUSE_ONEDPL)
  add_definitions(-DONEDPL_USE_TBB_BACKEND=0)
endif()

set(AOT_ARCH_OPT "spir64_gen-unknown-unknown-sycldevice")
set(SPIRV_OPT "spir64-unknown-unknown-sycldevice")
if(USE_AOT_DEVLIST)
  set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
endif()

set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -D__STRICT_ANSI__")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl-unnamed-lambda")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fno-sycl-early-optimizations")
set_source_files_properties(${TORCH_IPEX_SRCS} COMPILE_FLAGS "${IPEX_COMPILE_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
if(USE_AOT_DEVLIST)
  set(BUILD_BY_PER_KERNEL OFF)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_source")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT} -Xsycl-target-backend=${AOT_ARCH_OPT} \"-device ${USE_AOT_DEVLIST}\"")
elseif(BUILD_BY_PER_KERNEL)
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

if (USE_ONEMKL)
  find_package(MKLDPCPP QUIET)
  if (MKLDPCPP_FOUND)
    target_link_libraries(torch_ipex PUBLIC ${ONEMKL_SHARED_LIBS})
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
  target_link_libraries(torch_ipex PUBLIC ${ONEDNN_LIBRARIES})
  include_directories(BEFORE SYSTEM ${ONEDNN_INCLUDE_DIR})
  add_dependencies(torch_ipex dnnl)
else()
  message(FATAL_ERROR "Cannot find oneDNN")
endif()

target_link_libraries(torch_ipex PUBLIC ${EXTRA_SHARED_LIBS})

install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib)
