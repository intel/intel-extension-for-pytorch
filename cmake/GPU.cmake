## Included by CMakeLists

## For ComputeCPP
include(cmake/DPCPP.cmake)

IF(USE_COMPUTECPP OR USE_DPCPP)
  INCLUDE_DIRECTORIES(SYSTEM ${ComputeCpp_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
  LIST(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${COMPUTECPP_RUNTIME_LIBRARY})
  MESSAGE(STATUS "ComputeCpp found. Compiling with SYCL support")
ELSE()
  MESSAGE(FATAL_ERROR "ComputeCpp not found. Compiling without SYCL support")
ENDIF()

# ---[ Build flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC")
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-typedef-redefinition")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-aligned-allocation-unavailable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++14-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
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

if (DPCPP_ENABLE_PROFILING)
  add_definitions(-DDPCPP_PROFILING)
endif()


# ---[ Main build

# includes
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

set (PYTORCH_INCLUDES "${PYTORCH_INSTALL_DIR}/include")
# set(PYTORCH_ROOT "${PROJECT_SOURCE_DIR}/third_party/pytorch")
# set(PYTORCH_ATEN_SRC_ROOT "${PYTORCH_ROOT}/aten/src")
# set(PYTORCH_ATEN_INCLUDES "${PYTORCH_ROOT}/aten/src/ATen")
# set(PYTORCH_ATEN_CORE_INCLUDES "${PYTORCH_ROOT}/aten/src/ATen/core")
# set(PYTORCH_C10_CORE_INCLUDES "${PYTORCH_ROOT}/c10/core")
# set(PYTORCH_C10_DPCPP_INCLUDES "${PYTORCH_ROOT}/c10/dpcpp")
# set(PYTORCH_C10_UTIL_INCLUDES "${PYTORCH_ROOT}/c10/util")
# set(PYTORCH_C10_MACROS_INCLUDES "${PYTORCH_ROOT}/c10/macros")

set(DPCPP_ROOT "${PROJECT_SOURCE_DIR}/torch_ipex/csrc")
set(DPCPP_GPU_ROOT "${PROJECT_SOURCE_DIR}/torch_ipex/csrc/gpu")
set(DPCPP_GPU_ATEN_SRC_ROOT "${DPCPP_GPU_ROOT}/aten")
set(DPCPP_GPU_ATEN_GENERATED "${DPCPP_GPU_ROOT}/aten/generated")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PYTORCH_INCLUDES})
# include_directories(${PYTORCH_ATEN_SRC_ROOT})
# include_directories(${PYTORCH_ATEN_INCLUDES})
# include_directories(${PYTORCH_ATEN_CORE_INCLUDES})
# include_directories(${PYTORCH_C10_CORE_INCLUDES})
# include_directories(${PYTORCH_C10_DPCPP_INCLUDES})
# include_directories(${PYTORCH_C10_UTIL_INCLUDES})
# include_directories(${PYTORCH_C10_MACROS_INCLUDES})
include_directories(${DPCPP_ROOT})
include_directories(${DPCPP_GPU_ROOT})
include_directories(${DPCPP_GPU_ATEN_SRC_ROOT})
include_directories(${DPCPP_GPU_ATEN_GENERATED})

set(C10_USE_GFLAGS ${USE_GFLAGS})
set(C10_USE_GLOG ${USE_GLOG})
set(C10_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(C10_USE_NUMA ${USE_NUMA})
set(C10_DISABLE_NUMA ${CAFFE2_DISABLE_NUMA})

# configure_file(
#     "${PYTORCH_C10_MACROS_INCLUDES}/cmake_macros.h.in"
#     "${CMAKE_BINARY_DIR}/c10/macros/cmake_macros.h")
# include_directories(${CMAKE_BINARY_DIR})

# configure_file(
#     "${PYTORCH_ATEN_INCLUDES}/Config.h.in"
#     "${CMAKE_BINARY_DIR}/include/ATen/Config.h")
# set(DPCPP_OUT_INCLUDE "${CMAKE_BINARY_DIR}/include")
# include_directories(${DPCPP_OUT_INCLUDE})

# do not compile pytorch any more
# use generated files for current stage
# we will generate files by ourselves in future
# so it is unnecessary to take efforts to integrate pytorch gen system
#
# set(PYTORCH_OUT_ATEN_SRC_ROOT "${CMAKE_BINARY_DIR}/aten/src")
# include_directories(${PYTORCH_OUT_ATEN_SRC_ROOT})
#
# # workaround for THGeneral.h
# set(CAFFE2_OUT_ATEN_SRC_ROOT "${CMAKE_BINARY_DIR}/third_party/pytorch/caffe2/aten/src")
# include_directories(${CAFFE2_OUT_ATEN_SRC_ROOT})

# generate c10 dispatch registration
add_custom_target(
  gen_dpcpp_gpu_c10_dispatch_registration
  COMMAND python gen-gpu-decl.py --gpu_decl=./ DPCPPGPUType.h DedicateType.h DispatchStubOverride.h RegistrationDeclarations.h
  COMMAND python gen-gpu-ops.py --output_folder=./ DPCPPGPUType.h RegistrationDeclarations_DPCPP.h Functions_DPCPP.h
  COMMAND cp ./aten_ipex_type_default.cpp.in ${DPCPP_GPU_ATEN_GENERATED}/ATen/aten_ipex_type_default.cpp
  COMMAND cp ./aten_ipex_type_default.h.in ${DPCPP_GPU_ATEN_GENERATED}/ATen/aten_ipex_type_default.h
  COMMAND cp ./aten_ipex_type_dpcpp.h.in ${DPCPP_GPU_ATEN_GENERATED}/ATen/aten_ipex_type_dpcpp.h
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts/gpu
)

# includes installation
add_custom_target(
  install_dpcpp_gpu_includes
  COMMAND rm -rf "${CMAKE_BINARY_DIR}/include"
  COMMAND mkdir "${CMAKE_BINARY_DIR}/include"
  COMMAND mkdir "${CMAKE_BINARY_DIR}/include/torch"
  COMMAND mkdir "${CMAKE_BINARY_DIR}/include/torch/extension"
  COMMAND mkdir "${CMAKE_BINARY_DIR}/include/torch/extension/dpcpp"
  COMMAND cp "${PROJECT_SOURCE_DIR}/torch_ipex/csrc/gpu/includes/*.h" "${CMAKE_BINARY_DIR}/include/torch/extension/dpcpp"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# dependencies
# set(DPCPP_DEP)
# include(cmake/Dependencies.cmake)

# sources
set(DPCPP_SRCS)

set(DPCPP_ATEN_SRCS)
add_subdirectory(torch_ipex/csrc/gpu/aten)
list(APPEND DPCPP_SRCS ${DPCPP_ATEN_SRCS})

link_directories(${PYTORCH_INSTALL_DIR}/lib)

add_library(torch_ipex SHARED ${DPCPP_SRCS})
target_link_libraries(torch_ipex PUBLIC dnnl)
target_link_libraries(torch_ipex PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(torch_ipex PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

set_target_properties(torch_ipex PROPERTIES PREFIX "")
set_target_properties(torch_ipex PROPERTIES OUTPUT_NAME "_torch_ipex")
add_dependencies(torch_ipex gen_dpcpp_gpu_c10_dispatch_registration)
add_dependencies(torch_ipex install_dpcpp_gpu_includes)
# add_dependencies(torch_ipex ${DPCPP_DEP})
# target_link_libraries(torch PUBLIC c10_sycl)
# target_include_directories(torch INTERFACE $<INSTALL_INTERFACE:include>)
# target_include_directories(torch PRIVATE ${Caffe2_SYCL_INCLUDE})
# target_link_libraries(torch PRIVATE ${Caffe2_SYCL_DEPENDENCY_LIBS})

IF(USE_COMPUTECPP)
  add_sycl_to_target(TARGET torch_ipex SOURCES ${DPCPP_SRCS})
ENDIF()

IF(USE_DPCPP)
  #add_library(c10_sycl ${C10_SYCL_SRCS} ${C10_CUDA_HEADERS})
  set_source_files_properties(${DPCPP_SRCS} COMPILE_FLAGS "-fsycl -D__STRICT_ANSI__")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl -fsycl-device-code-split=per_source")
ENDIF()

# if(USE_DPCPP)
#   set_source_files_properties(${Caffe2_SYCL_SRCS} COMPILE_FLAGS "-fsycl -D__STRICT_ANSI__ -DUSE_DPCPP")
#   target_link_libraries(torch PRIVATE "-fsycl")
# endif()

add_dependencies(torch_ipex dnnl)
