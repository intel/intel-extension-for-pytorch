cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
SET(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
SET(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
SET(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

set(DPCPP_CPU_ROOT "${PROJECT_SOURCE_DIR}/torch_ipex/csrc/cpu")


# TODO: Once llga is merged into oneDNN, use oneDNN directly as the third_party of IPEX
# use the oneDNN in llga temporarily: third_party/llga/third_party/oneDNN
add_subdirectory(${DPCPP_THIRD_PARTY_ROOT}/llga)
# add_subdirectory(${DPCPP_THIRD_PARTY_ROOT}/mkl-dnn)


#find_package(TorchCCL REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

FIND_PACKAGE(AVX)

IF (NOT C_AVX512_FOUND AND NOT CXX_AVX512_FOUND)
  message(FATAL_ERROR "Please build IPEX on Machines that support AVX512.")
ENDIF()

# Define build type
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

IF("${IPEX_DISP_OP}" STREQUAL "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIPEX_DISP_OP")
ENDIF()

IF("${IPEX_PROFILE_OP}" STREQUAL "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIPEX_PROFILE_OP")
ENDIF()

IF("${ENABLE_AUTOCAST_VERBOSE}" STREQUAL "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_AUTOCAST_VERBOSE")
ENDIF()

# ---[ Build flags
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
if (CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=old-style-cast")
IF (C_AVX512_FOUND OR CXX_AVX512_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_CAPABILITY_AVX2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_AVX512")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512bw")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512dq")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c")
ENDIF()
IF (C_AVX512_BF16_FOUND OR CXX_AVX512_BF16_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512bf16 -DAVX512_BF16")
ENDIF()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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

# ---[ Main build

# includes

# include mkl-dnn before PyTorch
# Otherwise, path_to_pytorch/torch/include/dnnl.hpp will be used as the header

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/torch_ipex)
include_directories(${PROJECT_SOURCE_DIR}/torch_ipex/csrc/)

include_directories(${DPCPP_THIRD_PARTY_ROOT}/llga/include)
include_directories(${PROJECT_SOURCE_DIR}/build/third_party/llga/third_party/oneDNN/include)
include_directories(${DPCPP_THIRD_PARTY_ROOT}/llga/third_party/oneDNN/include)
# TODO: once llga is merged into oneDNN, use oneDNN directly as the third_party instead of using that inside llga
# include_directories(${PROJECT_SOURCE_DIR}/build/third_party/mkl-dnn/include)
# include_directories(${DPCPP_THIRD_PARTY_ROOT}/mkl-dnn/include)

# Set installed PyTorch dir
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

# Set Python include dir
if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR, "Cannot find installed Python head file directory")
endif()

# include pybind11 after pytorch include
# to be able to use the py::class defined in PyTorch when binding c++ API to Python in IPEX
include_directories(${DPCPP_THIRD_PARTY_ROOT}/pybind11/include)

# sources
set(DPCPP_SRCS)
set(DPCPP_COMMON_SRCS)
set(DPCPP_CPU_SRCS)
set(DPCPP_JIT_SRCS)

add_subdirectory(${DPCPP_ROOT})
add_subdirectory(${DPCPP_ROOT}/quantization)
add_subdirectory(${DPCPP_ROOT}/jit)
add_subdirectory(${DPCPP_ROOT}/cpu)

# Compile code with pybind11
set(DPCPP_SRCS ${DPCPP_ATEN_SRCS} ${DPCPP_COMMON_SRCS} ${DPCPP_CPU_SRCS} ${DPCPP_JIT_SRCS})
add_library(${PLUGIN_NAME} SHARED ${DPCPP_SRCS})

#link_directories(${PYTORCH_INSTALL_DIR}/lib)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(${PLUGIN_NAME} PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(${PLUGIN_NAME} PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  target_compile_definitions(${PLUGIN_NAME} PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

add_dependencies(${PLUGIN_NAME} dnnl_graph)
target_link_libraries(${PLUGIN_NAME} PUBLIC dnnl_graph)
link_directories(${PYTORCH_INSTALL_DIR}/lib)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_python.so)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

target_compile_options(${PLUGIN_NAME} PRIVATE "-DC10_BUILD_MAIN_LIB")
install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
