cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

#find_package(TorchCCL REQUIRED)

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

# TODO: Once llga is merged into oneDNN, use oneDNN directly as the third_party of IPEX
# use the oneDNN in llga temporarily: third_party/llga/third_party/oneDNN

set(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
if(DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND})
  set(DNNL_GRAPH_BUILD_COMPILER_BACKEND ON CACHE BOOL "" FORCE)
  set(DNNL_GRAPH_LLVM_CONFIG "llvm-config-13" CACHE STRING "" FORCE)
endif()
add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/llga cpu_third_party/llga)
# add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/mkl-dnn cpu_third_party/mkl-dnn)

IF("${IPEX_DISP_OP}" STREQUAL "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIPEX_DISP_OP")
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
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-Bsymbolic-functions")

# ---[ Main build

# includes

# include mkl-dnn before PyTorch
# Otherwise, path_to_pytorch/torch/include/dnnl.hpp will be used as the header

include_directories(${IPEX_PROJECT_TOP_DIR})
include_directories(${IPEX_CPU_CPP_ROOT})
include_directories(${IPEX_CPU_CPP_ROOT}/aten)
include_directories(${IPEX_CPU_CPP_ROOT}/utils)

include_directories(${IPEX_JIT_CPP_ROOT})
include_directories(${IPEX_COMMON_CPP_ROOT})

include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/llga/include)
include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/llga/third_party/oneDNN/include)
# TODO: once llga is merged into oneDNN, use oneDNN directly as the third_party instead of using that inside llga
# include_directories(${PROJECT_SOURCE_DIR}/build/third_party/mkl-dnn/include)
# include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/mkl-dnn/include)

# Set installed MKL install dir
include_directories(${MKL_INSTALL_DIR}/include)

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

# sources
set(DPCPP_ISA_SRCS)
set(DPCPP_ISA_SRCS_ORIGIN)
include(${IPEX_PROJECT_TOP_DIR}/cmake/Codegen.cmake)

set(IPEX_CPU_CPP_SRCS)
set(IPEX_CPU_CPP_UTILS_SRCS)
set(DPCPP_QUANTIZATION_SRCS)
set(IPEX_CPU_CPP_AUTOCAST_SRCS)
set(IPEX_CPU_CPP_ATEN_SRCS)
set(IPEX_CPU_CPP_DYNDISP_SRCS)
set(IPEX_CPU_CPP_ISA_SRCS)
set(IPEX_CPU_CPP_IDEEP_SRCS)

set(IPEX_JIT_CPP_SRCS)
set(DPCPP_COMMON_SRCS)

# foreach(file_path ${DPCPP_ISA_SRCS})
#   message(${file_path})
# endforeach()

add_subdirectory(${IPEX_CPU_CPP_ROOT}/aten)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/autocast)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/dyndisp)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/ideep)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/isa)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/utils)


add_subdirectory(${IPEX_JIT_CPP_ROOT} jit_cpu)
add_subdirectory(${IPEX_COMMON_CPP_ROOT} common_cpu)

# Compile code with pybind11
set(IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_DYNDISP_SRCS} ${DPCPP_ISA_SRCS} ${DPCPP_COMMON_SRCS} ${IPEX_CPU_CPP_UTILS_SRCS} ${DPCPP_QUANTIZATION_SRCS} ${IPEX_JIT_CPP_SRCS}
    ${IPEX_CPU_CPP_ISA_SRCS} ${IPEX_CPU_CPP_IDEEP_SRCS} ${IPEX_CPU_CPP_AUTOCAST_SRCS} ${IPEX_CPU_CPP_ATEN_SRCS})

list(REMOVE_ITEM IPEX_CPU_CPP_SRCS ${DPCPP_ISA_SRCS_ORIGIN})

add_library(${PLUGIN_NAME_CPU} SHARED ${IPEX_CPU_CPP_SRCS})

link_directories(${PYTORCH_INSTALL_DIR}/lib)
add_dependencies(${PLUGIN_NAME_CPU} dnnl_graph)
# If Graph Compiler is built, then it should link to its LLVM dependencies,
# and not the LLVM symbols exposed by PyTorch.
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC dnnl_graph)
if (DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND})
  get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB dnnl_graphcompiler_llvm_lib INTERFACE_LINK_LIBRARIES)
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC graphcompiler ${DNNL_GRAPHCOMPILER_LLVM_LIB})
endif()

find_library(MKL_LIBRARY libmkl_core.a PATHS "${MKL_INSTALL_DIR}/lib" "${MKL_INSTALL_DIR}/lib/intel64")
if (NOT MKL_LIBRARY)
  message(FATAL_ERROR "libmkl_core.a not found in ${MKL_INSTALL_DIR}")
endif()
get_filename_component(MKL_LIBRARY_DIR ${MKL_LIBRARY} DIRECTORY)
message(STATUS "Using MKL in ${MKL_LIBRARY_DIR}")
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC
  -Wl,--start-group
  ${MKL_LIBRARY_DIR}/libmkl_intel_lp64.a
  ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a
  ${MKL_LIBRARY_DIR}/libmkl_core.a
  -Wl,--end-group)
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

target_compile_options(${PLUGIN_NAME_CPU} PRIVATE "-DC10_BUILD_MAIN_LIB")
