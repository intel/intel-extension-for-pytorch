# CMAKE FILE for libintel-ext-pt-cpu.so
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(IPEX CXX C)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME_CPU intel-ext-pt-cpu)

# Setup project top directory.
set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../")

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}/lib/:${RPATH_VALUE}/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IPEX_INSTALL_LIBDIR})

list(APPEND CMAKE_MODULE_PATH ${IPEX_PROJECT_TOP_DIR}/cmake/Modules)

set(IPEX_CPU_CPP_ROOT "${IPEX_PROJECT_TOP_DIR}/csrc/cpu")
set(IPEX_CPU_CPP_THIRD_PARTY_ROOT "${IPEX_PROJECT_TOP_DIR}/third_party")
set(IPEX_JIT_CPP_ROOT "${IPEX_PROJECT_TOP_DIR}/csrc/jit")
set(IPEX_UTLIS_CPP_ROOT "${IPEX_PROJECT_TOP_DIR}/csrc/utils")

set(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

#find_package(TorchCCL REQUIRED)

# TODO: Once llga is merged into oneDNN, use oneDNN directly as the third_party of IPEX
# use the oneDNN in llga temporarily: third_party/llga/third_party/oneDNN

set(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
if(DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND})
  set(DNNL_GRAPH_BUILD_COMPILER_BACKEND ON CACHE BOOL "" FORCE)
  set(DNNL_GRAPH_LLVM_CONFIG "llvm-config-13" CACHE STRING "" FORCE)
endif()
add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn cpu_third_party/ideep/mkl-dnn)
# add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/mkl-dnn cpu_third_party/mkl-dnn)

IF("${IPEX_DISP_OP}" STREQUAL "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIPEX_DISP_OP")
ENDIF()

# ---[ Build flags
include(${IPEX_PROJECT_TOP_DIR}/cmake/cpu/BuildFlags.cmake)

# ---[ Main build
# include mkl-dnn before PyTorch
# Otherwise, path_to_pytorch/torch/include/dnnl.hpp will be used as the header

# includes
include_directories(${IPEX_PROJECT_TOP_DIR})
include_directories(${IPEX_CPU_CPP_ROOT})
include_directories(${IPEX_CPU_CPP_ROOT}/aten)
include_directories(${IPEX_CPU_CPP_ROOT}/utils)

include_directories(${IPEX_JIT_CPP_ROOT})
include_directories(${IPEX_UTLIS_CPP_ROOT})

include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn/include)
include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn/third_party/oneDNN/include)
# TODO: once llga is merged into oneDNN, use oneDNN directly as the third_party instead of using that inside llga
# include_directories(${PROJECT_SOURCE_DIR}/build/third_party/mkl-dnn/include)
# include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/mkl-dnn/include)

include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/include)

# Set installed MKL install dir
include_directories(${MKL_INSTALL_DIR}/include)

# Set installed PyTorch dir
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

if(DEFINED PYBIND11_CL_FLAGS)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND11_CL_FLAGS}")
else()
  message(FATAL_ERROR, "Cannot get pybind11 abi compiler flags")
endif()

# Set Python include dir
if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR, "Cannot find installed Python head file directory")
endif()

# generate cpu isa sources
set(IPEX_CPU_CPP_ISA_SRCS_GEN)
set(IPEX_CPU_CPP_ISA_SRCS_ORIGIN)
include(${IPEX_PROJECT_TOP_DIR}/cmake/cpu/IsaCodegen.cmake)

set(IPEX_CPU_CPP_SRCS)
set(IPEX_CPU_CPP_UTILS_SRCS)
set(IPEX_CPU_CPP_QUANTIZATION_SRCS)
set(IPEX_CPU_CPP_AUTOCAST_SRCS)
set(IPEX_CPU_CPP_ATEN_SRCS)
set(IPEX_CPU_CPP_DYNDISP_SRCS)
set(IPEX_CPU_CPP_ISA_SRCS)
set(IPEX_CPU_CPP_IDEEP_SRCS)
set(IPEX_CPU_CPP_RUNTIME_SRCS)
set(IPEX_CPU_CPP_TOOLKIT_SRCS)

set(IPEX_JIT_CPP_SRCS)
set(IPEX_UTLIS_CPP_SRCS)

# foreach(file_path ${IPEX_CPU_CPP_ISA_SRCS_GEN})
#   message(${file_path})
# endforeach()

add_subdirectory(${IPEX_CPU_CPP_ROOT}/aten)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/autocast)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/dyndisp)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/ideep)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/isa)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/runtime)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/toolkit)
add_subdirectory(${IPEX_CPU_CPP_ROOT}/utils)

add_subdirectory(${IPEX_JIT_CPP_ROOT} jit_cpu)
add_subdirectory(${IPEX_UTLIS_CPP_ROOT} csrc_utlis)

# Compile code with pybind11
set(IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_DYNDISP_SRCS} ${IPEX_CPU_CPP_ISA_SRCS_GEN} ${IPEX_CPU_CPP_UTILS_SRCS} ${IPEX_CPU_CPP_QUANTIZATION_SRCS} ${IPEX_JIT_CPP_SRCS}
    ${IPEX_CPU_CPP_ISA_SRCS} ${IPEX_CPU_CPP_IDEEP_SRCS} ${IPEX_CPU_CPP_AUTOCAST_SRCS} ${IPEX_CPU_CPP_ATEN_SRCS} ${IPEX_CPU_CPP_RUNTIME_SRCS} ${IPEX_CPU_CPP_TOOLKIT_SRCS} ${IPEX_UTLIS_CPP_SRCS})

list(REMOVE_ITEM IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_ISA_SRCS_ORIGIN})

add_library(${PLUGIN_NAME_CPU} SHARED ${IPEX_CPU_CPP_SRCS})

include(${IPEX_PROJECT_TOP_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_NATIVE_CSRC_FILES ${IPEX_CPU_CPP_ROOT}/**.[ch] ${IPEX_CPU_CPP_ROOT}/**.[ch]pp 
      ${IPEX_JIT_CPP_ROOT}/**.[ch] ${IPEX_JIT_CPP_ROOT}/**.[ch]pp ${IPEX_UTLIS_CPP_ROOT}/**.[ch] ${IPEX_UTLIS_CPP_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CPU_NATIVE_CSRC COMMAND clang-format -i -style=file ${ALL_NATIVE_CSRC_FILES})
  add_dependencies(${PLUGIN_NAME_CPU} CL_FORMAT_CPU_NATIVE_CSRC)
endif()

link_directories(${PYTORCH_INSTALL_DIR}/lib)
add_dependencies(${PLUGIN_NAME_CPU} dnnl_graph)
# If Graph Compiler is built, then it should link to its LLVM dependencies,
# and not the LLVM symbols exposed by PyTorch.
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC dnnl_graph)
if (DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND})
  get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB dnnl_graphcompiler_llvm_lib INTERFACE_LINK_LIBRARIES)
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC graphcompiler ${DNNL_GRAPHCOMPILER_LLVM_LIB})
  if (NOT MSVC)
    # Workaround to ensure that Graph Compiler would use LLVM symbols of libLLVM.
    get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB_EXCLUDE dnnl_graphcompiler_llvm_lib_exclude_string INTERFACE_LINK_LIBRARIES)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs=${DNNL_GRAPHCOMPILER_LLVM_LIB_EXCLUDE}")
  endif()
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
