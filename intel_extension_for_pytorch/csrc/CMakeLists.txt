# CMAKE FILE for libintel-ext-pt-python.so
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(IPEX CXX C)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME_COMMON_LIB intel-ext-pt-python)

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
set(IPEX_COMMON_CPP_ROOT "${IPEX_PROJECT_TOP_DIR}/intel_extension_for_pytorch/csrc")

# ---[ Build flags
include(${IPEX_PROJECT_TOP_DIR}/cmake/common_lib/BuildOptions.cmake)

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

# includes
include_directories(${IPEX_PROJECT_TOP_DIR})
include_directories(${IPEX_CPU_CPP_ROOT})
include_directories(${IPEX_CPU_CPP_ROOT}/aten)
include_directories(${IPEX_CPU_CPP_ROOT}/utils)

include_directories(${IPEX_JIT_CPP_ROOT})
include_directories(${IPEX_COMMON_CPP_ROOT})

FILE(GLOB _CPU_SRCS version.cpp initialization.cpp)

set(IPEX_COMMON_CPP_SRCS)

set(IPEX_COMMON_CPU_SRCS)
add_subdirectory(cpu)

LIST(APPEND IPEX_COMMON_CPP_SRCS ${_CPU_SRCS} ${IPEX_COMMON_CPU_SRCS})

# Pass to parent
# set(IPEX_COMMON_CPP_SRCS ${IPEX_COMMON_CPP_SRCS} PARENT_SCOPE)

add_library(${PLUGIN_NAME_COMMON_LIB} SHARED ${IPEX_COMMON_CPP_SRCS})

target_link_libraries(${PLUGIN_NAME_COMMON_LIB} PUBLIC ${CMAKE_INSTALL_PREFIX}/libintel-ext-pt-cpu.so)