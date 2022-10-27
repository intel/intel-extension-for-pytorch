# CMAKE FILE for intel-ext-pt-python.so
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(IPEX CXX C)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME_PYTHON intel-ext-pt-python)

# Setup project top directory.
set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../../")

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

# ---[ Build flags
include(${IPEX_PROJECT_TOP_DIR}/cmake/python/BuildOptions.cmake)

# includes
include_directories(${IPEX_PROJECT_TOP_DIR})
include_directories(${IPEX_CPU_CPP_ROOT})
include_directories(${IPEX_CPU_CPP_ROOT}/aten)
include_directories(${IPEX_CPU_CPP_ROOT}/utils)

include_directories(${IPEX_JIT_CPP_ROOT})
include_directories(${IPEX_UTLIS_CPP_ROOT})

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

set(IPEX_PYTHON_CPP_SRCS)
set(IPEX_PYTHON_CPP_CPU_SRCS)
add_subdirectory(cpu)

set(IPEX_PYTHON_CPP_SRCS ${IPEX_PYTHON_CPP_CPU_SRCS})

add_library(${PLUGIN_NAME_PYTHON} SHARED ${IPEX_PYTHON_CPP_SRCS})

target_link_directories(${PLUGIN_NAME_PYTHON} PUBLIC ${IPEX_INSTALL_LIBDIR})
target_link_libraries(${PLUGIN_NAME_PYTHON} PUBLIC intel-ext-pt-cpu)
