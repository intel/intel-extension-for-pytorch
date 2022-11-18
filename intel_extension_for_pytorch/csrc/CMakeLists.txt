# CMAKE FILE for intel-ext-pt-python.so
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(IPEX CXX C)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME_PYTHON intel-ext-pt-python)

# Setup project top directory.
# TODO: CPU: why need cpu to be 'set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../../")'
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

# ---[ Build flags
# TODO: CPU: include(${IPEX_PROJECT_TOP_DIR}/cmake/python/BuildOptions.cmake)
include(${IPEX_PROJECT_TOP_DIR}/cmake/BuildOptions.cmake)

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
# TODO: specific for cpu
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
set(IPEX_PYTHON_CPP_XPU_SRCS)

add_subdirectory(cpu)
# for xpu
if(BUILD_WITH_XPU)
  add_subdirectory(xpu)
endif()

list(APPEND IPEX_PYTHON_CPP_SRCS ${IPEX_PYTHON_CPP_CPU_SRCS})
list(APPEND IPEX_PYTHON_CPP_SRCS ${IPEX_PYTHON_CPP_XPU_SRCS})

add_library(${PLUGIN_NAME_PYTHON} SHARED ${IPEX_PYTHON_CPP_SRCS})

if(BUILD_WITH_XPU)
  message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
  target_include_directories(${PLUGIN_NAME_PYTHON} PUBLIC ${pybind11_INCLUDE_DIRS})

  target_compile_definitions(${PLUGIN_NAME_PYTHON} PUBLIC BUILD_WITH_XPU)

  # xpu specific BUILD_SIMPLE_TRACE
  if(BUILD_SIMPLE_TRACE)
    target_compile_definitions(${PLUGIN_NAME_PYTHON} PUBLIC BUILD_SIMPLE_TRACE)
  endif()

  # xpu specific BUILD_STRIPPED_BIN
  if(BUILD_STRIPPED_BIN)
    set_target_properties(${PLUGIN_NAME_PYTHON} PROPERTIES LINK_FLAGS_RELEASE -s)
  endif()
endif()

# TODO: XPU no need to set the search directory
# target_link_directories(${PLUGIN_NAME_PYTHON} PUBLIC ${IPEX_INSTALL_LIBDIR})

# TODO: temp comment the python so link cpu.so
# target_link_libraries(${PLUGIN_NAME_PYTHON} PUBLIC intel-ext-pt-cpu)

if(BUILD_WITH_XPU)
  target_link_libraries(${PLUGIN_NAME_PYTHON} PUBLIC intel-ext-pt-gpu)
  # TODO: install python so for xpu
  install(TARGETS intel-ext-pt-python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
