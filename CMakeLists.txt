cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(${IPEX_PROJ_NAME} VERSION ${CMAKE_PROJECT_VERSION})

set(IPEX_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(IPEX_CPU_ROOT_DIR ${IPEX_ROOT_DIR}/csrc/cpu)
set(IPEX_GPU_ROOT_DIR ${IPEX_ROOT_DIR}/csrc/gpu)
set(IPEX_FRONT_END_ROOT_DIR ${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)
list(APPEND CMAKE_MODULE_PATH ${IPEX_ROOT_DIR}/cmake/Modules)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# relocatable binary with automatically rpath of depend libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)

# Need the torch package
find_package(Torch REQUIRED)
if(NOT EXISTS ${TORCH_INSTALL_PREFIX})
  message(FATAL_ERROR "Can NOT find torch install path at ${TORCH_INSTALL_PREFIX}!")
endif()

include(${IPEX_ROOT_DIR}/cmake/BuildFlags.cmake)
include(${IPEX_ROOT_DIR}/cmake/Options.cmake)

add_subdirectory(${IPEX_ROOT_DIR}/csrc)
add_subdirectory(${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)
