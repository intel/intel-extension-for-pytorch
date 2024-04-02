cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(${IPEX_PROJ_NAME} VERSION ${CMAKE_PROJECT_VERSION})

# Platform
set(LINUX False)
set(MACOS False)
set(WINDOWS False)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(LINUX True)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(WINDOWS True)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(MACOS True)
endif()

set(IPEX_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(IPEX_CSRC_ROOT_DIR ${IPEX_ROOT_DIR}/csrc)
set(IPEX_CPU_ROOT_DIR ${IPEX_ROOT_DIR}/csrc/cpu)
set(IPEX_GPU_ROOT_DIR ${IPEX_ROOT_DIR}/csrc/gpu)
set(IPEX_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
set(IPEX_FRONT_END_ROOT_DIR ${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)
list(APPEND CMAKE_MODULE_PATH ${IPEX_ROOT_DIR}/cmake/Modules)

set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# relocatable binary with automatically rpath of depend libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)

# Need the torch package
set(Torch_COMP_VERION "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}")
find_package(Torch ${Torch_COMP_VERION} REQUIRED)

if(NOT EXISTS ${TORCH_INSTALL_PREFIX})
  message(FATAL_ERROR "Can NOT find torch install path at ${TORCH_INSTALL_PREFIX}!")
endif()

if(NOT ${Torch_COMP_VERION} VERSION_EQUAL "${Torch_VERSION_MAJOR}.${Torch_VERSION_MINOR}")
  message(FATAL_ERROR "Not compatible Torch version ${Torch_VERSION} at ${TORCH_INSTALL_PREFIX}!\nTorch ${Torch_COMP_VERION} is needed!")
endif()

include(${IPEX_ROOT_DIR}/cmake/Options.cmake)
include(${IPEX_ROOT_DIR}/cmake/BuildFlags.cmake)

add_subdirectory(${IPEX_ROOT_DIR}/csrc)
add_subdirectory(${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)
