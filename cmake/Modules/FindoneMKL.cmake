#===============================================================================
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# The following are set after configuration is done:
#  ONEMKL_FOUND            : set to true if oneMKL is found.
#  ONEMKL_INCLUDE_DIR      : path to oneMKL include dir.
#  ONEMKL_GPU_LIBS         : list of oneMKL libraries for GPU
#  ONEMKL_CPU_LIBS         : list of oneMKL libraries for CPU
#===============================================================================

if(ONEMKL_FOUND)
  return()
endif()

set(ONEMKL_FOUND OFF)
set(ONEMKL_INCLUDE_DIR)
set(ONEMKL_GPU_LIBS)
set(ONEMKL_CPU_LIBS)

set(mkl_root_hint)

# install mkl-include and mkl-static for CPU build
function (install_mkl_packages)
  set(REQ_MKL_VERSION 2021.0.0)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install "mkl-include>=${REQ_MKL_VERSION}"
      RESULT_VARIABLE mkl_iret COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install --no-deps "mkl-static>=${REQ_MKL_VERSION}"
      RESULT_VARIABLE mkl_sret COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip show mkl-include -f RESULT_VARIABLE mkl_show_ret
      OUTPUT_VARIABLE mkl_show_out COMMAND_ERROR_IS_FATAL ANY)
  if (${mkl_iret} OR ${mkl_sret} OR ${mkl_show_ret})
    message(FATAL_ERROR "Failed to install/fetch mkl-include or mkl-static by pip!")
  endif()
  string(REGEX MATCH "Location: *([^\t\r\n]+)" mkl_prefix ${mkl_show_out})
  set(mkl_prefix ${CMAKE_MATCH_1})
  string(REGEX MATCH " *([^\t\r\n]*include[/\\]mkl.h)" mkl_h_rel ${mkl_show_out})
  set(mkl_h_rel ${CMAKE_MATCH_1})
  get_filename_component(mkl_inc "${mkl_prefix}/${mkl_h_rel}/" DIRECTORY)
  get_filename_component(mkl_root_hint "${mkl_inc}/../" ABSOLUTE)
  set(mkl_root_hint ${mkl_root_hint} PARENT_SCOPE)
endfunction()

function(get_mkl_from_env_var)
  if(DEFINED ENV{MKLROOT})
    set(mkl_root_hint $ENV{MKLROOT})
  elseif(DEFINED ENV{MKL_ROOT})
    set(mkl_root_hint $ENV{MKL_ROOT})
  elseif(MKL_ROOT)
    set(mkl_root_hint ${MKL_ROOT})
  else()
    message(WARNING "Please set oneMKL root path by MKLROOT, or MKL_ROOT for IPEX build.")
    return()
  endif()
  set(mkl_root_hint ${mkl_root_hint} PARENT_SCOPE)
endfunction()

# IPEX XPU lib always use the dynamic linker for oneMKL lib if USE_ONEMKL is ON and oneMKL is available.
# IPEX CPU lib always download and install mkl-static lib and use static linker for mkl-static lib.
# IPEX CPU lib can manual config to use the dynamic link for oneMKL lib.
if(BUILD_MODULE_TYPE STREQUAL "GPU")
  get_mkl_from_env_var()
else()
  if(BUILD_WITH_XPU)
    get_mkl_from_env_var()
  else()
    if(BUILD_STATIC_ONEMKL)
      message(STATUS "Download and install mkl-include and mkl-static for IPEX CPU build automatically.")
      install_mkl_packages()
    else()
      get_mkl_from_env_var()
    endif()
  endif()
endif()

# Try to find Intel MKL header
find_file(MKL_HEADER NAMES mkl.h PATHS ${mkl_root_hint}
    PATH_SUFFIXES include NO_DEFAULT_PATH)

if(NOT MKL_HEADER)
  message(FATAL_ERROR "Intel oneMKL not found. No oneMKL support ${MKL_HEADER} -- ${mkl_root_hint}")
endif()
get_filename_component(ONEMKL_INCLUDE_DIR "${MKL_HEADER}/.." ABSOLUTE)

if(BUILD_STATIC_ONEMKL)
  set(LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  if (NOT WINDOWS)
    set(LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(LIB_SUFFIX ".lib")
  endif()
else()
  set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  if (NOT WINDOWS)
    set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(LIB_SUFFIX "_dll.lib")
  endif()
endif()

if (NOT WINDOWS)
  set(MKL_THREAD "${LIB_PREFIX}mkl_gnu_thread${LIB_SUFFIX}")
else()
  set(MKL_THREAD "${LIB_PREFIX}mkl_intel_thread${LIB_SUFFIX}")
endif()
find_library(MKL_LIB_THREAD ${MKL_THREAD} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_THREAD)
  message(FATAL_ERROR "oneMKL library ${MKL_THREAD} not found")
endif()

set(MKL_LP64 "${LIB_PREFIX}mkl_intel_lp64${LIB_SUFFIX}")
find_library(MKL_LIB_LP64 ${MKL_LP64} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_LP64)
  message(FATAL_ERROR "oneMKL library ${MKL_LP64} not found")
endif()

set(MKL_CORE "${LIB_PREFIX}mkl_core${LIB_SUFFIX}")
find_library(MKL_LIB_CORE ${MKL_CORE} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_CORE)
  message(FATAL_ERROR "oneMKL library ${MKL_CORE} not found")
endif()

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  set(MKL_SYCL "${LIB_PREFIX}mkl_sycl${LIB_SUFFIX}")
  find_library(MKL_LIB_SYCL ${MKL_SYCL} HINTS ${mkl_root_hint}
      PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
  if(NOT MKL_LIB_SYCL)
    message(FATAL_ERROR "oneMKL library ${MKL_SYCL} not found")
  endif()
endif()

# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
set(START_GROUP)
set(END_GROUP)
if(BUILD_STATIC_ONEMKL)
  set(START_GROUP "-Wl,--start-group")
  set(END_GROUP "-Wl,--end-group")
else()
  if(BUILD_MODULE_TYPE STREQUAL "CPU")
    set(START_GROUP "-Wl,--push-state,--no-as-needed")
    set(END_GROUP "-Wl,--pop-state")
  endif()
endif()

set(ONEMKL_CPU_LIBS ${START_GROUP} ${MKL_LIB_LP64} ${MKL_LIB_CORE} ${MKL_LIB_THREAD} ${END_GROUP})

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  set(ONEMKL_GPU_LIBS ${START_GROUP} ${MKL_LIB_LP64} ${MKL_LIB_CORE} ${MKL_LIB_THREAD} ${MKL_LIB_SYCL} ${END_GROUP})
endif()

set(ONEMKL_FOUND ON)
message("Intel oneMKL found.")
