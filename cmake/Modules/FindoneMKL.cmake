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

if (ONEMKL_FOUND)
  return()
endif()

set(ONEMKL_FOUND OFF)

set(mkl_root_hint)
if(DEFINED ENV{MKL_DPCPP_ROOT})
  set(mkl_root_hint $ENV{MKL_DPCPP_ROOT})
elseif(DEFINED ENV{MKLROOT})
  set(mkl_root_hint $ENV{MKLROOT})
elseif(DEFINED ENV{MKL_ROOT})
  set(mkl_root_hint $ENV{MKL_ROOT})
elseif(MKL_ROOT)
  set(mkl_root_hint ${MKL_ROOT})
elseif(NOT BUILD_WITH_XPU)
  # install mkl-include and mkl-static for CPU build
  set(REQ_MKL_VERSION 2021.0.0)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install "mkl-include>=${REQ_MKL_VERSION}" RESULT_VARIABLE mkl_iret)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install --no-deps "mkl-static>=${REQ_MKL_VERSION}" RESULT_VARIABLE mkl_sret)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip show mkl-include -f RESULT_VARIABLE mkl_show_ret OUTPUT_VARIABLE mkl_show_out)
  if (${mkl_iret} OR ${mkl_sret} OR ${mkl_show_ret})
    message(FATAL_ERROR "Failed to install/fetch mkl-include or mkl-static by pip!")
  endif()
  string(REGEX MATCH "Location: *([//\/a-zA-Z0-9\.\-]+)" mkl_prefix ${mkl_show_out})
  set(mkl_prefix ${CMAKE_MATCH_1})
  string(REGEX MATCH " *([//\/a-zA-Z0-9\.\-]+include/mkl.h)" mkl_h_rel ${mkl_show_out})
  set(mkl_h_rel ${CMAKE_MATCH_1})
  get_filename_component(mkl_inc "${mkl_prefix}/${mkl_h_rel}/" DIRECTORY)
  get_filename_component(mkl_root_hint "${mkl_inc}/../" ABSOLUTE)
endif()

# Try to find Intel MKL DPCPP header
find_file(MKL_HEADER NAMES mkl.h PATHS ${mkl_root_hint}
    PATH_SUFFIXES include NO_DEFAULT_PATH)

if(NOT MKL_HEADER)
  message(FATAL_ERROR "Intel oneMKL not found. No oneMKL support ${MKL_HEADER} -- ${mkl_root_hint}")
endif()
get_filename_component(ONEMKL_INCLUDE_DIR "${MKL_HEADER}/.." ABSOLUTE)

if(BUILD_STATIC_ONEMKL)
  set(LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

set(MKL_THREAD "${LIB_PREFIX}mkl_gnu_thread${LIB_SUFFIX}")
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

set(MKL_SYCL "${LIB_PREFIX}mkl_sycl${LIB_SUFFIX}")
find_library(MKL_LIB_SYCL ${MKL_SYCL} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_LIB_SYCL)
  message(FATAL_ERROR "oneMKL library ${MKL_SYCL} not found")
endif()

set(START_GROUP)
set(END_GROUP)
if(BUILD_STATIC_ONEMKL)
set(START_GROUP "-Wl,--start-group")
set(END_GROUP "-Wl,--end-group")
endif()

list(APPEND ONEMKL_CPU_LIBS ${START_GROUP} ${MKL_LIB_LP64} ${MKL_LIB_CORE} ${MKL_LIB_THREAD} ${END_GROUP})
list(APPEND ONEMKL_GPU_LIBS ${START_GROUP} ${MKL_LIB_LP64} ${MKL_LIB_CORE} ${MKL_LIB_THREAD} ${MKL_LIB_SYCL} ${END_GROUP})

set(ONEMKL_FOUND ON)
message("Intel oneMKL found.")
