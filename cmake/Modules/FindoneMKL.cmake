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
if(MKL_DPCPP_ROOT)
    set(mkl_root_hint ${MKL_DPCPP_ROOT})
# elseif(MKL_ROOT)
#    set(mkl_root_hint ${MKL_ROOT})
elseif(DEFINED ENV{MKL_DPCPP_ROOT})
    set(mkl_root_hint $ENV{MKL_DPCPP_ROOT})
elseif(DEFINED ENV{MKLROOT})
    set(mkl_root_hint $ENV{MKLROOT})
elseif(NOT BUILD_WITH_XPU)
    # install mkl-include and mkl-static for CPU build
    set(REQ_MKL_VERSION 2021.0.0)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    if(NOT Python_Interpreter_FOUND)
      message(FATAL_ERROR "Cannot find Python interpreter!")
    endif()
    execute_process(COMMAND ${Python_EXECUTABLE} -m pip install "mkl-include>=${REQ_MKL_VERSION}" RESULT_VARIABLE mkl_iret)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pip install --no-deps "mkl-static>=${REQ_MKL_VERSION}" RESULT_VARIABLE mkl_sret)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pip show mkl-include -f RESULT_VARIABLE mkl_show_ret OUTPUT_VARIABLE mkl_show_out)
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
find_file(MKL_HEADER
    NAMES mkl.h
    PATHS
        ${mkl_root_hint}
    PATH_SUFFIXES
        include
    NO_DEFAULT_PATH)

if(NOT MKL_HEADER)
  message(FATAL_ERROR "Intel oneMKL not found. No oneMKL support ${MKL_HEADER} -- ${mkl_root_hint}")
  return()
endif()
get_filename_component(ONEMKL_INCLUDE_DIR "${MKL_HEADER}/.." ABSOLUTE)

if(NOT BUILD_WITH_XPU)
    find_library(ONEMKL_CORE_LIBRARY
        "libmkl_core.a"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_CORE_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_core not found")
    endif()

    find_library(ONEMKL_GNU_THREAD_LIBRARY
        "libmkl_gnu_thread.a"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_GNU_THREAD_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_gnu_thread not found")
    endif()

    find_library(ONEMKL_LP64_LIBRARY
        "libmkl_intel_lp64.a"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_LP64_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_intel_lp64 not found")
    endif()

    list(APPEND ONEMKL_CPU_LIBS ${ONEMKL_LP64_LIBRARY} ${ONEMKL_CORE_LIBRARY} ${ONEMKL_GNU_THREAD_LIBRARY})
    set(ONEMKL_FOUND ON)
    message("Intel oneMKL found.")

else() # BUILD_WITH_XPU

    find_library(ONEMKL_SYCL_LIBRARY
        NAMES "mkl_sycl"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_SYCL_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_sycl not found")
    endif()

    find_library(ONEMKL_ILP64_LIBRARY
        NAMES "mkl_intel_ilp64"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_ILP64_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_intel_ilp64 not found")
    endif()

    find_library(ONEMKL_CORE_LIBRARY
        NAMES "mkl_core"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_CORE_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_core not found")
    endif()

    find_library(ONEMKL_SEQUENTIAL_LIBRARY
        NAMES "mkl_sequential"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_SEQUENTIAL_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_sequential not found")
    endif()

    find_library(ONEMKL_LP64_LIBRARY
        NAMES "mkl_intel_lp64"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_LP64_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_intel_lp64 not found")
    endif()

    find_library(ONEMKL_GNU_THREAD_LIBRARY
        NAMES "mkl_gnu_thread"
        HINTS
            ${mkl_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_GNU_THREAD_LIBRARY)
        message(FATAL_ERROR "oneMKL library mkl_gnu_thread not found")
    endif()

    list(APPEND ONEMKL_CPU_LIBS ${ONEMKL_LP64_LIBRARY} ${ONEMKL_CORE_LIBRARY} ${ONEMKL_GNU_THREAD_LIBRARY})
    list(APPEND ONEMKL_GPU_LIBS ${ONEMKL_SYCL_LIBRARY} ${ONEMKL_ILP64_LIBRARY} ${ONEMKL_CORE_LIBRARY} ${ONEMKL_SEQUENTIAL_LIBRARY})
    set(ONEMKL_FOUND ON)
    message(STATUS "Intel oneMKL found.")
endif()
