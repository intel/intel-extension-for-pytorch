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
#  ONETRACE_FOUND            : set to true if onetrace is found.
#  ONETRACE_INCLUDE_DIRs     : path to onetrace include dirs.
#  ONETRACE_LIBS             : list of onetrace libraries.
#===============================================================================


if(ONETRACE_FOUND)
  return()
endif()

set(BUILD_FOR_IPEX ON)
set(ONETRACE_FOUND OFF)
set(ONETRACE_UTILS_DIR)
set(ONETRACE_INCLUDE_DIRS)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(ONETRACE_DIR "pti-gpu/tools/onetrace")
set(ONETRACE_ROOT "${THIRD_PARTY_DIR}/${ONETRACE_DIR}")

set(PTI_DIR "pti-gpu")
set(PTI_ROOT "${THIRD_PARTY_DIR}/${PTI_DIR}")
file(GLOB PTI_PATCHES "${PROJECT_SOURCE_DIR}/cmake/gpu/onetrace_patches/*.patch")
list(SORT PTI_PATCHES)

find_package(Git)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Cannot find Git executable!")
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} submodule sync ${PTI_DIR}
  WORKING_DIRECTORY ${THIRD_PARTY_DIR}
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND ${GIT_EXECUTABLE} submodule update --init ${PTI_DIR}
  WORKING_DIRECTORY ${THIRD_PARTY_DIR}
  COMMAND_ERROR_IS_FATAL ANY)
foreach (PTI_PATCH ${PTI_PATCHES})
  message(STATUS "Git am patch on oneTrace: ${PTI_PATCH}")
  set(ENV{GIT_COMMITTER_NAME} "IPEX_BUILDER")
  set(ENV{GIT_COMMITTER_EMAIL} "ipex_builder@example.com")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} am ${PTI_PATCH}
    WORKING_DIRECTORY ${PTI_ROOT}
    RESULT_VARIABLE git_am_result
    ERROR_VARIABLE git_am_error)
  if (NOT ${git_am_result} EQUAL 0)
    message(STATUS "Git am failed with error: ${git_am_error}")
    message(STATUS "Run $git am --abort automatically due to failure.")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} am --abort
      WORKING_DIRECTORY ${PTI_ROOT}
      COMMAND_ERROR_IS_FATAL ANY)
    message(FATAL_ERROR "Git am has been aborted. Please check repository status of ./third_party/pti-gpu/")
  endif ()
endforeach()
message(STATUS "Git am patches on oneTrace ... SUCCESS")

add_subdirectory(${ONETRACE_ROOT} oneTrace EXCLUDE_FROM_ALL)
set(ONETRACE_LIBRARY onetrace_tool)
if (NOT TARGET ${ONETRACE_LIBRARY})
  message(FATAL_ERROR "Failed to include onetrace_tool target")
endif()
list(APPEND ONETRACE_INCLUDE_DIRS "${ONETRACE_ROOT}")
list(APPEND ONETRACE_INCLUDE_DIRS "${ONETRACE_ROOT}/../utils")
list(APPEND ONETRACE_INCLUDE_DIRS "${ONETRACE_ROOT}/../../utils")
list(APPEND ONETRACE_INCLUDE_DIRS "${ONETRACE_ROOT}/../cl_tracer")
list(APPEND ONETRACE_INCLUDE_DIRS "${ONETRACE_ROOT}/../ze_tracer")

set(ONETRACE_FOUND ON)
set(BUILD_FOR_IPEX OFF)
message(STATUS "Found onetrace: TRUE")
