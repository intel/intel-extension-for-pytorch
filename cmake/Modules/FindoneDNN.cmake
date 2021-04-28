# - Try to find oneDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  ONEDNN_FOUND          : set to true if oneDNN is found.
#  ONEDNN_INCLUDE_DIR    : path to oneDNN include dir.
#  ONEDNN_LIBRARIES      : list of libraries for oneDNN
#
# The following variables are used:
#  ONEDNN_USE_NATIVE_ARCH : Whether native CPU instructions should be used in ONEDNN. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF (NOT ONEDNN_FOUND)
SET(ONEDNN_FOUND OFF)

SET(ONEDNN_LIBRARIES)
SET(ONEDNN_INCLUDE_DIR)

SET(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
SET(ONEDNN_DIR "oneDNN")
SET(ONEDNN_ROOT "${THIRD_PARTY_DIR}/${ONEDNN_DIR}")

FIND_PATH(ONEDNN_INCLUDE_DIR dnnl.hpp dnnl.h PATHS ${ONEDNN_ROOT} PATH_SUFFIXES include)
IF (NOT ONEDNN_INCLUDE_DIR)
  EXECUTE_PROCESS(
    COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init ${ONEDNN_DIR}
    WORKING_DIRECTORY ${THIRD_PARTY_DIR})
  FIND_PATH(ONEDNN_INCLUDE_DIR dnnl.hpp dnnl.h PATHS ${ONEDNN_ROOT} PATH_SUFFIXES include)
ENDIF(NOT ONEDNN_INCLUDE_DIR)

IF (NOT ONEDNN_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "oneDNN source files not found!")
ENDIF(NOT ONEDNN_INCLUDE_DIR)

IF (USE_PRIMITIVE_CACHE)
  # Disable oneDNN primitive cache if already cached by FRAMEWORK
  SET(DNNL_ENABLE_PRIMITIVE_CACHE FALSE CACHE BOOL "oneDNN sycl primitive cache" FORCE)
ELSE()
  SET(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "oneDNN sycl primitive cache" FORCE)
ENDIF()

ADD_DEFINITIONS(-DDNNL_USE_DPCPP_USM)
IF(BUILD_NO_L0_ONEDNN)
  MACRO(unset_onednn_l0)
    UNSET(DNNL_WITH_LEVEL_ZERO)
    MESSAGE(WARNING "LevelZero is being forced to disable in oneDNN!")
  ENDMACRO()
  VARIABLE_WATCH(DNNL_WITH_LEVEL_ZERO unset_onednn_l0)
ENDIF()

IF(ONEDNN_USE_NATIVE_ARCH)  # Disable HostOpts in oneDNN unless ONEDNN_USE_NATIVE_ARCH is set.
  SET(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING "" FORCE)
ELSE()
  IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(DNNL_ARCH_OPT_FLAGS "-msse4" CACHE STRING "" FORCE)
  ELSE()
    SET(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
  ENDIF()
ENDIF()

# FIXME: Set threading to THREADPOOL to bypass issues due to not found TBB or OMP.
# NOTE: We will not use TBB, but we cannot enable OMP. We build whole oneDNN by DPC++
# compiler which only provides the Intel iomp. But oneDNN cmake scripts only try to
# find the iomp in ICC compiler, which caused a build fatal error here.
SET(DNNL_CPU_RUNTIME "THREADPOOL" CACHE STRING "oneDNN cpu backend" FORCE)
SET(DNNL_GPU_RUNTIME "SYCL" CACHE STRING "oneDNN gpu backend" FORCE)
SET(DNNL_BUILD_TESTS FALSE CACHE BOOL "build with oneDNN tests" FORCE)
SET(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "build with oneDNN examples" FORCE)
SET(DNNL_ENABLE_CONCURRENT_EXEC TRUE CACHE BOOL "multi-thread primitive execution" FORCE)
SET(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

ADD_SUBDIRECTORY(${ONEDNN_ROOT} oneDNN EXCLUDE_FROM_ALL)
IF(NOT TARGET dnnl)
  MESSAGE(FATAL_ERROR "Failed to include oneDNN target")
ENDIF(NOT TARGET dnnl)

IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-uninitialized)
  TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-strict-overflow)
  TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-error=strict-overflow)
ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)

TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-tautological-compare)
GET_TARGET_PROPERTY(DNNL_INCLUDES dnnl INCLUDE_DIRECTORIES)
LIST(APPEND ONEDNN_INCLUDE_DIR ${DNNL_INCLUDES})
LIST(APPEND ONEDNN_LIBRARIES dnnl)
SET(ONEDNN_FOUND ON)
MESSAGE(STATUS "Found oneDNN: TRUE")

ENDIF(NOT ONEDNN_FOUND)
