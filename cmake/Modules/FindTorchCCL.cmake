# - Try to find torch-ccl
#
# The following are set after configuration is done:
#  TORCHCCL_FOUND          : set to true if oneCCL is found.
#  TORCHCCL_INCLUDE_DIR    : path to oneCCL include dir.
#  TORCHCCL_LIBRARIES      : list of libraries for oneCCL
#
# The following variables are used:
#  TORCHCCL_USE_NATIVE_ARCH : Whether native CPU instructions should be used in TORCHCCL. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF (NOT TORCHCCL_FOUND)
SET(TORCHCCL_FOUND OFF)

SET(TORCHCCL_LIBRARIES)
SET(TORCHCCL_INCLUDE_DIR)

SET(TORCHCCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/torch_ccl")

ADD_SUBDIRECTORY(${TORCHCCL_ROOT})
IF(NOT TARGET torch_ccl)
    MESSAGE(FATAL_ERROR "Failed to include torch_ccl target")
ENDIF()
GET_TARGET_PROPERTY(INCLUDE_DIRS torch_ccl INCLUDE_DIRECTORIES)
SET(TORCHCCL_INCLUDE_DIR ${INCLUDE_DIRS})
SET(TORCHCCL_LIBRARIES torch_ccl)

ENDIF(NOT TORCHCCL_FOUND)
