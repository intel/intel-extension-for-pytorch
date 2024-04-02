# - Try to find oneCCL
#
# The following are set after configuration is done:
#  ONECCL_FOUND          : set to true if oneCCL is found.
#  ONECCL_INCLUDE_DIRS   : path to oneCCL include dir.
#  ONECCL_LIBRARIES      : list of libraries for oneCCL
#
# and the following imported targets:
#
#   oneCCL

IF (NOT ONECCL_FOUND)
SET(ONECCL_FOUND OFF)
SET(ONECCL_LIBRARIES)
SET(ONECCL_INCLUDE_DIRS)

SET(ONECCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/oneCCL")

IF(BUILD_NO_ONECCL_PACKAGE)
    ADD_SUBDIRECTORY(${ONECCL_ROOT} oneCCL EXCLUDE_FROM_ALL)
ELSE()
    ADD_SUBDIRECTORY(${ONECCL_ROOT} build)
ENDIF()

IF(NOT TARGET ccl)
    MESSAGE(FATAL_ERROR "Failed to find oneCCL target")
ENDIF()
add_library(oneCCL ALIAS ccl)

GET_TARGET_PROPERTY(INCLUDE_DIRS oneCCL INCLUDE_DIRECTORIES)
SET(ONECCL_INCLUDE_DIRS ${INCLUDE_DIRS})
SET(ONECCL_LIBRARIES oneCCL)

find_package_handle_standard_args(oneCCL FOUND_VAR ONECCL_FOUND REQUIRED_VARS ONECCL_LIBRARIES ONECCL_INCLUDE_DIRS)

ENDIF(NOT ONECCL_FOUND)