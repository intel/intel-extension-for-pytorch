# - Try to find Vtune
#

if (NOT ITT_FOUND)
SET (ITT_FOUND OFF)

SET (ITT_LIBRARIES)
SET (ITT_INCLUDE_DIR)

if(DEFINED INTELONEAPIROOT)
    SET(ITT_ROOT "${INTELONEAPIROOT}/vtune/latest")
elseif(DEFINED ENV{INTELONEAPIROOT})
    SET(ITT_ROOT "$ENV{INTELONEAPIROOT}/vtune/latest")
else()
    SET(ITT_ROOT "${PROJECT_SOURCE_DIR}/third_party/ittapi")
    FIND_PATH(ITT_INCLUDE_DIR ittnotify.h PATHS ${ITT_ROOT} PATH_SUFFIXES include)
    IF (NOT ITT_INCLUDE_DIR)
        EXECUTE_PROCESS(
            COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init ittapi
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/third_party)
    ENDIF(NOT ITT_INCLUDE_DIR)
    IF(NOT CMAKE_BUILD_TYPE)
      SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
          "Choose the type of build, options are: Debug Release"
          FORCE)
    ENDIF(NOT CMAKE_BUILD_TYPE)
    ADD_SUBDIRECTORY(${ITT_ROOT} ${CMAKE_BINARY_DIR}/ittapi)
endif()

FIND_PATH(ITT_INCLUDE_DIR
          NAMES ittnotify.h
          HINTS ${ITT_ROOT} $ENV{ITT_ROOT}
          PATH_SUFFIXES include)

# Find ITT
FIND_LIBRARY(ITT_LIBRARY
             NAMES ittnotify
             HINTS ${ITT_ROOT} $ENV{ITT_ROOT}
             PATH_SUFFIXES lib/x64 lib lib64)

if (NOT ITT_INCLUDE_DIR)
    MESSAGE(WARNING "ITT headers not found!")
endif (NOT ITT_INCLUDE_DIR)

if (NOT ITT_LIBRARY)
    SET(ITT_LIBRARY ittnotify)
endif (NOT ITT_LIBRARY)

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        ITT
        FOUND_VAR ITT_FOUND
        REQUIRED_VARS ITT_INCLUDE_DIR ITT_LIBRARY)

endif (NOT ITT_FOUND)
