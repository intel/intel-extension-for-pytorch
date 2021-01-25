# - Try to find Vtune
#

IF (NOT ITT_FOUND)
SET(ITT_FOUND OFF)

SET(ITT_LIBRARIES)
SET(ITT_INCLUDE_DIR)

set(oneapi_root_hint)
if(DEFINED INTELONEAPIROOT)
    set(oneapi_root_hint ${INTELONEAPIROOT})
elseif(DEFINED ENV{INTELONEAPIROOT})
    set(oneapi_root_hint $ENV{INTELONEAPIROOT})
endif()
SET(ITT_ROOT "${oneapi_root_hint}/vtune/latest")

find_path(ITT_INCLUDE_DIR
        NAMES ittnotify.h
        HINTS ${ITT_ROOT} $ENV{ITT_ROOT}
        PATH_SUFFIXES include)

# Find ITT
find_library(ITT_LIBRARY
          NAMES ittnotify
          HINTS ${ITT_ROOT} $ENV{ITT_ROOT}
          PATH_SUFFIXES lib/x64 lib lib64)

IF (NOT ITT_INCLUDE_DIR)
    MESSAGE(WARNING "ITT headers not found!")
ENDIF(NOT ITT_INCLUDE_DIR)

IF (NOT ITT_LIBRARY)
    MESSAGE(WARNING "ITT libraries not found!")
ENDIF(NOT ITT_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        ITT
        FOUND_VAR ITT_FOUND
        REQUIRED_VARS ITT_INCLUDE_DIR ITT_LIBRARY)

ENDIF(NOT ITT_FOUND)
