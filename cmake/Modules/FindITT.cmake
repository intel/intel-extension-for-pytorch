# - Try to find Vtune
#

if (NOT ITT_FOUND)
set (ITT_FOUND OFF)

set (ITT_LIBRARIES)
set (ITT_INCLUDE_DIR)

set(oneapi_root_hint)
if(DEFINED INTELONEAPIROOT)
    set(oneapi_root_hint ${INTELONEAPIROOT})
elseif(DEFINED ENV{INTELONEAPIROOT})
    set(oneapi_root_hint $ENV{INTELONEAPIROOT})
endif()
set (ITT_ROOT "${oneapi_root_hint}/vtune/latest")

find_path(ITT_INCLUDE_DIR
        NAMES ittnotify.h
        HINTS ${ITT_ROOT} $ENV{ITT_ROOT}
        PATH_SUFFIXES include)

# Find ITT
find_library(ITT_LIBRARY
          NAMES ittnotify
          HINTS ${ITT_ROOT} $ENV{ITT_ROOT}
          PATH_SUFFIXES lib/x64 lib lib64)

if (NOT ITT_INCLUDE_DIR)
    MESSAGE(WARNING "ITT headers not found!")
endif (NOT ITT_INCLUDE_DIR)

if (NOT ITT_LIBRARY)
    MESSAGE(WARNING "ITT libraries not found!")
endif (NOT ITT_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        ITT
        FOUND_VAR ITT_FOUND
        REQUIRED_VARS ITT_INCLUDE_DIR ITT_LIBRARY)

endif (NOT ITT_FOUND)
