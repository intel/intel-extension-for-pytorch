# Try to find the pybind11 library and headers.
#  pybind11_FOUND        - system has pybind11
#  pybind11_INCLUDE_DIRS - the pybind11 include directory

set(pybind11_INCLUDE_DIRS)
set(torch_pybind11_root_hint)

if(TORCH_FOUND)
        set(torch_pybind11_root_hint ${TORCH_INCLUDE_DIRS})
elseif(DEFINED ENV{TORCH_ROOT})
        set(torch_pybind11_root_hint "$ENV{TORCH_ROOT}/include")
endif()

find_path(pybind11_INCLUDE_DIR
        NAMES pybind11/pybind11.h
        HINTS ${torch_pybind11_root_hint}
        DOC "The directory where pybind11 includes reside"
)

set(pybind11_INCLUDE_DIRS ${pybind11_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(pybind11
        FOUND_VAR pybind11_FOUND
        REQUIRED_VARS pybind11_INCLUDE_DIR
)

mark_as_advanced(pybind11_FOUND)
