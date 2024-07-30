set(TARGET xetla_kernels)

# ------------------------------------------------------------------------------------------------
# Configure GPU architecture
set(XETLA_AVAILABLE_ARCHS xe_hpc xe_hpg xe_lpg)
string(REPLACE "," ";" USE_XETLA ${USE_XETLA})
# set true value to all available archs ; set false value to empty list
if("${USE_XETLA}")  # A quoted string always evaluates to false unless: The string's value is one of the true constants
  # Note: CMake < 3.20 will evaluate `("${USE_XETLA}")` to True if USE_XETLA is mtl-p, and then be replaced by the whole
  # XETLA_AVAILABLE_ARCHS. Therefore cmake >= 3.20 should be used to partially enable xetla kernels for development.
  set(USE_XETLA ${XETLA_AVAILABLE_ARCHS})
elseif(NOT USE_XETLA) # if(<variable>): True if given a variable that is defined to a value that is not a false constant
  set(USE_XETLA "")
endif()

set(AOT_REGEX_XE_HPC "(xe-hpc.*|pvc|bmg-*|lnl-*|xe2-.*)")
set(AOT_REGEX_XE_HPG "(xe-hpg.*|ats-m150|acm-.*|dg2-.*)")
set(AOT_REGEX_XE_LPG "(xe-lpg.*|mtl-.*|0x7d55|0x7dd5|0x7d57|0x7dd7)")

foreach(used_arch IN LISTS USE_XETLA)
  foreach(available_arch IN LISTS XETLA_AVAILABLE_ARCHS)
    if (used_arch STREQUAL available_arch)
      string(TOUPPER "${used_arch}" arch_upper)
      set(USE_XETLA_${arch_upper} ON)
      message(STATUS "XeTLA: Found arch from list: ${arch_upper}")
      break()
    endif()
  endforeach()
  if (NOT USE_XETLA_${arch_upper})
    message(FATAL_ERROR "Unexpected XeTLA architecture: ${used_arch}")
  endif()
endforeach()

set(XETLA_USE_AOT_DEVLIST)
if (USE_AOT_DEVLIST)
  # dispatch aot targets to xetla's gpu_arch
  string(REPLACE "," ";" L_USE_AOT_DEVLIST ${USE_AOT_DEVLIST})
  foreach(aot_target IN LISTS L_USE_AOT_DEVLIST)
    set(UNKNOWN_AOT_TARGET ON)
    foreach(arch IN LISTS XETLA_AVAILABLE_ARCHS)
      string(TOUPPER ${arch} ARCH)
      if(aot_target MATCHES "^${AOT_REGEX_${ARCH}}$")  # Requires full match: e.g. pvc-vg can not match pvc
        set(UNKNOWN_AOT_TARGET OFF)
        list(APPEND XETLA_USE_AOT_DEVLIST_${ARCH} "${aot_target}")
        message(STATUS "XeTLA: match arch from AOT: ${aot_target} - ${ARCH}")
      endif()
    endforeach()
    if(UNKNOWN_AOT_TARGET)
      message(FATAL_ERROR "XeTLA: unknown AOT target: ${aot_target}")
    endif()
  endforeach()

  foreach(arch IN LISTS XETLA_AVAILABLE_ARCHS)
    string(TOUPPER ${arch} ARCH)
    if("${XETLA_USE_AOT_DEVLIST_${ARCH}}" STREQUAL "")
      # Disable implementations for architectures not in USE_AOT_DEVLIST
      set(USE_XETLA_${ARCH} OFF)
    else()
      # To comma separated string as aot device list
      string(REPLACE ";" "," XETLA_USE_AOT_DEVLIST_${ARCH} "${XETLA_USE_AOT_DEVLIST_${ARCH}}")
    endif()
    message(STATUS "XeTLA: XETLA_USE_AOT_DEVLIST_${ARCH}: ${XETLA_USE_AOT_DEVLIST_${ARCH}}")
  endforeach()
  
  # General USE_AOT_DEVLIST for XeTLA libs
  # Use XETLA_USE_AOT_DEVLIST_${ARCH} if possible
  set(XETLA_USE_AOT_DEVLIST "${USE_AOT_DEVLIST}")
  if (USE_XETLA_XE_HPC)  # Temporary fix as AOT fails of try to compile XE_HPC kernels for ats-m150 etc
    message(STATUS "XeTLA: XE_HPC suppress other aot targets")
    set(XETLA_USE_AOT_DEVLIST "${USE_AOT_DEVLIST_XE_HPC}")
  elseif(USE_XETLA_XE_HPG) # Temporary fix as AOT fails of try to compile XE_HPG kernels for mtl-p etc
    message(STATUS "XeTLA: XE_HPG suppress other aot targets")
    set(XETLA_USE_AOT_DEVLIST "${USE_AOT_DEVLIST_XE_HPG}")
  endif()
  message(STATUS "XeTLA: XETLA_USE_AOT_DEVLIST: ${XETLA_USE_AOT_DEVLIST}")
endif()

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
SET(BLA_VENDOR INTEL10_64LP)

set(XETLA_INCLUDE_DIR "${USE_XETLA_SRC}/include")
find_file(XETLA_HEADER NAMES xetla.hpp PATHS ${XETLA_INCLUDE_DIR} PATH_SUFFIXES NO_DEFAULT_PATH)
if(NOT XETLA_HEADER)
  message(FATAL_ERROR "XeTLA header ${XETLA_HEADER} not found in ${XETLA_INCLUDE_DIR}")
endif()

set(XETLA_USED_ARCHS)
foreach(available_arch IN LISTS XETLA_AVAILABLE_ARCHS)
  string(TOUPPER ${available_arch} arch_upper)
  if(USE_XETLA_${arch_upper})
    list(APPEND XETLA_USED_ARCHS ${available_arch})
  endif()
endforeach()


include(compile_library.cmake)

# ------------------------------------------------------------------------------------------------
# compile GEMM libraries of FMHA and HGEMM
FILE(GLOB_RECURSE kernels_src *.cpp)
include(SDP/fmha_forward_configure.cmake)
fmha_forward_configure("${XETLA_USED_ARCHS}")

# Xetla kernel apply different offline compiler options than framework.
# If framework link XeTLA as static lib, link options would conflict now.
# Temporarily adopting shared lib, will fix it soon.
add_library_with_options(xetla_gemm TRUE "${XETLA_USE_AOT_DEVLIST}" ${kernels_src} ${FMHA_FORWARD_KERNEL_SRCS})

# ------------------------------------------------------------------------------------------------
# compile libraries for INT4, one library for one configuration
include(GEMM/GEMM_int4_configure.cmake)
GEMM_int4_configure("${XETLA_USED_ARCHS}")

# ------------------------------------------------------------------------------------------------
# link the libraries to the interface library\
add_library(${TARGET} INTERFACE)
list(APPEND XETLA_LIBS xetla_gemm)
list(APPEND XETLA_LIBS ${GEMM_INT4_LIBS})
message(STATUS "XETLA_LIBS: ${XETLA_LIBS}")
foreach(LIB ${XETLA_LIBS})
  target_link_libraries(${TARGET} INTERFACE ${LIB})
endforeach()

set(XETLA_LIBS ${XETLA_LIBS} PARENT_SCOPE)
