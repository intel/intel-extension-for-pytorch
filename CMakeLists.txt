cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(intel_extension_for_pytorch VERSION ${CMAKE_PROJECT_VERSION})

set(IPEX_ROOT_DIR ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${IPEX_ROOT_DIR}/cmake/Modules)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# relocatable binary with automatically rpath of depend libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)

include(${IPEX_ROOT_DIR}/cmake/xpu/Options.cmake)

if (BUILD_STATS)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "time -v")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "time -v")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "time -v")
endif()

include(${IPEX_ROOT_DIR}/cmake/xpu/BuildFlags.cmake)
include(${IPEX_ROOT_DIR}/cmake/xpu/DPCPP.cmake)

# Need the torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(${IPEX_ROOT_DIR}/csrc)
add_subdirectory(${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)

function (print_configuration_summary)
  # Fetch definitions of intel-ext-pt-gpu as main target
  get_target_property(IPEX_NATIVE_DEFINITIONS intel-ext-pt-gpu COMPILE_DEFINITIONS)

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Platform              : ${PYTHON_PLATFORM_INFO}")

    message(STATUS "  Target name           : ${LIB_NAME}")
    message(STATUS "  Target version        : ${CMAKE_PROJECT_VERSION}")
    message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")

    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

    message(STATUS "  SYCL Language version : ${SYCL_LANGUAGE_VERSION}")
    message(STATUS "  SYCL Compiler version : ${SYCL_COMPILER_VERSION}")
    message(STATUS "  SYCL Driver version   : ${SYCL_DRIVER_VERSION}")
    message(STATUS "  SYCL LevelZero version: ${SYCL_LEVEL_ZERO_VERSION}")

    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")
    message(STATUS "  Compile definitions   : ${IPEX_NATIVE_DEFINITIONS}")
    message(STATUS "  SYCL Kernel flags     : ${IPEX_SYCL_KERNEL_FLAGS}")
    message(STATUS "  SYCL Linker options   : ${IPEX_SYCL_LINKER_FLAGS}")

    message(STATUS "  Intel SYCL instance ID: ${SYCL_IMPLEMENTATION_ID}")
    message(STATUS "  Intel SYCL include    : ${SYCL_INCLUDE_DIR}")
    message(STATUS "  Intel SYCL library    : ${SYCL_LIBRARY_DIR}")

    message(STATUS "  LevelZero include     : ${LevelZero_INCLUDE_DIR}")
    message(STATUS "  LevelZero library     : ${LevelZero_LIBRARY}")

    message(STATUS "  OpenCL include        : ${OpenCL_INCLUDE_DIR}")
    message(STATUS "  OpenCL library        : ${OpenCL_LIBRARY}")

    message(STATUS "  BUILD_WITH_XPU        : ${BUILD_WITH_XPU}")
    message(STATUS "  USE_ONEMKL            : ${USE_ONEMKL}")
  if (USE_ONEMKL)
    message(STATUS "  ONEMKL include        : ${ONEMKL_INCLUDE_DIR}")
  endif(USE_ONEMKL)

    message(STATUS "  USE_CHANNELS_LAST_1D  : ${USE_CHANNELS_LAST_1D}")

    message(STATUS "  USE_PERSIST_STREAM    : ${USE_PERSIST_STREAM}")
    message(STATUS "  USE_PRIMITIVE_CACHE   : ${USE_PRIMITIVE_CACHE}")
    message(STATUS "  USE_QUEUE_BARRIER     : ${USE_QUEUE_BARRIER}")
    message(STATUS "  USE_SCRATCHPAD_MODE   : ${USE_SCRATCHPAD_MODE}")
    message(STATUS "  USE_MULTI_CONTEXT     : ${USE_MULTI_CONTEXT}")

    message(STATUS "  BUILD_STATS           : ${BUILD_STATS}")
    message(STATUS "  BUILD_BY_PER_KERNEL   : ${BUILD_BY_PER_KERNEL}")
    message(STATUS "  BUILD_STRIPPED_BIN    : ${BUILD_STRIPPED_BIN}")

  if(NOT BUILD_BY_PER_KERNEL AND USE_AOT_DEVLIST)
    message(STATUS "  USE_AOT_DEVLIST       : ${USE_AOT_DEVLIST}")
  else()
    message(STATUS "  USE_AOT_DEVLIST       : OFF")
  endif()

    message(STATUS "  BUILD_INTERNAL_DEBUG  : ${BUILD_INTERNAL_DEBUG}")
    message(STATUS "  BUILD_NO_CLANGFORMAT  : ${BUILD_NO_CLANGFORMAT}")
    message(STATUS "  BUILD_SEPARATE_OPS    : ${BUILD_SEPARATE_OPS}")
    message(STATUS "  USE_PROFILER          : ${USE_PROFILER}")
    message(STATUS "  USE_SYCL_ASSERT       : ${USE_SYCL_ASSERT}")
    message(STATUS "  BUILD_SIMPLE_TRACE    : ${BUILD_SIMPLE_TRACE}")
    message(STATUS "  BUILD_JIT_QUANTIZATION_SAVE : ${BUILD_JIT_QUANTIZATION_SAVE}")
    message(STATUS "")

endfunction()

print_configuration_summary()
