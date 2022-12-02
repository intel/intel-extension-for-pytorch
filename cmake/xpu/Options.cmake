## Included by CMakeLists
if(Options_GPU_cmake_included)
    return()
endif()
set(Options_GPU_cmake_included true)

# The options to build xpu
include(CMakeDependentOption)
option(USE_ONEMKL "Use oneMKL BLAS" ON)
option(USE_CHANNELS_LAST_1D "Use channels last 1d" ON)
option(USE_PERSIST_STREAM "Use persistent oneDNN stream" ON)
option(USE_SCRATCHPAD_MODE "Use oneDNN scratchpad mode" ON)
option(USE_PRIMITIVE_CACHE "Cache oneDNN primitives by FRAMEWORK, instead of oneDNN itself" OFF)
option(USE_QUEUE_BARRIER "Use queue submit_barrier, otherwise use dummy kernel" ON)
option(USE_MULTI_CONTEXT "Create DPC++ runtime context per device" ON)
set(USE_AOT_DEVLIST "" CACHE STRING "Set device list for AOT build")
option(USE_PROFILER "USE XPU Profiler in build." ON)
option(USE_SYCL_ASSERT "Enables assert in sycl kernel" OFF)
option(USE_ITT_ANNOTATION "Enables ITT annotation in sycl kernel" OFF)

option(BUILD_BY_PER_KERNEL "Build by DPC++ per_kernel option (exclusive with USE_AOT_DEVLIST)" OFF)
option(BUILD_INTERNAL_DEBUG "Use internal debug code path" OFF)
option(BUILD_SEPARATE_OPS "Build each operator in separate library" OFF)
option(BUILD_SIMPLE_TRACE "Build simple trace for each registered operator" ON)
set(BUILD_OPT_LEVEL "" CACHE STRING "Add build option -Ox, accept values: 0/1")
option(BUILD_JIT_QUANTIZATION_SAVE "Support jit quantization model save and load" OFF)

include(${IPEX_ROOT_DIR}/cmake/xpu/BuildFlags.cmake)
include(${IPEX_ROOT_DIR}/cmake/xpu/DPCPP.cmake)

function (print_xpu_config_summary)
  # Fetch configurations of intel-ext-pt-gpu
  get_target_property(NATIVE_DEFINITIONS intel-ext-pt-gpu COMPILE_DEFINITIONS)
  get_target_property(ONEDNN_INCLUDE_DIR intel-ext-pt-gpu ONEDNN_INCLUDE_DIR)
  get_target_property(USE_ONEMKL intel-ext-pt-gpu USE_ONEMKL)
  get_target_property(ONEMKL_INCLUDE_DIR intel-ext-pt-gpu ONEMKL_INCLUDE_DIR)
  get_target_property(ONEDPL_INCLUDE_DIR intel-ext-pt-gpu ONEDPL_INCLUDE_DIR)

    message(STATUS "")

    message(STATUS "******** Summary on XPU ********")
    message(STATUS "General:")

    message(STATUS "  C compiler            : ${CMAKE_C_COMPILER}")

    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

    message(STATUS "  SYCL Language version : ${SYCL_LANGUAGE_VERSION}")
    message(STATUS "  SYCL Compiler version : ${SYCL_COMPILER_VERSION}")
    message(STATUS "  SYCL Driver version   : ${SYCL_DRIVER_VERSION}")
    message(STATUS "  SYCL LevelZero version: ${SYCL_LEVEL_ZERO_VERSION}")

    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")
    message(STATUS "  Compile definitions   : ${NATIVE_DEFINITIONS}")
    message(STATUS "  SYCL Kernel flags     : ${IPEX_SYCL_KERNEL_FLAGS}")
    message(STATUS "  SYCL Linker options   : ${IPEX_SYCL_LINKER_FLAGS}")

    message(STATUS "  Intel SYCL instance ID: ${SYCL_IMPLEMENTATION_ID}")
    message(STATUS "  Intel SYCL include    : ${SYCL_INCLUDE_DIR}")
    message(STATUS "  Intel SYCL library    : ${SYCL_LIBRARY_DIR}")

    message(STATUS "  LevelZero include     : ${LevelZero_INCLUDE_DIR}")
    message(STATUS "  LevelZero library     : ${LevelZero_LIBRARY}")

    message(STATUS "  OpenCL include        : ${OpenCL_INCLUDE_DIR}")
    message(STATUS "  OpenCL library        : ${OpenCL_LIBRARY}")

    message(STATUS "  Torch include         : ${TORCH_INCLUDE_DIRS}")

    message(STATUS "  oneDNN include        : ${ONEDNN_INCLUDE_DIR}")
  if (USE_ONEMKL)
    message(STATUS "  oneMKL include        : ${ONEMKL_INCLUDE_DIR}")
  endif(USE_ONEMKL)
    message(STATUS "  oneDPL include        : ${ONEDPL_INCLUDE_DIR}")

    message(STATUS "Options:")
    message(STATUS "  USE_ONEMKL            : ${USE_ONEMKL}")
    message(STATUS "  USE_CHANNELS_LAST_1D  : ${USE_CHANNELS_LAST_1D}")
    message(STATUS "  USE_PERSIST_STREAM    : ${USE_PERSIST_STREAM}")
    message(STATUS "  USE_PRIMITIVE_CACHE   : ${USE_PRIMITIVE_CACHE}")
    message(STATUS "  USE_QUEUE_BARRIER     : ${USE_QUEUE_BARRIER}")
    message(STATUS "  USE_SCRATCHPAD_MODE   : ${USE_SCRATCHPAD_MODE}")
    message(STATUS "  USE_MULTI_CONTEXT     : ${USE_MULTI_CONTEXT}")
    message(STATUS "  USE_PROFILER          : ${USE_PROFILER}")
    message(STATUS "  USE_SYCL_ASSERT       : ${USE_SYCL_ASSERT}")
    message(STATUS "  USE_ITT_ANNOTATION    : ${USE_ITT_ANNOTATION}")

  if(NOT BUILD_BY_PER_KERNEL AND USE_AOT_DEVLIST)
    message(STATUS "  USE_AOT_DEVLIST       : ${USE_AOT_DEVLIST}")
  else()
    message(STATUS "  USE_AOT_DEVLIST       : OFF")
  endif()

    message(STATUS "  BUILD_BY_PER_KERNEL   : ${BUILD_BY_PER_KERNEL}")
    message(STATUS "  BUILD_INTERNAL_DEBUG  : ${BUILD_INTERNAL_DEBUG}")
    message(STATUS "  BUILD_SEPARATE_OPS    : ${BUILD_SEPARATE_OPS}")
    message(STATUS "  BUILD_SIMPLE_TRACE    : ${BUILD_SIMPLE_TRACE}")
    message(STATUS "  BUILD_JIT_QUANTIZATION_SAVE : ${BUILD_JIT_QUANTIZATION_SAVE}")

    message(STATUS "")
endfunction()
