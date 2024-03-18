## Included by CMakeLists
if(Options_CPU_cmake_included)
    return()
endif()
set(Options_CPU_cmake_included true)

# The options to build cpu
include(CMakeDependentOption)

option(BUILD_LIBXSMM_VIA_CMAKE "Build LIBXSMM via CMake" ON)
option(USE_LIBXSMM "Enable LIBXSMM" ON)
if(WIN32)
  set(USE_LIBXSMM ON)
endif()

if(WIN32)
  set(USE_SHM OFF)
  set(USE_CCL OFF)
endif()


function (print_cpu_config_summary)
  # Fetch configurations of intel-ext-pt-cpu
  get_target_property(CPU_NATIVE_DEFINITIONS intel-ext-pt-cpu COMPILE_DEFINITIONS)
  get_target_property(ONEDNN_INCLUDE_DIR intel-ext-pt-cpu ONEDNN_INCLUDE_DIR)
  get_target_property(ONEMKL_INCLUDE_DIR intel-ext-pt-cpu ONEMKL_INCLUDE_DIR)
  get_target_property(CPU_LINK_LIBRARIES intel-ext-pt-cpu LINK_LIBRARIES)

    print_config_summary()
    message(STATUS "******** Summary on CPU ********")
    message(STATUS "General:")

    message(STATUS "  C compiler            : ${CMAKE_C_COMPILER}")

    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler ID       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

    message(STATUS "  CXX standard          : ${CMAKE_CXX_STANDARD}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  Compile definitions   : ${CPU_NATIVE_DEFINITIONS}")
    message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")
    message(STATUS "  Link libraries        : ${CPU_LINK_LIBRARIES}")

    message(STATUS "  Torch version         : ${Torch_VERSION}")
    message(STATUS "  Torch include         : ${TORCH_INCLUDE_DIRS}")

    message(STATUS "  oneDNN include        : ${ONEDNN_INCLUDE_DIR}")
    message(STATUS "  oneMKL include        : ${ONEMKL_INCLUDE_DIR}")

    message(STATUS "Options:")
    message(STATUS "  BUILD_STATIC_ONEMKL   : ${BUILD_STATIC_ONEMKL}")
    message(STATUS "  IPEX_DISP_OP          : ${IPEX_DISP_OP}")
    message(STATUS "  BUILD_XSMM_VIA_CMAKE  : ${BUILD_LIBXSMM_VIA_CMAKE}")
    message(STATUS "  USE_LIBXSMM           : ${USE_LIBXSMM}")
    message(STATUS "  USE_CCL               : ${USE_CCL}")
    message(STATUS "  USE_SHM               : ${USE_SHM}")
    message(STATUS "")
    message(STATUS "********************************")
endfunction()
