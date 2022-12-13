## Included by CMakeLists
if(Options_CPU_cmake_included)
    return()
endif()
set(Options_CPU_cmake_included true)

# The options to build xpu
include(CMakeDependentOption)
option(IPEX_DISP_OP "output the extension operators name for debug purpose" OFF)
cmake_dependent_option(BUILD_STATIC_ONEMKL "Static link with oneMKL" OFF "BUILD_WITH_XPU" ON)

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
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  Compile definitions   : ${CPU_NATIVE_DEFINITIONS}")
    message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")
    message(STATUS "  Link libraries        : ${CPU_LINK_LIBRARIES}")

    message(STATUS "  Torch include         : ${TORCH_INCLUDE_DIRS}")

    message(STATUS "  oneDNN include        : ${ONEDNN_INCLUDE_DIR}")
    message(STATUS "  oneMKL include        : ${ONEMKL_INCLUDE_DIR}")

    message(STATUS "Options:")
    message(STATUS "  BUILD_STATIC_ONEMKL   : ${BUILD_STATIC_ONEMKL}")
    message(STATUS "  IPEX_DISP_OP          : ${IPEX_DISP_OP}")

    message(STATUS "")
    message(STATUS "********************************")
endfunction()
