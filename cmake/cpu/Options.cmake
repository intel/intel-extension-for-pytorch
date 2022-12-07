## Included by CMakeLists
if(Options_CPU_cmake_included)
    return()
endif()
set(Options_CPU_cmake_included true)

# The options to build xpu
include(CMakeDependentOption)
option(IPEX_DISP_OP "output the extension operators name for debug purpose" OFF)

function (print_cpu_config_summary)
  # Fetch configurations of intel-ext-pt-cpu
  get_target_property(ONEDNN_INCLUDE_DIR intel-ext-pt-cpu ONEDNN_INCLUDE_DIR)
  get_target_property(ONEMKL_INCLUDE_DIR intel-ext-pt-cpu ONEMKL_INCLUDE_DIR)

    print_config_summary()
    message(STATUS "******** Summary on CPU ********")
    message(STATUS "General:")

    message(STATUS "  C compiler            : ${CMAKE_C_COMPILER}")

    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")

    message(STATUS "  oneDNN include        : ${ONEDNN_INCLUDE_DIR}")
    message(STATUS "  oneMKL include        : ${ONEMKL_INCLUDE_DIR}")

    message(STATUS "Options:")
    message(STATUS "  IPEX_DISP_OP          : ${IPEX_DISP_OP}")

    message(STATUS "")
    message(STATUS "********************************")
endfunction()
