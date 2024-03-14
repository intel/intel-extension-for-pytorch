## Included by CMakeLists
if(Options_cmake_included)
    return()
endif()
set(Options_cmake_included true)

# The options to build xpu
include(CMakeDependentOption)
# General options:

option(BUILD_WITH_CPU "Build CPU backend implementation" ON)
option(BUILD_WITH_XPU "Build XPU backend implementation" OFF)
option(BUILD_NO_CLANGFORMAT "Build without force clang-format" OFF)
option(BUILD_STATS "Count statistics for each component during build process" OFF)
option(BUILD_STRIPPED_BIN "Strip all symbols after build" OFF)
option(BUILD_STATIC_ONEMKL "build ipex cpu/gpu with static mkl." ON)

if (BUILD_STATS)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "time -v")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "time -v")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "time -v")
endif()

if(BUILD_MODULE_TYPE STREQUAL "CPU")
  include(${IPEX_ROOT_DIR}/cmake/cpu/Options.cmake)
endif()

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  include(${IPEX_ROOT_DIR}/cmake/gpu/Options.cmake)
endif()

function (print_config_summary)
  message(STATUS "")

  message(STATUS "******** General Summary ********")
  message(STATUS "General:")
  message(STATUS "  CMake version         : ${CMAKE_VERSION}")
  message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  Platform              : ${PYTHON_PLATFORM_INFO}")
  message(STATUS "  Target name           : ${IPEX_PROJ_NAME}")
  message(STATUS "  Target version        : ${CMAKE_PROJECT_VERSION}")
  message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")

  message(STATUS "Options:")
  message(STATUS "  BUILD_WITH_CPU        : ${BUILD_WITH_CPU}")
  message(STATUS "  BUILD_WITH_XPU        : ${BUILD_WITH_XPU}")
  message(STATUS "  BUILD_NO_CLANGFORMAT  : ${BUILD_NO_CLANGFORMAT}")
  message(STATUS "  BUILD_STATS           : ${BUILD_STATS}")
  message(STATUS "  BUILD_STRIPPED_BIN    : ${BUILD_STRIPPED_BIN}")

  message(STATUS "")
endfunction()
