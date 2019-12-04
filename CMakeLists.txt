cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Common dependencies
include(cmake/Dependencies.cmake)

IF(USE_SYCL)
  include(cmake/GPU.cmake)
ELSE()
  include(cmake/CPU.cmake)
ENDIF()
