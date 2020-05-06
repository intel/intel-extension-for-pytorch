cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_ipex)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME _torch_ipex)

set(TORCH_IPEX_C_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_ipex/csrc")

# Common dependencies

include(cmake/Dependencies.cmake)

include(cmake/GPU.cmake)
