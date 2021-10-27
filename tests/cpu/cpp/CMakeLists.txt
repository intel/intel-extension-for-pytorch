cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX_CPP_TEST)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# specify the C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(TEST_NAME ipex_cpp_test)

IF(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
  message(FATAL_ERROR "Error: Please set {CMAKE_PREFIX_PATH} to the anaconda path")
ENDIF()

find_package(GTest REQUIRED)
IF(GTEST_FOUND)
  message(STATUS "GTEST_FOUND")
ELSE()
  message(FATAL_ERROR "GTEST NOT FOUND")
ENDIF()

# Set the include dir
include_directories($ENV{CMAKE_PREFIX_PATH}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
include_directories(${PROJECT_DIR})

link_directories($ENV{CMAKE_PREFIX_PATH}/lib)
link_directories(${PYTORCH_INSTALL_DIR}/lib)

add_executable(${TEST_NAME} test_runtime_api.cpp)

# Link GTest
target_link_libraries(${TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${TEST_NAME} PUBLIC gtest)

# Link Pytorch
target_link_libraries(${TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

# Link IPEX
target_link_libraries(${TEST_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/libintel-ext-pt-cpu.so)
