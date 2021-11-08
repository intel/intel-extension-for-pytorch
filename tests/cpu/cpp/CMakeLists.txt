cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX_CPP_TEST)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# specify the C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(TEST_NAME ipex_cpp_test)
set(THIRD_PARTY_ROOT "${PROJECT_DIR}/third_party")

# Set the include dir
include_directories(${PYTORCH_INSTALL_DIR}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
include_directories(${THIRD_PARTY_ROOT}/googletest/googletest/include)
include_directories(${PROJECT_DIR})

link_directories(${PYTORCH_INSTALL_DIR}/lib)
# search the lib directory for gtest
link_directories(${CPP_TEST_BUILD_DIR}/lib)

# add gtest cmake path
add_subdirectory(${THIRD_PARTY_ROOT}/googletest ${CPP_TEST_BUILD_DIR}/third_party/googletest)

# Add the Test Files
set(IPEX_CPP_TEST_SOURCES test_runtime_api.cpp)

add_executable(${TEST_NAME} ${IPEX_CPP_TEST_SOURCES})

# Link GTest
target_link_libraries(${TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${TEST_NAME} PUBLIC gtest)

# Link Pytorch
target_link_libraries(${TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

# Link IPEX
target_link_libraries(${TEST_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/libintel-ext-pt-cpu.so)
