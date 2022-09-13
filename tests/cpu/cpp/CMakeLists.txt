cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX_CPP_TEST)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# specify the C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(CPU_CPP_TEST_NAME ipex_cpp_test)

# Setup project top directory.
set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../../")

set(THIRD_PARTY_ROOT "${IPEX_PROJECT_TOP_DIR}/third_party")

set(_CXX11_ABI_FLAG 0)
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(_CXX11_ABI_FLAG 1)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-Bsymbolic-functions")

# Set the include dir
include_directories(${MKL_INSTALL_DIR}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
include_directories(${THIRD_PARTY_ROOT}/googletest/googletest/include)
include_directories(${IPEX_PROJECT_TOP_DIR})

link_directories(${PYTORCH_INSTALL_DIR}/lib)
# search the lib directory for gtest
link_directories(${CPP_TEST_BUILD_DIR}/lib)

# add gtest cmake path
add_subdirectory(${THIRD_PARTY_ROOT}/googletest ${CPP_TEST_BUILD_DIR}/third_party/googletest)

# Add the Test Files
set(IPEX_CPP_TEST_SOURCES test_runtime_api.cpp test_dyndisp_and_isa_api.cpp)

add_executable(${CPU_CPP_TEST_NAME} ${IPEX_CPP_TEST_SOURCES})

# Link GTest
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC gtest)

# Link Pytorch
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

# Link IPEX
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/libintel-ext-pt-cpu.so)
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/libintel-ext-pt-python.so)