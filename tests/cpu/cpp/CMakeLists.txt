cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX_CPP_TEST)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake/Modules)

# specify the C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# relocatable binary with automatically rpath of depend libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)

set(CPU_CPP_TEST_NAME ipex_cpp_test)

# Setup project top directory.
set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../../")

set(THIRD_PARTY_ROOT "${IPEX_PROJECT_TOP_DIR}/third_party")

# Need the torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-Bsymbolic-functions")

# Set the include dir
include_directories(${PYTORCH_INSTALL_DIR}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
include_directories(${THIRD_PARTY_ROOT}/googletest/googletest/include)
include_directories(${IPEX_PROJECT_TOP_DIR})

link_directories(${PYTORCH_INSTALL_DIR}/lib)
# search the lib directory for gtest
link_directories(${CPP_TEST_BUILD_DIR}/lib)

# add gtest cmake path
add_subdirectory(${THIRD_PARTY_ROOT}/googletest ${CPP_TEST_BUILD_DIR}/third_party/googletest EXCLUDE_FROM_ALL)

# Add the Test Files
set(IPEX_CPP_TEST_SOURCES test_runtime_api.cpp test_dyndisp_and_isa_api.cpp)

add_executable(${CPU_CPP_TEST_NAME} ${IPEX_CPP_TEST_SOURCES})

set(BUILD_STATIC_ONEMKL ON)
find_package(oneMKL QUIET)
if (ONEMKL_FOUND)
  target_include_directories(${CPU_CPP_TEST_NAME} PUBLIC ${ONEMKL_INCLUDE_DIR})
endif()

target_link_directories(${CPU_CPP_TEST_NAME} PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/)

# Link GTest
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC gtest_main)
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC gtest)

# Link Pytorch
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${TORCH_LIBRARIES})

# Link IPEX
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${CMAKE_INSTALL_PREFIX}/lib/libintel-ext-pt-cpu.so)

install(TARGETS ${CPU_CPP_TEST_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
