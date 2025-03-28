cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX_CPP_TEST)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake/Modules)

# specify the C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# relocatable binary with automatically rpath of depend libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)

set(CPU_CPP_TEST_NAME ipex_cpp_test)

# Setup project top directory.
set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../../")

set(THIRD_PARTY_ROOT "${IPEX_PROJECT_TOP_DIR}/third_party")

# Need the torch package
find_package(Torch REQUIRED)
if(NOT EXISTS ${TORCH_INSTALL_PREFIX})
  message(FATAL_ERROR "Can NOT find torch install path at ${TORCH_INSTALL_PREFIX}!")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-Bsymbolic-functions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-Bsymbolic-functions")

# Set the include dir
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${IPEX_PROJECT_TOP_DIR})
include_directories(${IPEX_PROJECT_TOP_DIR}/csrc/include)

link_directories(${PYTORCH_INSTALL_DIR}/lib)
# search the lib directory for gtest
link_directories(${CPP_TEST_BUILD_DIR}/lib)

# Add the Test Files
set(IPEX_CPP_TEST_SOURCES test_runtime_api.cpp test_dyndisp_and_isa_api.cpp)

add_executable(${CPU_CPP_TEST_NAME} ${IPEX_CPP_TEST_SOURCES})

if(USE_SYSTEM_GTEST)
  find_package(GTest REQUIRED)
  target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC GTest::gtest GTest::gtest_main)
else()
  # add gtest cmake path
  add_subdirectory(${THIRD_PARTY_ROOT}/googletest ${CPP_TEST_BUILD_DIR}/third_party/googletest EXCLUDE_FROM_ALL)

  # Link GTest
  target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC gtest_main)
  target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC gtest)

  target_include_directories(${CPU_CPP_TEST_NAME} PRIVATE ${THIRD_PARTY_ROOT}/googletest/googletest/include)
endif()

set(BUILD_STATIC_ONEMKL ON)
find_package(oneMKL QUIET)
if (ONEMKL_FOUND)
  target_include_directories(${CPU_CPP_TEST_NAME} PUBLIC ${ONEMKL_INCLUDE_DIR})
endif()

target_link_directories(${CPU_CPP_TEST_NAME} PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/)

# Link Pytorch
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC ${TORCH_LIBRARIES})

# Link IPEX
target_link_libraries(${CPU_CPP_TEST_NAME} PUBLIC intel-ext-pt-cpu)

install(TARGETS ${CPU_CPP_TEST_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
