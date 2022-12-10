## Included by CMakeLists
if(BUILD_OPTIONS_cmake_included)
    return()
endif()
set(BUILD_OPTIONS_cmake_included true)

if(BUILD_MODULE_TYPE STREQUAL "CPU")
  include(${IPEX_ROOT_DIR}/cmake/cpu/BuildFlags.cmake)
endif()

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  include(${IPEX_ROOT_DIR}/cmake/xpu/BuildFlags.cmake)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
string(REGEX MATCH "-D_GLIBCXX_USE_CXX11_ABI=([0-9]+)" torch_cxx11 ${TORCH_CXX_FLAGS})
set(GLIBCXX_USE_CXX11_ABI ${CMAKE_MATCH_1})
if(BUILD_WITH_XPU AND NOT GLIBCXX_USE_CXX11_ABI)
  message(FATAL_ERROR "Must set _GLIBCXX_USE_CXX11_ABI=1 for XPU build, but not is ${GLIBCXX_USE_CXX11_ABI}!")
endif()
