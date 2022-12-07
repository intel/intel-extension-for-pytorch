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

set(_CXX11_ABI_FLAG 0)
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(_CXX11_ABI_FLAG 1)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI_FLAG}")
