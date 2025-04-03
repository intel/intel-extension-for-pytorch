## Included by CMakeLists
if(BUILD_OPTIONS_cmake_included)
    return()
endif()
set(BUILD_OPTIONS_cmake_included true)

if(BUILD_MODULE_TYPE STREQUAL "CPU")
  include(${IPEX_ROOT_DIR}/cmake/cpu/BuildFlags.cmake)
endif()

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  include(${IPEX_ROOT_DIR}/cmake/gpu/BuildFlags.cmake)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Since 2016 Debian start using RUNPATH instead of normally RPATH, which gave the annoy effect that
# allow LD_LIBRARY_PATH to override dynamic linking path. Depends on intention of linking priority,
# change below for best outcome: disable, using RPATH, enable, using RUNPATH
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
