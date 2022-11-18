set(IPEX_CSRC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_WITH_XPU)
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/gpu)
endif()
