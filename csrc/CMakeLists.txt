set(IPEX_CSRC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/gpu)
  print_xpu_config_summary()
endif()

if(BUILD_MODULE_TYPE STREQUAL "CPU")
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/cpu)
  print_cpu_config_summary()
endif()
