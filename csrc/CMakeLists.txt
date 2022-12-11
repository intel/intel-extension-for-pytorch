set(IPEX_CSRC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/gpu)
  print_xpu_config_summary()
endif()

if(BUILD_MODULE_TYPE STREQUAL "CPU")
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/cpu)
  print_cpu_config_summary()
endif()

# CMake config file for the external project.
configure_file(
        ${IPEX_ROOT_DIR}/cmake/IntelExtensionPytorchConfig.cmake.in
        ${PROJECT_BINARY_DIR}/IntelExtensionPytorchConfig.cmake
        @ONLY)

install(FILES
        ${PROJECT_BINARY_DIR}/IntelExtensionPytorchConfig.cmake
        DESTINATION share/cmake/IntelExtensionPytorch)

if(BUILD_MODULE_TYPE STREQUAL "CPPSDK")
  install(PROGRAMS "${IPEX_ROOT_DIR}/scripts/cppsdk/install_c++_sdk.sh.in"
    DESTINATION scripts)

  set(RUN_SCRIPT "libintel-ext-pt-${CMAKE_PROJECT_VERSION}.run")
  if(GLIBCXX_USE_CXX11_ABI)
    set(RUN_SCRIPT "libintel-ext-pt-cxx11-abi-${CMAKE_PROJECT_VERSION}.run")
  endif()
  install(PROGRAMS "${IPEX_ROOT_DIR}/scripts/cppsdk/intel-ext-pt-cpu.run.in"
    RENAME ${RUN_SCRIPT}
    DESTINATION scripts)

  set(CPU_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}intel-ext-pt-cpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
  install(FILES ${CPU_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  if(BUILD_WITH_XPU)
    set(GPU_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}intel-ext-pt-gpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    install(FILES ${GPU_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  set(CPACK_PACKAGE_NAME "${IPEX_PROJ_NAME}")
  if(GLIBCXX_USE_CXX11_ABI)
    set(CPACK_PACKAGE_NAME "${IPEX_PROJ_NAME}-cxx11-abi")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_GENERATOR "TXZ")
    set(CPACK_SOURCE_GENERATOR "TXZ")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SOURCE_GENERATOR "ZIP")
  endif()

  if(CPACK_CONFIG_FILE)
    set(CPACK_OUTPUT_CONFIG_FILE "${CPACK_CONFIG_FILE}")
  endif()
  set(CPACK_RESOURCE_FILE_LICENSE "${IPEX_ROOT_DIR}/LICENSE")

  if(CPACK_OUTPUT_DIR)
    set(CPACK_PACKAGE_DIRECTORY "${CPACK_OUTPUT_DIR}")
  endif()

  include(CPack)
endif()
