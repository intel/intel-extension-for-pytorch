set(IPEX_CSRC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_MODULE_TYPE STREQUAL "GPU")
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/gpu)
  print_xpu_config_summary()
endif()

if(BUILD_MODULE_TYPE STREQUAL "CPU")
  add_subdirectory(${IPEX_CSRC_ROOT_DIR}/cpu)
  print_cpu_config_summary()
endif()

set(LIBIPEX_COMP_LIST)
include(CMakePackageConfigHelpers)
# CMake config file for the external project.
configure_package_config_file(
  ${IPEX_ROOT_DIR}/cmake/IPEXConfig.cmake.in
  ${PROJECT_BINARY_DIR}/IPEXConfig.cmake
  INSTALL_DESTINATION share/cmake/IPEX)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/IPEXVersionConfig.cmake
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

install(FILES
  ${PROJECT_BINARY_DIR}/IPEXConfig.cmake
  ${PROJECT_BINARY_DIR}/IPEXVersionConfig.cmake
  DESTINATION share/cmake/IPEX)
list(APPEND LIBIPEX_COMP_LIST "share/cmake/IPEX/IPEXConfig.cmake")
list(APPEND LIBIPEX_COMP_LIST "share/cmake/IPEX/IPEXVersionConfig.cmake")

set(CPACK_RESOURCE_FILE_LICENSE "${IPEX_ROOT_DIR}/LICENSE")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION .)

# XXX: All variables with prefix LIBIPEX_ will be used to set in LIBIPEX installation scripts
if(BUILD_MODULE_TYPE STREQUAL "CPPSDK")
  set(CPU_LIB "${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}intel-ext-pt-cpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
  install(FILES "${CMAKE_INSTALL_PREFIX}/${CPU_LIB}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
  list(APPEND LIBIPEX_COMP_LIST "${CPU_LIB}")

  if(BUILD_WITH_XPU)
    set(GPU_LIB "${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}intel-ext-pt-gpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    install(FILES "${CMAKE_INSTALL_PREFIX}/${GPU_LIB}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    list(APPEND LIBIPEX_COMP_LIST "${GPU_LIB}")
  endif()

  set(CPACK_PACKAGE_NAME "${IPEX_PROJ_NAME}")
  if(GLIBCXX_USE_CXX11_ABI)
    set(CPACK_PACKAGE_NAME "${IPEX_PROJ_NAME}-cxx11-abi")
  endif()
  set(LIBIPEX_PACK_PREFIX "${CPACK_OUTPUT_DIR}/${CPACK_PACKAGE_NAME}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_GENERATOR "TXZ")
    set(CPACK_SOURCE_GENERATOR "TXZ")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SOURCE_GENERATOR "ZIP")
  endif()

  if(NOT CPACK_CONFIG_FILE)
    message(FATAL_ERROR "Please set correct CPACK_CONFIG_FILE!")
  endif()
  set(CPACK_OUTPUT_CONFIG_FILE "${CPACK_CONFIG_FILE}")

  if(NOT CPACK_OUTPUT_DIR)
    message(FATAL_ERROR "Please set correct CPACK_OUTPUT_DIR!")
  endif()
  set(CPACK_PACKAGE_DIRECTORY "${CPACK_OUTPUT_DIR}")

  include(CPack)

  set(INSTALL_SCRIPT "libintel-ext-pt-${LIBIPEX_VERSION}.run")
  if(GLIBCXX_USE_CXX11_ABI)
    set(INSTALL_SCRIPT "libintel-ext-pt-cxx11-abi-${LIBIPEX_VERSION}.run")
  endif()
  configure_package_config_file(
    ${IPEX_ROOT_DIR}/cmake/cppsdk/libintel-ext-pt.installer.sh.in
    ${PROJECT_BINARY_DIR}/${INSTALL_SCRIPT}
    INSTALL_DESTINATION ${CPACK_OUTPUT_DIR})
  install(PROGRAMS ${PROJECT_BINARY_DIR}/${INSTALL_SCRIPT}
    DESTINATION ${CPACK_OUTPUT_DIR})
  set(LIBIPEX_INSTALL_SCRIPT "${CPACK_OUTPUT_DIR}/${INSTALL_SCRIPT}")

  if(NOT LIBIPEX_GEN_SCRIPT)
    message(FATAL_ERROR "Please set script name for LIBIPEX_GEN_SCRIPT!")
  endif()
  configure_package_config_file(
    ${IPEX_ROOT_DIR}/cmake/cppsdk/gen_self_extract.sh.in
    ${PROJECT_BINARY_DIR}/${LIBIPEX_GEN_SCRIPT}
    INSTALL_DESTINATION ${CPACK_OUTPUT_DIR})
  install(PROGRAMS ${PROJECT_BINARY_DIR}/${LIBIPEX_GEN_SCRIPT}
    DESTINATION ${CPACK_OUTPUT_DIR})
endif()
