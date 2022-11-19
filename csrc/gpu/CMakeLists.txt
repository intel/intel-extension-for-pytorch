set(IPEX_GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(IPEX_GPU_ATEN_SRC_DIR "${IPEX_GPU_ROOT}/aten")
set(IPEX_GPU_ATEN_CORE_DIR "${IPEX_GPU_ATEN_SRC_DIR}/core")
set(IPEX_GPU_RUNTIME_SRC_DIR "${IPEX_GPU_ROOT}/runtime")
set(IPEX_GPU_UTILS_SRC_DIR "${IPEX_GPU_ROOT}/utils")
set(IPEX_GPU_INCLUDE_DIR "${IPEX_GPU_ROOT}/include")

# sources
set(IPEX_GPU_GEN_FILES)
set(IPEX_GPU_GEN_TARGET)
include(${PROJECT_SOURCE_DIR}/cmake/xpu/Codegen.cmake)

set(IPEX_ATEN_CORE_SRCS)
set(IPEX_ATEN_OPS_SRCS)
add_subdirectory(${IPEX_GPU_ATEN_SRC_DIR})

set(IPEX_RUNTIME_SRCS)
add_subdirectory(${IPEX_GPU_RUNTIME_SRC_DIR})

set(IPEX_JIT_SRCS)
add_subdirectory(${IPEX_GPU_ROOT}/jit)

set(IPEX_UTILS_SRCS)
add_subdirectory(${IPEX_GPU_ROOT}/utils)

set(IPEX_DISTRIBUTED_SRCS)
add_subdirectory(${IPEX_GPU_ROOT}/distributed)

if(NOT DEFINED SYCL_INCLUDE_DIR)
    message(FATAL_ERROR, "Cannot find SYCL include directory")
endif()

set(IPEX_INCLUDE_DIRS
        ${PYTHON_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${SYCL_INCLUDE_DIR}
        ${IPEX_GPU_ROOT}
        ${IPEX_GPU_ATEN_SRC_DIR})

set(IPEX_GPU_CORE_SRC
        ${IPEX_JIT_SRCS}
        ${IPEX_DISTRIBUTED_SRCS}
        ${IPEX_ATEN_CORE_SRCS}
        ${IPEX_RUNTIME_SRCS}
        ${IPEX_GPU_GEN_FILES}
        ${IPEX_UTILS_SRCS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${IPEX_SYCL_KERNEL_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${IPEX_SYCL_LINKER_FLAGS}")

if(BUILD_SEPARATE_OPS)
  set(IPEX_GPU_OP_LIBRARIES)
  foreach(op_src ${IPEX_ATEN_OPS_SRCS})
      get_filename_component(obj ${op_src} NAME_WLE REALPATH)
      set(op_lib intel-ext-pt-gpu-op-${obj})
      add_library(${op_lib} SHARED ${op_src})
      target_include_directories(${op_lib} PUBLIC ${IPEX_INCLUDE_DIRS})
      target_link_directories(${op_lib} PUBLIC ${SYCL_LIBRARY_DIR})
      target_link_libraries(${op_lib} PUBLIC ${TORCH_LIBRARIES})
      list(APPEND IPEX_GPU_OP_LIBRARIES ${op_lib})
  endforeach()
else()
  list(APPEND IPEX_GPU_CORE_SRC ${IPEX_ATEN_OPS_SRCS})
endif()

add_library(intel-ext-pt-gpu SHARED ${IPEX_GPU_CORE_SRC})
target_include_directories(intel-ext-pt-gpu PUBLIC ${IPEX_INCLUDE_DIRS})
target_link_directories(intel-ext-pt-gpu PUBLIC ${SYCL_LIBRARY_DIR})

add_dependencies(intel-ext-pt-gpu IPEX_GPU_GEN_TARGET)
include(${PROJECT_SOURCE_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_NATIVE_CSRC_FILES ${IPEX_GPU_ROOT}/**.[ch] ${IPEX_GPU_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_NATIVE_CSRC COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_NATIVE_CSRC_FILES})
  add_dependencies(intel-ext-pt-gpu CL_FORMAT_NATIVE_CSRC)
endif()

if(BUILD_SEPARATE_OPS)
target_link_libraries(intel-ext-pt-gpu PUBLIC ${TORCH_LIBRARIES} ${IPEX_GPU_OP_LIBRARIES})
else()
target_link_libraries(intel-ext-pt-gpu PUBLIC ${TORCH_LIBRARIES})
endif()

set(IPEX_COMPILE_DEFINITIONS)
if(USE_PERSIST_STREAM)
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_PERSIST_STREAM")
endif()

if(BUILD_SIMPLE_TRACE)
  list(APPEND IPEX_COMPILE_DEFINITIONS "BUILD_SIMPLE_TRACE")
endif()

if(BUILD_INTERNAL_DEBUG)
  list(APPEND IPEX_COMPILE_DEFINITIONS "BUILD_INTERNAL_DEBUG")
endif()

if (USE_QUEUE_BARRIER)
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_QUEUE_BARRIER")
endif()

if (USE_PRIMITIVE_CACHE)
  # Enable FRAMEWORK primitive cache
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_PRIMITIVE_CACHE")
endif()

if (USE_SCRATCHPAD_MODE)
  # Enable oneDNN scratchpad user mode
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_SCRATCHPAD_MODE")
endif()

if (USE_CHANNELS_LAST_1D)
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_CHANNELS_LAST_1D")
endif()

if (BUILD_JIT_QUANTIZATION_SAVE)
  list(APPEND IPEX_COMPILE_DEFINITIONS "BUILD_JIT_QUANTIZATION_SAVE")
endif()

# use onedpl
find_package(oneDPL)
if (ONEDPL_FOUND)
  target_link_libraries(intel-ext-pt-gpu PUBLIC oneDPL)
  if(BUILD_SEPARATE_OPS)
      foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
          target_link_libraries(${op_lib} PUBLIC oneDPL)
      endforeach()
  endif()
  set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};ONEDPL_USE_TBB_BACKEND=0")
  # FIXME:
  # Refer to oneAPI TBB release note:
  # https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html
  # We must set below two flags to bypass TBB checking by GCC with libstdc++ 9 and libstdc++ 10
  # Related Jira: https://jira.devtools.intel.com/browse/ONEDPL-245
  set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};PSTL_USE_PARALLEL_POLICIES=0")
  set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};_GLIBCXX_USE_TBB_PAR_BACKEND=0")
else()
  message(FATAL_ERROR "Cannot find oneDPL")
endif()

if (USE_MULTI_CONTEXT)
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_MULTI_CONTEXT")
  message(STATUS "multi context is enabled!")
endif()

if (USE_PROFILER)
  # Enable Profiler
  list(APPEND IPEX_COMPILE_DEFINITIONS "USE_PROFILER")
endif()

if (USE_ONEMKL)
    find_package(oneMKL QUIET)
    if (ONEMKL_FOUND)
        set(ONEMKL_INCLUDE_DIR ${ONEMKL_INCLUDE_DIR} PARENT_SCOPE)
        target_link_libraries(intel-ext-pt-gpu PUBLIC ${ONEMKL_SHARED_LIBS})
        target_include_directories(intel-ext-pt-gpu PUBLIC ${ONEMKL_INCLUDE_DIR})
        list(APPEND IPEX_COMPILE_DEFINITIONS "USE_ONEMKL")
        if(BUILD_SEPARATE_OPS)
            foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
                target_link_libraries(${op_lib} PUBLIC ${ONEMKL_SHARED_LIBS})
                target_include_directories(${op_lib} PUBLIC ${ONEMKL_INCLUDE_DIR})
            endforeach()
        endif()
    else()
        set(USE_ONEMKL OFF PARENT_SCOPE)
        message(WARNING "WARNING: Cannot find oneMKL! Continue to build without oneMKL!")
    endif()
endif()

# XXX: Try to find oneMKL support before oneDNN settings
find_package(oneDNN QUIET)
if(ONEDNN_FOUND)
    target_link_libraries(intel-ext-pt-gpu PRIVATE ${ONEDNN_LIBRARY})
    target_include_directories(intel-ext-pt-gpu BEFORE PUBLIC ${ONEDNN_INCLUDE_DIR})
    if(BUILD_SEPARATE_OPS)
      # Suppress soname to prevent from creating symbolic link
      # Change output path to avoid link error from dependents
      get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
      set_target_properties(${ONEDNN_LIBRARY} PROPERTIES NO_SONAME ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DIR_NAME}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DIR_NAME}")
        foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
            target_link_libraries(${op_lib} PRIVATE ${ONEDNN_LIBRARY})
            target_include_directories(${op_lib} BEFORE PUBLIC ${ONEDNN_INCLUDE_DIR})
        endforeach()
    endif()
else()
    message(FATAL_ERROR "Cannot find oneDNN")
endif()

target_compile_definitions(intel-ext-pt-gpu PUBLIC ${IPEX_COMPILE_DEFINITIONS})
if(BUILD_SEPARATE_OPS)
  foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
      target_compile_definitions(${op_lib} PUBLIC ${IPEX_COMPILE_DEFINITIONS})
  endforeach()
endif()

if(BUILD_STRIPPED_BIN)
  set_target_properties(intel-ext-pt-gpu PROPERTIES LINK_FLAGS_RELEASE -s)
  if(BUILD_SEPARATE_OPS)
    foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
        set_target_properties(${op_lib} PROPERTIES LINK_FLAGS_RELEASE -s)
    endforeach()
  endif()
endif()

if(BUILD_SEPARATE_OPS)
  install(TARGETS ${ONEDNN_LIBRARY}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
    install(TARGETS ${op_lib}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endforeach()
endif()
install(TARGETS intel-ext-pt-gpu
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# install IPEX GPU public header files
install(DIRECTORY ${IPEX_GPU_INCLUDE_DIR} DESTINATION .)

# CMake config file for the external project.
configure_file(
        ${IPEX_ROOT_DIR}/cmake/IntelExtensionPytorchConfig.cmake.in
        ${PROJECT_BINARY_DIR}/IntelExtensionPytorchConfig.cmake
        @ONLY)
install(FILES
        ${PROJECT_BINARY_DIR}/IntelExtensionPytorchConfig.cmake
        DESTINATION share/cmake/IntelExtensionPytorch)
