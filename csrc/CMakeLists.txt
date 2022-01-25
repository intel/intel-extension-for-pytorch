set(IPEX_GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(IPEX_GPU_ATEN_SRC_DIR "${IPEX_GPU_ROOT}/aten")
set(IPEX_GPU_ATEN_CORE_DIR "${IPEX_GPU_ATEN_SRC_DIR}/core")
set(IPEX_GPU_ATEN_GENERATED "${IPEX_GPU_ATEN_SRC_DIR}/generated")
set(IPEX_GPU_RUNTIME_SRC_DIR "${IPEX_GPU_ROOT}/runtime")
set(IPEX_GPU_AUTOGRAD_SRC_DIR "${IPEX_GPU_ROOT}/autograd")
set(IPEX_GPU_UTILS_SRC_DIR "${IPEX_GPU_ROOT}/utils")

# sources
set(IPEX_GPU_GEN_FILES)
set(IPEX_GPU_GEN_TARGET)
include(${PROJECT_SOURCE_DIR}/cmake/Codegen.cmake)

set(IPEX_ATEN_OPS_SRCS)
set(IPEX_ATEN_CORE_SRCS)
set(IPEX_ATEN_HEADERS)
add_subdirectory(${IPEX_GPU_ATEN_SRC_DIR})

set(IPEX_RUNTIME_SRCS)
add_subdirectory(${IPEX_GPU_RUNTIME_SRC_DIR})

set(IPEX_AUTOGRAD_SRCS)
add_subdirectory(${IPEX_GPU_AUTOGRAD_SRC_DIR})

set(IPEX_JIT_SRCS)
add_subdirectory(${IPEX_GPU_ROOT}/jit)

set(IPEX_TENSOR_SRCS)
add_subdirectory(${IPEX_GPU_ROOT}/tensor)

set(IPEX_UTILS_SRCS)
set(IPEX_UTILS_HEADERS)
add_subdirectory(${IPEX_GPU_ROOT}/utils)

if(NOT DEFINED SYCL_INCLUDE_DIR)
    message(FATAL_ERROR, "Cannot find SYCL include directory")
endif()

set(IPEX_INCLUDE_DIRS
        ${PYTHON_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${SYCL_INCLUDE_DIR}
        ${IPEX_GPU_ROOT}
        ${IPEX_GPU_ATEN_SRC_DIR}
        ${IPEX_GPU_ATEN_GENERATED})

set(IPEX_GPU_CORE_SRC
        ${IPEX_JIT_SRCS}
        ${IPEX_ATEN_CORE_SRCS}
        ${IPEX_RUNTIME_SRCS}
        ${IPEX_AUTOGRAD_SRCS}
        ${IPEX_GPU_GEN_FILES}
        ${IPEX_TENSOR_SRCS}
        ${IPEX_UTILS_SRCS})

# WARNING XXX: We assume that kernel codes only are contained in IPEX_ATEN_OPS_SRCS.
# WARNING XXX: Therefore we only set sycl flags for these C++ files here.
append_file_property(IPEX_ATEN_OPS_SRCS COMPILE_FLAGS ${IPEX_SYCL_KERNEL_FLAGS})

if(BUILD_SEPARATE_OPS)
  set(IPEX_GPU_OP_LIBRARIES)
  foreach(op_src ${IPEX_ATEN_OPS_SRCS})
      get_filename_component(obj ${op_src} NAME_WLE REALPATH)
      set(op_lib ipex_gpu_op_${obj})
      add_library(${op_lib} SHARED ${op_src})
      append_target_property(${op_lib} LINK_FLAGS "${IPEX_SYCL_LINKER_FLAGS}")
      target_include_directories(${op_lib} PUBLIC ${IPEX_INCLUDE_DIRS})
      target_link_directories(${op_lib} PUBLIC ${SYCL_LIBRARY_DIR})
      target_link_libraries(${op_lib} PUBLIC ${TORCH_LIBRARIES})
      list(APPEND IPEX_GPU_OP_LIBRARIES ${op_lib})
  endforeach()
else()
  list(APPEND IPEX_GPU_CORE_SRC ${IPEX_ATEN_OPS_SRCS})
endif()

add_library(ipex_gpu_core SHARED ${IPEX_GPU_CORE_SRC})
append_target_property(ipex_gpu_core LINK_FLAGS "${IPEX_SYCL_LINKER_FLAGS}")
target_include_directories(ipex_gpu_core PUBLIC ${IPEX_INCLUDE_DIRS})
target_link_directories(ipex_gpu_core PUBLIC ${SYCL_LIBRARY_DIR})

add_dependencies(ipex_gpu_core IPEX_GPU_GEN_TARGET)
include(${PROJECT_SOURCE_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_NATIVE_CSRC_FILES ${IPEX_GPU_ROOT}/**.[ch] ${IPEX_GPU_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_NATIVE_CSRC COMMAND clang-format -i -style=file ${ALL_NATIVE_CSRC_FILES})
  add_dependencies(ipex_gpu_core CL_FORMAT_NATIVE_CSRC)
endif()

if(BUILD_SEPARATE_OPS)
target_link_libraries(ipex_gpu_core PUBLIC ${TORCH_LIBRARIES} ${IPEX_GPU_OP_LIBRARIES})
else()
target_link_libraries(ipex_gpu_core PUBLIC ${TORCH_LIBRARIES})
endif()

set(IPEX_COMPILE_DEFINITIONS)

if(USE_LEVEL_ZERO_ONLY)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_LEVEL_ZERO_ONLY")
endif()

if(USE_PERSIST_STREAM)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_PERSIST_STREAM")
endif()

if(BUILD_INTERNAL_DEBUG)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};BUILD_INTERNAL_DEBUG")
endif()

if(BUILD_DOUBLE_KERNEL)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};BUILD_DOUBLE_KERNEL")
endif()

if (USE_QUEUE_BARRIER)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_QUEUE_BARRIER")
endif()

if (USE_PRIMITIVE_CACHE)
    # Enable FRAMEWORK primitive cache
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_PRIMITIVE_CACHE")
endif()

if (USE_SCRATCHPAD_MODE)
    # Enable oneDNN scratchpad user mode
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_SCRATCHPAD_MODE")
endif()

if (USE_CHANNELS_LAST_1D)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_CHANNELS_LAST_1D")
endif()

if (USE_MULTI_CONTEXT)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_MULTI_CONTEXT")
    message(STATUS "multi context is enabled!")
endif()

if (USE_ONEMKL)
    find_package(oneMKL QUIET)
    if (ONEMKL_FOUND)
        set(ONEMKL_INCLUDE_DIR ${ONEMKL_INCLUDE_DIR} PARENT_SCOPE)
        target_link_libraries(ipex_gpu_core PUBLIC ${ONEMKL_SHARED_LIBS})
        target_include_directories(ipex_gpu_core PUBLIC ${ONEMKL_INCLUDE_DIR})
        set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_ONEMKL")
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

if (USE_ITT)
    find_package(ITT QUIET)
    if (ITT_FOUND)
        target_link_libraries(ipex_gpu_core PRIVATE ${ITT_LIBRARY})
        target_include_directories(ipex_gpu_core PRIVATE ${ITT_INCLUDE_DIR})
        target_compile_definitions(ipex_gpu_core PUBLIC USE_ITT)
        add_subdirectory(${IPEX_GPU_ROOT}/itt)
        target_sources(ipex_gpu_core PRIVATE ${DPCPP_ITT_SRCS})
    else()
        set(USE_ITT OFF PARENT_SCOPE)
        message(WARNING "Cannot find ITT! Continue to build without ITT!")
    endif()
endif()

# XXX: Try to find ITT & oneMKL support before oneDNN settings
find_package(oneDNN QUIET)
if(ONEDNN_FOUND)
    target_link_libraries(ipex_gpu_core PRIVATE ${ONEDNN_LIBRARY})
    target_include_directories(ipex_gpu_core BEFORE PUBLIC ${ONEDNN_INCLUDE_DIR})
    add_dependencies(ipex_gpu_core ${ONEDNN_LIBRARY})
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
            add_dependencies(${op_lib} ${ONEDNN_LIBRARY})
        endforeach()
    endif()
else()
    message(FATAL_ERROR "Cannot find oneDNN")
endif()

target_compile_definitions(ipex_gpu_core PUBLIC ${IPEX_COMPILE_DEFINITIONS})
if(BUILD_SEPARATE_OPS)
  append_target_property(ipex_gpu_core LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")
  foreach(op_lib ${IPEX_GPU_OP_LIBRARIES})
      target_compile_definitions(${op_lib} PUBLIC ${IPEX_COMPILE_DEFINITIONS})
      append_target_property(${op_lib} LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")
  endforeach()
endif()

if(BUILD_STRIPPED_BIN)
  set_target_properties(ipex_gpu_core PROPERTIES LINK_FLAGS_RELEASE -s)
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
install(TARGETS ipex_gpu_core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${IPEX_ATEN_HEADERS} DESTINATION include/core)
install(FILES ${IPEX_UTILS_HEADERS} DESTINATION include/utils)

# Generate the IPEX All-In-One header
set(ipex_header "#pragma once\n")
foreach(uheader ${IPEX_UTILS_HEADERS})
  get_filename_component(header ${uheader} NAME)
  append(ipex_header "\n#include <utils/${header}>")
endforeach()
append(ipex_header "\n")
foreach(aheader ${IPEX_ATEN_HEADERS})
  get_filename_component(header ${aheader} NAME)
  append(ipex_header "\n#include <core/${header}>")
endforeach()
file(WRITE ${IPEX_GPU_ATEN_GENERATED}/ipex.h ${ipex_header})
install(FILES ${IPEX_GPU_ATEN_GENERATED}/ipex.h DESTINATION include)

# CMake config file for the external project.
configure_file(
        ${IPEX_ROOT_DIR}/cmake/IpexConfig.cmake.in
        ${PROJECT_BINARY_DIR}/IpexConfig.cmake
        @ONLY)
install(FILES
        ${PROJECT_BINARY_DIR}/IpexConfig.cmake
        DESTINATION share/cmake/Ipex)
