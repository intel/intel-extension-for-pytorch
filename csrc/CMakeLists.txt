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
        ${IPEX_ATEN_OPS_SRCS}
        ${IPEX_ATEN_CORE_SRCS}
        ${IPEX_RUNTIME_SRCS}
        ${IPEX_AUTOGRAD_SRCS}
        ${IPEX_GPU_GEN_FILES}
        ${IPEX_TENSOR_SRCS}
        ${IPEX_UTILS_SRCS})
# WARNING XXX: We assume that kernel codes only are contained in IPEX_ATEN_OPS_SRCS.
# WARNING XXX: Therefore we only set sycl flags for these C++ files here.
append_sycl_flag(IPEX_ATEN_OPS_SRCS ${IPEX_SYCL_KERNEL_FLAGS})
add_library(ipex_gpu_core SHARED ${IPEX_GPU_CORE_SRC})

set(IPEX_GPU_SRC ${IPEX_JIT_SRCS})
add_library(ipex_gpu SHARED ${IPEX_GPU_SRC})

target_include_directories(ipex_gpu_core PUBLIC ${IPEX_INCLUDE_DIRS})
target_include_directories(ipex_gpu PUBLIC ${IPEX_INCLUDE_DIRS})

include(${PROJECT_SOURCE_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_NATIVE_CSRC_FILES ${IPEX_GPU_ROOT}/**.[ch] ${IPEX_GPU_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_NATIVE_CSRC COMMAND clang-format -i -style=file ${ALL_NATIVE_CSRC_FILES})
  add_dependencies(ipex_gpu_core CL_FORMAT_NATIVE_CSRC)
endif()

add_dependencies(ipex_gpu_core IPEX_GPU_GEN_TARGET)

target_link_libraries(ipex_gpu_core PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(ipex_gpu PUBLIC ${TORCH_LIBRARIES})

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

if (USE_PRIMITIVE_CACHE)
    # Enable FRAMEWORK primitive cache
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_PRIMITIVE_CACHE")
endif()

if (USE_SCRATCHPAD_MODE)
    # Enable oneDNN scratchpad user mode
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_SCRATCHPAD_MODE")
endif()

# Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};CL_TARGET_OPENCL_VERSION=220")
if(USE_ONEDPL)
    find_package(oneDPL)
    target_link_libraries(ipex_gpu PUBLIC oneDPL)
    target_link_libraries(ipex_gpu_core PUBLIC oneDPL)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_ONEDPL")
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};ONEDPL_USE_TBB_BACKEND=0")
    # FIXME:
    # Refer to oneAPI TBB release note:
    # https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html
    # We must set below two flags to bypass TBB checking by GCC with libstdc++ 9 and libstdc++ 10
    # Related Jira:
    # https://jira.devtools.intel.com/browse/ONEDPL-245?focusedCommentId=12217329&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-12217329
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};PSTL_USE_PARALLEL_POLICIES=0")
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};_GLIBCXX_USE_TBB_PAR_BACKEND=0")
endif()

if (USE_MULTI_CONTEXT)
    set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_MULTI_CONTEXT")
    message(STATUS "multi context is enabled!")
endif()

if (USE_ONEMKL)
    find_package(MKLDPCPP QUIET)
    if (MKLDPCPP_FOUND)
        set(ONEMKL_INCLUDE_DIR ${ONEMKL_INCLUDE_DIR} PARENT_SCOPE)
        target_link_libraries(ipex_gpu_core PUBLIC ${ONEMKL_SHARED_LIBS})
        target_include_directories(ipex_gpu_core PUBLIC ${ONEMKL_INCLUDE_DIR})
        set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_ONEMKL")
    else()
        set(USE_ONEMKL OFF PARENT_SCOPE)
        message(WARNING "WARNING: Cannot find oneMKL! Continue to build without oneMKL!")
    endif()
endif()

find_package(oneDNN QUIET)
if(ONEDNN_FOUND)
    target_link_libraries(ipex_gpu_core PRIVATE ${ONEDNN_LIBRARIES})
    target_include_directories(ipex_gpu_core BEFORE PUBLIC ${ONEDNN_INCLUDE_DIR})
    add_dependencies(ipex_gpu_core dnnl)
else()
    message(FATAL_ERROR "Cannot find oneDNN")
endif()

if (USE_ITT)
    find_package(ITT QUIET)
    if (ITT_FOUND)
        target_link_libraries(ipex_gpu_core PRIVATE ${ITT_LIBRARY})
        target_include_directories(ipex_gpu_core PRIVATE ${ITT_INCLUDE_DIR})
        target_compile_definitions(ipex_gpu_core PUBLIC USE_ITT)
        add_subdirectory(./itt)
        target_sources(ipex_gpu_core PRIVATE ${DPCPP_ITT_SRCS})
    else()
        set(USE_ITT OFF PARENT_SCOPE)
        message(WARNING "Cannot find ITT! Continue to build without ITT!")
    endif()
endif()

target_compile_definitions(ipex_gpu PUBLIC ${IPEX_COMPILE_DEFINITIONS})
target_compile_definitions(ipex_gpu_core PUBLIC ${IPEX_COMPILE_DEFINITIONS})

target_link_libraries(ipex_gpu PUBLIC ${EXTRA_SHARED_LIBS} ipex_gpu_core)

if(BUILD_STRIPPED_BIN)
  set_target_properties(ipex_gpu PROPERTIES LINK_FLAGS_RELEASE -s)
  set_target_properties(ipex_gpu_core PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

install(TARGETS ipex_gpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ipex_gpu_core LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

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
