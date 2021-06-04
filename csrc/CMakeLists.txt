include(${PROJECT_SOURCE_DIR}/cmake/Codegen.cmake)
set(IPEX_GPU_ATEN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aten")
set(IPEX_GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# sources
set(IPEX_ATEN_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/aten)

set(IPEX_JIT_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit)

set(IPEX_TENSOR_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensor)

if(NOT DEFINED SYCL_INCLUDE_DIR)
    message(FATAL_ERROR, "Cannot find SYCL include directory")
endif()

set(IPEX_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${SYCL_INCLUDE_DIR}
        ${IPEX_C_SOURCE_DIR}
        ${IPEX_GPU_ROOT}
        ${IPEX_GPU_ATEN_SRC_DIR}
        ${IPEX_GPU_ATEN_GENERATED})

FILE(GLOB GPU_UTIL_SRC "*.cpp")

set(SYCL_SOURCES ${GPU_UTIL_SRC} ${IPEX_JIT_SRCS} ${IPEX_ATEN_SRCS})
set_sycl_flag("${SYCL_SOURCES}")

set(IPEX_GPU_SRC ${GPU_UTIL_SRC} ${IPEX_JIT_SRCS})

add_library(torch_ipex_kernel SHARED ${IPEX_ATEN_SRCS} ${IPEX_TENSOR_SRCS})
add_library(torch_ipex_gpu SHARED ${IPEX_GPU_SRC})

target_include_directories(torch_ipex_gpu PUBLIC ${IPEX_INCLUDE_DIRS})
target_include_directories(torch_ipex_kernel PUBLIC ${IPEX_INCLUDE_DIRS})

target_link_libraries(torch_ipex_gpu PUBLIC IPEX_GPU_FILES_GEN_LIB)

target_link_libraries(torch_ipex_gpu PUBLIC ${TORCH_LIBRARIES})

set(IPEX_COMPILE_DEFINITIONS)

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

# Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};CL_TARGET_OPENCL_VERSION=220")
if(USE_ONEDPL)
    find_package(oneDPL)
    target_link_libraries(torch_ipex_gpu PUBLIC oneDPL)
    target_link_libraries(torch_ipex_kernel PUBLIC oneDPL)
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
        target_link_libraries(torch_ipex_kernel PUBLIC ${ONEMKL_SHARED_LIBS})
        target_include_directories(torch_ipex_kernel PUBLIC ${ONEMKL_INCLUDE_DIR})
        set(IPEX_COMPILE_DEFINITIONS "${IPEX_COMPILE_DEFINITIONS};USE_ONEMKL")
    else()
        set(USE_ONEMKL OFF PARENT_SCOPE)
        message(WARNING "WARNING: Cannot find oneMKL! Continue to build without oneMKL!")
    endif()
endif()

find_package(oneDNN QUIET)
if(ONEDNN_FOUND)
    target_link_libraries(torch_ipex_kernel PRIVATE ${ONEDNN_LIBRARIES})
    target_include_directories(torch_ipex_kernel BEFORE PUBLIC ${ONEDNN_INCLUDE_DIR})
    add_dependencies(torch_ipex_kernel dnnl)
else()
    message(FATAL_ERROR "Cannot find oneDNN")
endif()

target_compile_definitions(torch_ipex_gpu PUBLIC ${IPEX_COMPILE_DEFINITIONS})
target_compile_definitions(torch_ipex_kernel PUBLIC ${IPEX_COMPILE_DEFINITIONS})

target_link_libraries(torch_ipex_gpu PUBLIC ${EXTRA_SHARED_LIBS} torch_ipex_kernel)

install(TARGETS torch_ipex_gpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS torch_ipex_kernel LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "./" DESTINATION include/ipex FILES_MATCHING PATTERN "*.h")
