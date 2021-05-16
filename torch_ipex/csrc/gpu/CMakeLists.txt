include(${PROJECT_SOURCE_DIR}/cmake/Codegen.cmake)
set(IPEX_GPU_ATEN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aten")

# sources
set(IPEX_ATEN_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/aten)

set(IPEX_JIT_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit)

set(IPEX_TENSOR_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensor)

FILE(GLOB GPU_UTIL_SRC "*.cpp")

install(DIRECTORY "./" DESTINATION include FILES_MATCHING PATTERN "*.h")

set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -D__STRICT_ANSI__")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl-unnamed-lambda")
set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fno-sycl-early-optimizations")
# Explicitly limit the index range (< Max int32) in kernel
# set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl-id-queries-fit-in-int")
set_source_files_properties(${GPU_UTIL_SRC} ${IPEX_ATEN_SRCS} COMPILE_FLAGS "${IPEX_COMPILE_FLAGS}")

set(IPEX_GPU_SRC ${GPU_UTIL_SRC} ${IPEX_ATEN_SRCS} ${IPEX_JIT_SRCS} ${IPEX_TENSOR_SRCS})


add_library(torch_ipex_gpu SHARED ${IPEX_GPU_SRC})
set_target_properties(torch_ipex_gpu PROPERTIES PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
set_target_properties(torch_ipex_gpu PROPERTIES OUTPUT_NAME torch_ipex_gpu)

# includes
if(DEFINED PYTORCH_INCLUDE_DIR)
    target_include_directories(torch_ipex_gpu PUBLIC ${PYTORCH_INCLUDE_DIR})
else()
    message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

if(DEFINED SYCL_INCLUDE_DIR)
    target_include_directories(torch_ipex_gpu PUBLIC ${SYCL_INCLUDE_DIR})
else()
    message(FATAL_ERROR, "Cannot find SYCL include directory")
endif()

target_include_directories(torch_ipex_gpu PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(torch_ipex_gpu PUBLIC ${IPEX_C_SOURCE_DIR})
target_include_directories(torch_ipex_gpu PUBLIC ${IPEX_GPU_ROOT})
target_include_directories(torch_ipex_gpu PUBLIC ${IPEX_GPU_ATEN_SRC_DIR})
target_include_directories(torch_ipex_gpu PUBLIC ${IPEX_GPU_ATEN_GENERATED})

target_link_libraries(torch_ipex_gpu PUBLIC IPEX_GPU_FILES_GEN_LIB)


# pytorch library
if(DEFINED PYTORCH_LIBRARY_DIR)
    link_directories(${PYTORCH_LIBRARY_DIR})
    target_link_libraries(torch_ipex_gpu PUBLIC ${PYTORCH_LIBRARY_DIR}/libtorch_cpu.so)
    target_link_libraries(torch_ipex_gpu PUBLIC ${PYTORCH_LIBRARY_DIR}/libtorch_python.so)
    target_link_libraries(torch_ipex_gpu PUBLIC ${PYTORCH_LIBRARY_DIR}/libc10.so)
else()
    message(FATAL_ERROR, "Cannot find PyTorch library directory")
endif()

if(USE_PERSIST_STREAM)
    target_compile_definitions(torch_ipex_gpu PUBLIC USE_PERSIST_STREAM)
endif()

if(BUILD_INTERNAL_DEBUG)
    target_compile_definitions(torch_ipex_gpu PUBLIC BUILD_INTERNAL_DEBUG)
endif()

if(BUILD_DOUBLE_KERNEL)
    target_compile_definitions(torch_ipex_gpu PUBLIC BUILD_DOUBLE_KERNEL)
endif()

if (USE_PRIMITIVE_CACHE)
    # Enable FRAMEWORK primitive cache
    target_compile_definitions(torch_ipex_gpu PUBLIC USE_PRIMITIVE_CACHE)
endif()

# Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
target_compile_definitions(torch_ipex_gpu PUBLIC CL_TARGET_OPENCL_VERSION=220)
if(USE_ONEDPL)
    find_package(oneDPL)
    target_link_libraries(torch_ipex_gpu PUBLIC oneDPL)
    target_compile_definitions(torch_ipex_gpu PUBLIC USE_ONEDPL)
    target_compile_definitions(torch_ipex_gpu PUBLIC ONEDPL_USE_TBB_BACKEND=0)
    # FIXME:
    # Refer to oneAPI TBB release note:
    # https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html
    # We must set below two flags to bypass TBB checking by GCC with libstdc++ 9 and libstdc++ 10
    # Related Jira:
    # https://jira.devtools.intel.com/browse/ONEDPL-245?focusedCommentId=12217329&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-12217329
    target_compile_definitions(torch_ipex_gpu PUBLIC PSTL_USE_PARALLEL_POLICIES=0)
    target_compile_definitions(torch_ipex_gpu PUBLIC _GLIBCXX_USE_TBB_PAR_BACKEND=0)
endif()


if (USE_MULTI_CONTEXT)
    target_compile_definitions(torch_ipex_gpu PUBLIC USE_MULTI_CONTEXT)
    message(STATUS "multi context is enabled!")
endif()

if (USE_ONEMKL)
    find_package(MKLDPCPP QUIET)
    if (MKLDPCPP_FOUND)
        set(ONEMKL_INCLUDE_DIR ${ONEMKL_INCLUDE_DIR} PARENT_SCOPE)
        target_link_libraries(torch_ipex_gpu PUBLIC ${ONEMKL_SHARED_LIBS})
        target_include_directories(torch_ipex_gpu PUBLIC ${ONEMKL_INCLUDE_DIR})
        target_compile_definitions(torch_ipex_gpu PUBLIC USE_ONEMKL)
    else()
        set(USE_ONEMKL OFF PARENT_SCOPE)
        message(WARNING "WARNING: Cannot find oneMKL! Continue to build without oneMKL!")
    endif()
endif()

find_package(oneDNN QUIET)
if(ONEDNN_FOUND)
    target_link_libraries(torch_ipex_gpu PUBLIC ${ONEDNN_LIBRARIES})
    target_include_directories(torch_ipex_gpu BEFORE PUBLIC ${ONEDNN_INCLUDE_DIR})
    add_dependencies(torch_ipex_gpu dnnl)
else()
    message(FATAL_ERROR "Cannot find oneDNN")
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
if(BUILD_BY_PER_KERNEL)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_kernel")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl, -T ${PROJECT_SOURCE_DIR}/cmake/per_ker.ld")
elseif(USE_AOT_DEVLIST)
    set(SPIRV_OPT "spir64-unknown-unknown-sycldevice")
    set(AOT_ARCH_OPT "spir64_gen-unknown-unknown-sycldevice")
    set(IPEX_COMPILE_FLAGS "${IPEX_COMPILE_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_source")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Xsycl-target-backend=${AOT_ARCH_OPT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} '-device ${USE_AOT_DEVLIST}'")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_source")
endif()
message(STATUS "DPCPP found. Compiling with SYCL support")

target_link_libraries(torch_ipex_gpu PUBLIC ${EXTRA_SHARED_LIBS})

install(TARGETS torch_ipex_gpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
