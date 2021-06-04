set(IPEX_GPU_SRCS)
add_subdirectory(./gpu)
set_sycl_flag("${IPEX_GPU_SRCS}")

set(IPEX_COMMON_SRCS)
FILE(GLOB IPEX_COMMON_SRCS "*.cpp")
list(REMOVE_ITEM IPEX_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/_C.cpp")

set(TORCH_IPEX_SRCS)
list(APPEND TORCH_IPEX_SRCS ${IPEX_COMMON_SRCS} ${IPEX_GPU_SRCS})
add_library(torch_ipex_python SHARED ${TORCH_IPEX_SRCS})


if (USE_ITT)
    find_package(ITT QUIET)
    if (ITT_FOUND)
        set(ITT_LIBRARY ${ITT_LIBRARY} PARENT_SCOPE)
        set(ITT_INCLUDE_DIR ${ITT_INCLUDE_DIR} PARENT_SCOPE)
        target_link_libraries(torch_ipex_python PUBLIC ${ITT_LIBRARY})
        target_include_directories(torch_ipex_python PUBLIC ${ITT_INCLUDE_DIR})
        target_compile_definitions(torch_ipex_python PUBLIC USE_ITT)

        set(DPCPP_ITT_SRCS)
        add_subdirectory(./itt)
        target_sources(torch_ipex_python PUBLIC ${DPCPP_ITT_SRCS})
    else()
        set(USE_ITT OFF PARENT_SCOPE)
        message(WARNING "Cannot find ITT! Continue to build without ITT!")
    endif()
endif()

set_target_properties(torch_ipex_python PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

target_link_libraries(torch_ipex_python PUBLIC torch_ipex_gpu)

install(TARGETS torch_ipex_python LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
