
set(PLUGIN_NAME_CPU intel-ext-pt-cpu)

set(IPEX_CPU_CPP_THIRD_PARTY_ROOT "${IPEX_ROOT_DIR}/third_party")
set(IPEX_JIT_CPP_ROOT "${IPEX_ROOT_DIR}/csrc/jit")
set(IPEX_UTLIS_CPP_ROOT "${IPEX_ROOT_DIR}/csrc/utils")

set(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

#find_package(TorchCCL REQUIRED)

# TODO: Once llga is merged into oneDNN, use oneDNN directly as the third_party of IPEX
# use the oneDNN in llga temporarily: third_party/llga/third_party/oneDNN

set(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
if(DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND})
  set(DNNL_GRAPH_BUILD_COMPILER_BACKEND ON CACHE BOOL "" FORCE)
  set(DNNL_GRAPH_LLVM_CONFIG "llvm-config-13" CACHE STRING "" FORCE)
endif()
add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn cpu_third_party/ideep/mkl-dnn EXCLUDE_FROM_ALL)
# add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/mkl-dnn cpu_third_party/mkl-dnn)

IF(IPEX_DISP_OP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIPEX_DISP_OP")
ENDIF()

# ---[ Main build
# include mkl-dnn before PyTorch
# Otherwise, path_to_pytorch/torch/include/dnnl.hpp will be used as the header

# generate cpu isa sources
set(IPEX_CPU_CPP_ISA_SRCS_GEN)
set(IPEX_CPU_CPP_ISA_SRCS_ORIGIN)
include(${IPEX_ROOT_DIR}/cmake/cpu/IsaCodegen.cmake)

set(IPEX_CPU_CPP_SRCS)
set(IPEX_CPU_CPP_UTILS_SRCS)
set(IPEX_CPU_CPP_QUANTIZATION_SRCS)
set(IPEX_CPU_CPP_AUTOCAST_SRCS)
set(IPEX_CPU_CPP_ATEN_SRCS)
set(IPEX_CPU_CPP_DYNDISP_SRCS)
set(IPEX_CPU_CPP_ISA_SRCS)
set(IPEX_CPU_CPP_IDEEP_SRCS)
set(IPEX_CPU_CPP_RUNTIME_SRCS)

set(IPEX_JIT_CPP_SRCS)
set(IPEX_UTLIS_CPP_SRCS)

# foreach(file_path ${IPEX_CPU_CPP_ISA_SRCS_GEN})
#   message(${file_path})
# endforeach()

add_subdirectory(${IPEX_CPU_ROOT_DIR}/aten)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/autocast)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/dyndisp)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/ideep)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/isa)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/runtime)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/utils)

add_subdirectory(${IPEX_JIT_CPP_ROOT} jit_cpu)
add_subdirectory(${IPEX_UTLIS_CPP_ROOT} csrc_utlis)

set(IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_DYNDISP_SRCS} ${IPEX_CPU_CPP_ISA_SRCS_GEN} ${IPEX_CPU_CPP_UTILS_SRCS} ${IPEX_CPU_CPP_QUANTIZATION_SRCS} ${IPEX_JIT_CPP_SRCS}
    ${IPEX_CPU_CPP_ISA_SRCS} ${IPEX_CPU_CPP_IDEEP_SRCS} ${IPEX_CPU_CPP_AUTOCAST_SRCS} ${IPEX_CPU_CPP_ATEN_SRCS} ${IPEX_CPU_CPP_RUNTIME_SRCS} ${IPEX_UTLIS_CPP_SRCS})

list(REMOVE_ITEM IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_ISA_SRCS_ORIGIN})

add_library(${PLUGIN_NAME_CPU} SHARED ${IPEX_CPU_CPP_SRCS})

set_target_properties(${PLUGIN_NAME_CPU} PROPERTIES ONEDNN_INCLUDE_DIR "${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn/include")

# includes
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_ROOT_DIR})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR}/aten)
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR}/utils)

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_JIT_CPP_ROOT})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_UTLIS_CPP_ROOT})

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn/include)
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn/third_party/oneDNN/include)
# TODO: once llga is merged into oneDNN, use oneDNN directly as the third_party instead of using that inside llga
# target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${PROJECT_SOURCE_DIR}/build/third_party/mkl-dnn/include)
# target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/mkl-dnn/include)

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/include)
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${PYTHON_INCLUDE_DIR})

include(${IPEX_ROOT_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_NATIVE_CSRC_FILES ${IPEX_CPU_ROOT_DIR}/**.[ch] ${IPEX_CPU_ROOT_DIR}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CPU_NATIVE_CSRC COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_NATIVE_CSRC_FILES})
  add_dependencies(${PLUGIN_NAME_CPU} CL_FORMAT_CPU_NATIVE_CSRC)
endif()

add_dependencies(${PLUGIN_NAME_CPU} dnnl_graph)
# If Graph Compiler is built, then it should link to its LLVM dependencies,
# and not the LLVM symbols exposed by PyTorch.
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC dnnl_graph)
if (DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND})
  get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB dnnl_graphcompiler_llvm_lib INTERFACE_LINK_LIBRARIES)
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC graphcompiler ${DNNL_GRAPHCOMPILER_LLVM_LIB})
  # BUILD_SHARED_LIBS is not used to control ipex library type, it is always shared
  if (NOT MSVC)
    get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB_EXCLUDE dnnl_graphcompiler_llvm_lib_exclude_string INTERFACE_LINK_LIBRARIES)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs=${DNNL_GRAPHCOMPILER_LLVM_LIB_EXCLUDE}")
  endif()
endif()

find_package(oneMKL QUIET)
if (ONEMKL_FOUND)
  target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${ONEMKL_INCLUDE_DIR})
  set_target_properties(${PLUGIN_NAME_CPU} PROPERTIES ONEMKL_INCLUDE_DIR "${ONEMKL_INCLUDE_DIR}")
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC ${ONEMKL_CPU_LIBS})
else()
  message(FATAL_ERROR "ERROR: Cannot find oneMKL!")
endif()

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC ${TORCH_LIBRARIES})

set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

target_compile_options(${PLUGIN_NAME_CPU} PRIVATE "-DC10_BUILD_MAIN_LIB")

install(TARGETS ${PLUGIN_NAME_CPU}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
