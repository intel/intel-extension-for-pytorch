
set(PLUGIN_NAME_CPU intel-ext-pt-cpu)

set(IPEX_CPU_CPP_THIRD_PARTY_ROOT "${IPEX_ROOT_DIR}/third_party")
set(IPEX_JIT_CPP_ROOT "${IPEX_ROOT_DIR}/csrc/jit")
set(IPEX_UTLIS_CPP_ROOT "${IPEX_ROOT_DIR}/csrc/utils")

set(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

# TODO: Once llga is merged into oneDNN, use oneDNN directly as the third_party of IPEX
# use the oneDNN in llga temporarily: third_party/llga/third_party/oneDNN

set(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
if((DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND}) AND USE_DNNL_GRAPH_COMPILER)
  set(ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND ON CACHE BOOL "" FORCE)
  set(ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT "llvm;builtin" CACHE STRING "" FORCE)
  set(ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_LLVM_CONFIG "llvm-config-13" CACHE STRING "" FORCE)
endif()

set(THIRD_PARTY_BUILD_PATH_NAME "cpu_third_party")
if(USE_SYSTEM_ONEDNN)
  find_package(dnnl 3.4.1 CONFIG REQUIRED)
  get_target_property(ONEDNN_INCLUDE_DIR DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
  set(ONEDNN_LIBRARY DNNL::dnnl)
  set(ONEDNN_GENERATED_INCLUDE ${ONEDNN_INCLUDE_DIR})
else()
  add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn ${THIRD_PARTY_BUILD_PATH_NAME}/ideep/mkl-dnn EXCLUDE_FROM_ALL)
  set(ONEDNN_INCLUDE_DIR ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/mkl-dnn/include)
  set(ONEDNN_LIBRARY dnnl)

  # path of oneDNN .h.in generated file
  file(RELATIVE_PATH CUR_DIR_REL_PATH "${IPEX_ROOT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  set(ONEDNN_GENERATED_INCLUDE "${CMAKE_BINARY_DIR}/${CUR_DIR_REL_PATH}/${THIRD_PARTY_BUILD_PATH_NAME}/ideep/mkl-dnn/include")
endif()

IF(IPEX_DISP_OP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIPEX_DISP_OP")
ENDIF()

if(USE_LIBXSMM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBXSMM")
endif(USE_LIBXSMM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_IPEX_MAIN_LIB")

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
set(IPEX_CPU_CPP_TOOLKIT_SRCS)
set(IPEX_CPU_CPP_IDEEP_SRCS)
set(IPEX_CPU_CPP_RUNTIME_SRCS)
set(IPEX_CPU_CPP_TPP_SRCS)
set(IPEX_CPU_CPP_JIT_SRCS)

set(IPEX_UTLIS_CPP_SRCS)
set(IPEX_JIT_COMMON_CPP_SRCS)

# foreach(file_path ${IPEX_CPU_CPP_ISA_SRCS_GEN})
#   message(${file_path})
# endforeach()

add_subdirectory(${IPEX_CPU_ROOT_DIR}/aten)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/autocast)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/dyndisp)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/ideep)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/isa)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/toolkit)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/runtime)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/utils)

add_subdirectory(${IPEX_CPU_ROOT_DIR}/jit)

add_subdirectory(${IPEX_JIT_CPP_ROOT} jit_common)
add_subdirectory(${IPEX_UTLIS_CPP_ROOT} csrc_utlis)

if(USE_LIBXSMM)
  add_subdirectory(${IPEX_CPU_ROOT_DIR}/tpp)
endif(USE_LIBXSMM)

set(IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_DYNDISP_SRCS} ${IPEX_CPU_CPP_ISA_SRCS_GEN} ${IPEX_CPU_CPP_UTILS_SRCS} ${IPEX_CPU_CPP_QUANTIZATION_SRCS} ${IPEX_CPU_CPP_JIT_SRCS} ${IPEX_JIT_COMMON_CPP_SRCS}
    ${IPEX_CPU_CPP_ISA_SRCS} ${IPEX_CPU_CPP_IDEEP_SRCS} ${IPEX_CPU_CPP_AUTOCAST_SRCS} ${IPEX_CPU_CPP_ATEN_SRCS} ${IPEX_CPU_CPP_RUNTIME_SRCS} ${IPEX_CPU_CPP_TOOLKIT_SRCS} ${IPEX_UTLIS_CPP_SRCS} 
    ${IPEX_CPU_CPP_TPP_SRCS})

list(REMOVE_ITEM IPEX_CPU_CPP_SRCS ${IPEX_CPU_CPP_ISA_SRCS_ORIGIN})

# Disable XCR check to support Virtual Machines. Some hypervisor can't simulate XCR0 correctly.
# add_definitions (-DENABLE_XCR_CHECK)

add_library(${PLUGIN_NAME_CPU} SHARED ${IPEX_CPU_CPP_SRCS})

# For IPEX_API macro
target_compile_definitions(${PLUGIN_NAME_CPU} PUBLIC "BUILD_IPEX_MAIN_LIB")

set_target_properties(${PLUGIN_NAME_CPU} PROPERTIES ONEDNN_INCLUDE_DIR ${ONEDNN_INCLUDE_DIR})

# includes
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_ROOT_DIR})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_ROOT_DIR}/csrc/include)
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR}/aten)
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR}/utils)
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR}/jit)

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_JIT_CPP_ROOT})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_UTLIS_CPP_ROOT})

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${ONEDNN_INCLUDE_DIR})

if(USE_LIBXSMM)
  target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IPEX_CPU_ROOT_DIR}/tpp)
  target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${LIBXSMM_INCLUDE_DIRS})
endif(USE_LIBXSMM)

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${ONEDNN_GENERATED_INCLUDE})

if(USE_SYSTEM_IDEEP)
  find_path(IDEEP_INCLUDE_DIR ideep.hpp REQUIRED)
else()
  set(IDEEP_INCLUDE_DIR ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/ideep/include)
endif()
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${IDEEP_INCLUDE_DIR})
target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${PYTHON_INCLUDE_DIR})

include(${IPEX_ROOT_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_CPU_NATIVE_CSRC_FILES ${IPEX_CPU_ROOT_DIR}/**.[ch] ${IPEX_CPU_ROOT_DIR}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CPU_NATIVE_CSRC COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_CPU_NATIVE_CSRC_FILES})
  add_dependencies(${PLUGIN_NAME_CPU} CL_FORMAT_CPU_NATIVE_CSRC)

  # clang-foamt CPU examples
  file(GLOB_RECURSE ALL_CPU_EXAMPLE_NATIVE_CSRC_FILES ${IPEX_CPU_EXAMPLE_ROOT_DIR}/**.[ch] ${IPEX_CPU_EXAMPLE_ROOT_DIR}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CPU_EXAMPLE_NATIVE_CSRC COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_CPU_EXAMPLE_NATIVE_CSRC_FILES})
  add_dependencies(${PLUGIN_NAME_CPU} CL_FORMAT_CPU_EXAMPLE_NATIVE_CSRC)
endif()

if(USE_LIBXSMM)
  if(USE_SYSTEM_LIBXSMM)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBXSMM REQUIRED libxsmm)
    target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${LIBXSMM_INCLUDE_DIRS})
    target_link_libraries(${PLUGIN_NAME_CPU} PRIVATE ${LIBXSMM_LIBRARIES})
  elseif(BUILD_LIBXSMM_VIA_CMAKE)
    add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/libxsmm cpu_third_party/libxsmm EXCLUDE_FROM_ALL)
    add_definitions(-DLIBXSMM_DEFAULT_CONFIG)
    set(LIBXSMM_INCLUDE_DIRS ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/libxsmm/include)
    target_link_libraries(${PLUGIN_NAME_CPU} PRIVATE xsmm)
  else()
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    set(args
    CC=${CMAKE_C_COMPILER}
    CXX=${CMAKE_CXX_COMPILER}
    FC=${CMAKE_Fortran_COMPILER}
    )

    ExternalProject_Add(libxsmm
      SOURCE_DIR ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/libxsmm
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        make
        "-j"
        ${args}
      INSTALL_COMMAND ""
      )
    set(LIBXSMM_INCLUDE_DIRS ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/libxsmm/include)
    target_link_libraries(${PLUGIN_NAME_CPU} PRIVATE ${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/libxsmm/lib/libxsmm.a)
  endif()
endif(USE_LIBXSMM)

if(USE_SYSTEM_SLEEF)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SLEEF REQUIRED sleef)
  target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${SLEEF_INCLUDE_DIRS})
  target_link_libraries(${PLUGIN_NAME_CPU} PRIVATE ${SLEEF_LIBRARIES})
else()
  # setup sleef options:
  set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef as static library" FORCE)
  set(SLEEF_BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
  set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
  set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
  set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "libsleefscalar will be built." FORCE)
  add_subdirectory(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/sleef ${THIRD_PARTY_BUILD_PATH_NAME}/sleef EXCLUDE_FROM_ALL)
  target_link_libraries(${PLUGIN_NAME_CPU} PRIVATE sleef)
endif()

if(NOT USE_SYSTEM_ONEDNN)
  add_dependencies(${PLUGIN_NAME_CPU} dnnl)
endif()

# If Graph Compiler is built, then it should link to its LLVM dependencies,
# and not the LLVM symbols exposed by PyTorch.
if ((DEFINED ENV{DNNL_GRAPH_BUILD_COMPILER_BACKEND}) AND USE_DNNL_GRAPH_COMPILER)
  get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB dnnl_graphcompiler_llvm_lib INTERFACE_LINK_LIBRARIES)
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC dnnl ${DNNL_GRAPHCOMPILER_LLVM_LIB})
  # BUILD_SHARED_LIBS is not used to control ipex library type, it is always shared
  if (NOT MSVC)
    get_target_property(DNNL_GRAPHCOMPILER_LLVM_LIB_EXCLUDE dnnl_graphcompiler_llvm_lib_exclude_string INTERFACE_LINK_LIBRARIES)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs=${DNNL_GRAPHCOMPILER_LLVM_LIB_EXCLUDE}")
  endif()
else()
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC ${ONEDNN_LIBRARY})
endif()
find_package(oneMKL QUIET)
if (ONEMKL_FOUND)
  target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${ONEMKL_INCLUDE_DIR})
  set_target_properties(${PLUGIN_NAME_CPU} PROPERTIES ONEMKL_INCLUDE_DIR "${ONEMKL_INCLUDE_DIR}")
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC ${ONEMKL_CPU_LIBS})
else()
  message(FATAL_ERROR "ERROR: Cannot find oneMKL!")
endif()

find_package(OMP QUIET)
if(${OpenMP_FOUND})
  # It will use openmp lib of current compiler.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC "${OpenMP_CXX_LIBRARIES}")
endif()

target_include_directories(${PLUGIN_NAME_CPU} PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_directories(${PLUGIN_NAME_CPU} PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC torch_cpu)
target_link_libraries(${PLUGIN_NAME_CPU} PUBLIC c10)

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

if(BUILD_STRIPPED_BIN)
  set_target_properties(${PLUGIN_NAME_CPU} PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

set(LIBRARIES_INSTALL)

install(TARGETS ${PLUGIN_NAME_CPU}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
list(APPEND LIBRARIES_INSTALL ${PLUGIN_NAME_CPU})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
  if(${OpenMP_FOUND})
    install(FILES ${OpenMP_CXX_LIBRARIES}
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  find_package(IcxCpuRt)
  if(${ICX_CPU_RT_FOUND})
    install(FILES ${INTEL_ICX_RT_LIBS}
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  
endif()

file(WRITE "${LIBRARIES_INSTALL_FILE_PREFIX}_cpu" "${LIBRARIES_INSTALL}")
