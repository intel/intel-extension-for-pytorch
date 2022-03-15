include(CheckCXXCompilerFlag)

FIND_PACKAGE(AVX)

if(MSVC)
  set(OPT_FLAG "/fp:strict ")
else(MSVC)
  set(OPT_FLAG "-O3 ")
  if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
    set(OPT_FLAG " ")
  endif()
endif(MSVC)

#[[
if(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/MapAllocator.cpp PROPERTIES COMPILE_FLAGS "-fno-openmp")
endif()
]]

file(GLOB_RECURSE cpu_kernel_cpp_in "${PROJECT_SOURCE_DIR}/intel_extension_for_pytorch/csrc/aten/cpu/kernels/*.cpp")

list(APPEND DPCPP_ISA_SRCS_ORIGIN ${cpu_kernel_cpp_in})

# foreach(file_path ${cpu_kernel_cpp_in})
#   message(${file_path})
# endforeach()


# Some versions of GCC pessimistically split unaligned load and store
# instructions when using the default tuning. This is a bad choice on
# new Intel and AMD processors so we disable it when compiling with AVX2.
# See https://stackoverflow.com/questions/52626726/why-doesnt-gcc-resolve-mm256-loadu-pd-as-single-vmovupd#tab-top
check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" COMPILER_SUPPORTS_NO_AVX256_SPLIT)
if(COMPILER_SUPPORTS_NO_AVX256_SPLIT)
  set(CPU_NO_AVX256_SPLIT_FLAGS "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
endif(COMPILER_SUPPORTS_NO_AVX256_SPLIT)

# Keep Default config to align to pytorch, but use AVX2 parameters as its real implement.
list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
if(MSVC)
list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2") # TODO: CHECK HERE
else(MSVC)
list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX__ -DCPU_CAPABILITY_AVX2 -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
endif(MSVC)

if(CXX_AVX512_BF16_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_BF16_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX512_BF16")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX512F__ -DCPU_CAPABILITY_AVX512 \
    -DCPU_CAPABILITY_AVX512_VNNI -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni \
    -mavx512bf16 -mfma")
  endif(MSVC)
endif(CXX_AVX512_BF16_FOUND)

if(CXX_AVX512_VNNI_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_VNNI_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX512_VNNI")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX512F__ -DCPU_CAPABILITY_AVX512 \
     -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma")
  endif(MSVC)
endif(CXX_AVX512_VNNI_FOUND)

if(CXX_AVX512_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX512")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX512F__ -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma")
  endif(MSVC)
endif(CXX_AVX512_FOUND)

if(CXX_AVX2_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX2")
  if(DEFINED ENV{ATEN_AVX512_256})
    if($ENV{ATEN_AVX512_256} MATCHES "TRUE")
      if(CXX_AVX512_FOUND)
        message("-- ATen AVX2 kernels will use 32 ymm registers")
        if(MSVC)
          list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512")
        else(MSVC)
          list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -march=native ${CPU_NO_AVX256_SPLIT_FLAGS}")
        endif(MSVC)
      endif(CXX_AVX512_FOUND)
    endif()
  else()
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2") # TODO: CHECK HERE
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX__ -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
    endif(MSVC)
  endif()
endif(CXX_AVX2_FOUND)

list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

# The sources list might get reordered later based on the capabilites.
# See NOTE [ Linking AVX and non-AVX files ]
foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
  foreach(IMPL ${cpu_kernel_cpp_in})
    file(RELATIVE_PATH NAME "${PROJECT_SOURCE_DIR}/intel_extension_for_pytorch/csrc/" "${IMPL}")
    list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
    set(NEW_IMPL ${CMAKE_BINARY_DIR}/intel_extension_for_pytorch/csrc/${NAME}.${CPU_CAPABILITY}.cpp)
    configure_file("${PROJECT_SOURCE_DIR}/cmake/IncludeSource.cpp.in" ${NEW_IMPL})
    set(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp}) # Create list of copies
    list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    if(MSVC)
      set(EXTRA_FLAGS "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
    else(MSVC)
      set(EXTRA_FLAGS "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
    endif(MSVC)
    # Disable certain warnings for GCC-9.X
    #[[
    if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
      if(("${NAME}" STREQUAL "native/cpu/GridSamplerKernel.cpp") AND ("${CPU_CAPABILITY}" STREQUAL "DEFAULT"))
        # See https://github.com/pytorch/pytorch/issues/38855
        set(EXTRA_FLAGS "${EXTRA_FLAGS} -Wno-uninitialized")
      endif()
      if("${NAME}" STREQUAL "native/quantized/cpu/kernels/QuantizedOpKernels.cpp")
        # See https://github.com/pytorch/pytorch/issues/38854
        set(EXTRA_FLAGS "${EXTRA_FLAGS} -Wno-deprecated-copy")
      endif()
    endif()
    ]]
    set_source_files_properties(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${EXTRA_FLAGS}")
    # message(${NEW_IMPL} - ${FLAGS})
  endforeach()
endforeach()

list(APPEND DPCPP_ISA_SRCS ${cpu_kernel_cpp})

# foreach(file_path ${cpu_kernel_cpp})
#   message(${file_path})
# endforeach()

