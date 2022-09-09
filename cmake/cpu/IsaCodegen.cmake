include(CheckCXXCompilerFlag)

FIND_PACKAGE(AVX)

file(GLOB_RECURSE cpu_kernel_cpp_in "${IPEX_CPU_CPP_ROOT}/aten/kernels/*.cpp")
list(APPEND IPEX_CPU_CPP_ISA_SRCS_ORIGIN ${cpu_kernel_cpp_in})

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

if(CXX_AMX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AMX_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AMX")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX512F__ -DCPU_CAPABILITY_AVX512 -DCPU_CAPABILITY_AVX512_VNNI \
    -DCPU_CAPABILITY_AVX512_BF16 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mfma \
    -mamx-tile -mamx-int8 -mamx-bf16")
  endif(MSVC)
else(CXX_AMX_FOUND)
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "WARNING! Please upgrade gcc version to 11.2+ to support CPU ISA AMX.")
  endif(CMAKE_COMPILER_IS_GNUCXX)
endif(CXX_AMX_FOUND)

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
else(CXX_AVX512_BF16_FOUND)
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "WARNING! Please upgrade gcc version to 10.3+ to support CPU ISA AVX512_BF16.")
  endif(CMAKE_COMPILER_IS_GNUCXX)  
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
else(CXX_AVX512_VNNI_FOUND)
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "WARNING! Please upgrade gcc version to 9.2+ to support CPU ISA AVX512_VNNI.")
  endif(CMAKE_COMPILER_IS_GNUCXX)   
endif(CXX_AVX512_VNNI_FOUND)

if(CXX_AVX512_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX512")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX512F__ -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma")
  endif(MSVC)
else(CXX_AVX512_FOUND)
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "WARNING! Please upgrade gcc version to 9.2+ to support CPU ISA AVX512.")
  endif(CMAKE_COMPILER_IS_GNUCXX)   
endif(CXX_AVX512_FOUND)

if(CXX_AVX2_VNNI_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_VNNI_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX2_VNNI")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX__ -DCPU_CAPABILITY_AVX2 -mavx2 -mavxvnni -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
  endif(MSVC)
else(CXX_AVX2_VNNI_FOUND)
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "WARNING! Please upgrade gcc version to 11.2+ to support CPU ISA AVX2 VNNI.")
  endif(CMAKE_COMPILER_IS_GNUCXX)
endif(CXX_AVX2_VNNI_FOUND)

if(CXX_AVX2_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX2")
  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2") # TODO: CHECK HERE
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -D__AVX__ -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
  endif(MSVC)
endif(CXX_AVX2_FOUND)

list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

# The sources list might get reordered later based on the capabilites.
# See NOTE [ Linking AVX and non-AVX files ]
foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
  foreach(IMPL ${cpu_kernel_cpp_in})
    file(RELATIVE_PATH NAME "${IPEX_PROJECT_TOP_DIR}/csrc/" "${IMPL}")
    list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
    set(NEW_IMPL ${CMAKE_BINARY_DIR}/isa_codegen/${NAME}.${CPU_CAPABILITY}.cpp)
    configure_file("${IPEX_PROJECT_TOP_DIR}/cmake/cpu/IncludeSource.cpp.in" ${NEW_IMPL})
    set(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp}) # Create list of copies
    list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    if(MSVC)
      set(EXTRA_FLAGS "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
    else(MSVC)
      set(EXTRA_FLAGS "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
    endif(MSVC)
    set_source_files_properties(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${EXTRA_FLAGS}")
    # message("SRC: ${NEW_IMPL} FLAG: ${FLAGS} ${EXTRA_FLAGS}")
  endforeach()
endforeach()

list(APPEND IPEX_CPU_CPP_ISA_SRCS_GEN ${cpu_kernel_cpp})
