set(BNB_TARGET intel-ext-pt-gpu-bitsandbytes)

set(bitsandbytes_optimizer_src)
FILE(GLOB bitsandbytes_optimizer_src "${IPEX_GPU_BITSANDBYTES_SRC_DIR}/optimizer/*.cpp")

set(bitsandbytes_inc)
list(APPEND bitsandbytes_inc "${IPEX_GPU_BITSANDBYTES_SRC_DIR}/includes")

set(IPEX_BITSANDBYTES_SRCS ${bitsandbytes_optimizer_src})
set(IPEX_BITSANDBYTES_INCLUDES ${bitsandbytes_inc})

add_library(${BNB_TARGET} SHARED ${IPEX_BITSANDBYTES_SRCS})

set_target_properties(${BNB_TARGET} PROPERTIES CXX_VISIBILITY_PRESET hidden CXX_VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(${BNB_TARGET} PRIVATE ${IPEX_BITSANDBYTES_INCLUDES})
target_link_libraries(${BNB_TARGET} PRIVATE ${GPU_TORCH_LIBS})
target_link_directories(${BNB_TARGET} PRIVATE ${SYCL_LIBRARY_DIR})
target_link_options(${BNB_TARGET} PRIVATE ${IPEX_SYCL_LINK_FLAGS})
target_compile_options(${BNB_TARGET} PRIVATE ${IPEX_SYCL_KERNEL_FLAGS})