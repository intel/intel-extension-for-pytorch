set(DS_TARGET intel-ext-pt-gpu-deepspeed)

set(deepspeed_quantization_src)
FILE(GLOB deepspeed_quantization_src "${IPEX_GPU_DEEPSPEED_SRC_DIR}/quantization/*.cpp")

set(deepspeed_transformer_inference_src)
FILE(GLOB deepspeed_transformer_inference_src "${IPEX_GPU_DEEPSPEED_SRC_DIR}/transformer/inference/csrc/*.cpp")

set(deepspeed_inc)
list(APPEND deepspeed_inc "${IPEX_GPU_DEEPSPEED_SRC_DIR}/includes")

set(IPEX_DEEPSPEED_SRCS ${deepspeed_quantization_src} ${deepspeed_transformer_inference_src})
set(IPEX_DEEPSPEED_INCLUDES ${deepspeed_inc})

add_library(${DS_TARGET} SHARED ${IPEX_DEEPSPEED_SRCS})

target_include_directories(${DS_TARGET} PRIVATE ${IPEX_DEEPSPEED_INCLUDES})
target_link_libraries(${DS_TARGET} PRIVATE ${GPU_TORCH_LIBS})
target_link_directories(${DS_TARGET} PRIVATE ${SYCL_LIBRARY_DIR})
target_link_options(${DS_TARGET} PRIVATE ${IPEX_SYCL_LINK_FLAGS})
target_compile_options(${DS_TARGET} PRIVATE ${IPEX_SYCL_KERNEL_FLAGS})
