
#pragma once

#include "./src/kernel_apis.hpp"

namespace esimd {
ESIMD_KERNEL_API cgf_t launch_fused_mha(
    sycl::half* query,
    sycl::half* key,
    sycl::half* value,
    sycl::half* output,
    uint8_t* mask,
    uint32_t num_batches,
    uint32_t num_heads_q,
    uint32_t num_heads_k,
    uint32_t head_size,
    uint32_t qo_len,
    uint32_t kv_len,
    bool is_head_first);

}