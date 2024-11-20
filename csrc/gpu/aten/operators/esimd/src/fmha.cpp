#include "fmha.hpp"

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
    bool is_head_first) {
  // microsoft phi3 model
  if (num_heads_q == 32 && num_heads_k == 32 && head_size == 96) {
    if (is_head_first)
      return launch_phi_fused_mha<true, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
    else
      return launch_phi_fused_mha<false, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
  }
  // microsoft phi3-small model
  if (num_heads_q == 32 && num_heads_k == 8 && head_size == 128) {
    if (is_head_first)
      return launch_phi3small_fused_mha<true, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
    else
      return launch_phi3small_fused_mha<false, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
  } else {
    std::cout
        << "----------Error in mha: please add policy for currently running "
           "model-----------"
        << std::endl;
    // TODO(zw): exception handle heres
    return {};
  }
}

} // namespace esimd
