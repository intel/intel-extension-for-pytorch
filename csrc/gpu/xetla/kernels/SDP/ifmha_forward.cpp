#include "ifmha_forward.h"

namespace gpu::xetla {
namespace fmha {

template <typename ifmha_policy, typename T, bool kUseBias, bool kIsTraining>
class IfmhaForwardKernel;

// The launcher of indexed flash mha forward kernel
template <typename ifmha_policy, typename T, bool kUseBias, bool kIsTraining>
void ifmha_forward_impl(
    sycl::queue& q,
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* bias,
    uint8_t* dropout,
    float dropout_prob,
    float sm_scale,
    T* out,
    uint32_t num_batches,
    uint32_t beam,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t kv_len0,
    uint32_t kv_len1,
    uint32_t attn_mask_padding) {
#ifdef SDP_DBG
  printf(
      "B, Bm, N, F, T0, T1, H: %d, %d, %d, %d, %d, %d, %d, UseBias: %d, IsTraining: %d, uPT %d, scale %f\n",
      num_batches,
      beam,
      num_heads,
      1,
      kv_len0,
      kv_len1,
      head_size,
      kUseBias,
      kIsTraining,
      attn_mask_padding,
      sm_scale);
#endif
  // ifmha forward kernel
  using ifmha_forward_op_t =
      ifmha_forward_t<ifmha_policy, T, kUseBias, kIsTraining>;

  sycl::nd_range<2> NdRange =
      ifmha_forward_op_t::get_nd_range(num_batches * beam * num_heads);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        class IfmhaForwardKernel<ifmha_policy, T, kUseBias, kIsTraining>>(
        NdRange, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
      // exec item
      xetla_exec_item<2> ei(item);

      // init ifmha forward op and arguments
      ifmha_forward_op_t ifmha_fwd_op;
      typename ifmha_forward_op_t::arguments_t args(
          query,
          key0,
          key1,
          value0,
          value1,
          index,
          bias,
          dropout,
          dropout_prob,
          sm_scale,
          out,
          num_batches,
          beam,
          num_heads,
          head_size,
          kv_len0,
          kv_len1,
          attn_mask_padding);

      // call the functor
      ifmha_fwd_op(ei, args);
        });
        // event.wait();
        // double time = (event.template get_profiling_info<
        //                    sycl::info::event_profiling::command_end>() -
        //                event.template get_profiling_info<
        //                    sycl::info::event_profiling::command_start>());
        // uint64_t ops = num_batches * num_heads *
        //                uint64_t(head_size * num_queries * num_keys) * 2L *
        //                2L;
        // double tflops = (ops / 1024.0f / 1024.0f / 1024.0f / 1024.0f) / (time
        // / 1e9); printf("B, N, F, T, H: %d, %d, %d, %d, %d, time: %f us,
        // tflops: %f\n",
        //        num_batches, num_heads, num_queries, num_keys, head_size, time
        //        / 1e3, tflops);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

} // namespace fmha

#define CALL_IMPL_FUNC(P)                                \
  fmha::ifmha_forward_impl<P, T, kUseBias, kIsTraining>( \
      q,                                                 \
      query,                                             \
      key0,                                              \
      key1,                                              \
      value0,                                            \
      value1,                                            \
      index,                                             \
      bias,                                              \
      dropout,                                           \
      dropout_prob,                                      \
      sm_scale,                                          \
      out,                                               \
      num_batches,                                       \
      beam,                                              \
      num_heads,                                         \
      head_size,                                         \
      kv_len0,                                           \
      kv_len1,                                           \
      attn_mask_padding)

/// @brief Main execution function for indexed flash mha forward.
template <typename T, bool kUseBias = false, bool kIsTraining = false>
void ifmha_forward(
    sycl::queue& q,
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* bias,
    uint8_t* dropout,
    float dropout_prob,
    float sm_scale,
    T* out,
    uint32_t num_batches,
    uint32_t beam,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t kv_len0,
    uint32_t kv_len1,
    uint32_t attn_mask_padding) {
  // occupancy first
  constexpr int hardware_concurrent_wg = 64;
  if (num_batches * beam <= hardware_concurrent_wg) {
    if (head_size <= 64) {
      CALL_IMPL_FUNC(ifmha_policy_64x64);
    } else if (head_size <= 128) {
      CALL_IMPL_FUNC(ifmha_policy_128x64);
    } else if (head_size <= 256) {
      CALL_IMPL_FUNC(ifmha_policy_s_256x64);
    } else {
      TORCH_CHECK(0, "SDP Index fusion kernel requires head_dim <= 256 ...");
      return;
    }
  } else {
    if (head_size <= 64) {
      CALL_IMPL_FUNC(ifmha_policy_64x64);
    } else if (head_size <= 128) {
      CALL_IMPL_FUNC(ifmha_policy_128x64);
    } else if (head_size <= 256) {
      CALL_IMPL_FUNC(ifmha_policy_l_256x64);
    } else {
      TORCH_CHECK(0, "SDP Index fusion kernel requires head_dim <= 256 ...");
      return;
    }
  }
}

#undef CALL_IMPL_FUNC

void fmha_forward_index_kernel(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* key_cache,
    void* value_cache,
    int32_t* index,
    void* alibi,
    void* attn_mask,
    uint8_t* dropout,
    void* out,
    uint32_t timestep,
    float alpha,
    float beta,
    float dropout_p,
    uint32_t num_batches,
    uint32_t beam_width,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_queries,
    uint32_t num_keys_in,
    uint32_t num_keys_out,
    uint32_t attn_mask_padding,
    bool is_causal) {
  using T = sycl::half;
  TORCH_CHECK(
      num_queries == 1,
      "SDP Index fusion kernel requires num_queries == 1 so far ...");
  TORCH_CHECK(
      is_causal == false,
      "SDP Index fusion kernel doesn't support causal so far ...");
  TORCH_CHECK(
      alibi == nullptr,
      "SDP Index fusion kernel doesn't support alibi so far ...");

  if (attn_mask) {
    ifmha_forward<T, true, false>(
        q,
        (T*)query,
        (T*)key,
        (T*)key_cache,
        (T*)value,
        (T*)value_cache,
        index,
        (T*)attn_mask,
        dropout,
        dropout_p,
        alpha,
        (T*)out,
        num_batches,
        beam_width,
        num_heads,
        head_dim,
        num_keys_in,
        num_keys_out,
        attn_mask_padding);
  } else {
    ifmha_forward<T, false, false>(
        q,
        (T*)query,
        (T*)key,
        (T*)key_cache,
        (T*)value,
        (T*)value_cache,
        index,
        (T*)attn_mask,
        dropout,
        dropout_p,
        alpha,
        (T*)out,
        num_batches,
        beam_width,
        num_heads,
        head_dim,
        num_keys_in,
        num_keys_out,
        attn_mask_padding);
  }
}
} // namespace gpu::xetla
