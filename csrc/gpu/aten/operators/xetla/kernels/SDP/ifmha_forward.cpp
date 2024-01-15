#include "ifmha_forward.h"

namespace gpu::xetla {
namespace fmha {

template <
    typename ifmha_policy,
    typename T,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsTraining>
class IfmhaForwardKernel;

template <typename ifmha_forward_op_t, typename T>
struct IfmhaForwardImplKernelFunctor {
  SYCL_ESIMD_KERNEL void operator()(sycl::nd_item<2> item) const {
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
        alibi,
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
        alibi_padding,
        attn_mask_padding);

    // call the functor
    ifmha_fwd_op(ei, args);
  }
  IfmhaForwardImplKernelFunctor(
      T* query_,
      T* key0_,
      T* key1_,
      T* value0_,
      T* value1_,
      int32_t* index_,
      T* alibi_,
      T* bias_,
      uint8_t* dropout_,
      float dropout_prob_,
      float sm_scale_,
      T* out_,
      uint32_t num_batches_,
      uint32_t beam_,
      uint32_t num_heads_,
      uint32_t head_size_,
      uint32_t kv_len0_,
      uint32_t kv_len1_,
      uint32_t alibi_padding_,
      uint32_t attn_mask_padding_)
      : query(query_),
        key0(key0_),
        key1(key1_),
        value0(value0_),
        value1(value1_),
        index(index_),
        alibi(alibi_),
        bias(bias_),
        dropout(dropout_),
        dropout_prob(dropout_prob_),
        sm_scale(sm_scale_),
        out(out_),
        num_batches(num_batches_),
        beam(beam_),
        num_heads(num_heads_),
        head_size(head_size_),
        kv_len0(kv_len0_),
        kv_len1(kv_len1_),
        alibi_padding(alibi_padding_),
        attn_mask_padding(attn_mask_padding_) {}

 private:
  T* query;
  T* key0;
  T* key1;
  T* value0;
  T* value1;
  int32_t* index;
  T* alibi;
  T* bias;
  uint8_t* dropout;
  float dropout_prob;
  float sm_scale;
  T* out;
  uint32_t num_batches;
  uint32_t beam;
  uint32_t num_heads;
  uint32_t head_size;
  uint32_t kv_len0;
  uint32_t kv_len1;
  uint32_t alibi_padding;
  uint32_t attn_mask_padding;
};

// The launcher of indexed flash mha forward kernel
template <
    typename ifmha_policy,
    typename T,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsTraining>
void ifmha_forward_impl(
    sycl::queue& q,
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* alibi,
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
    uint32_t alibi_padding,
    uint32_t attn_mask_padding) {
#ifdef SDP_DBG
  printf(
      "B, Bm, N, F, T0, T1, H: %d, %d, %d, %d, %d, %d, %d, UseAlibi: %d, UseBias: %d, IsTraining: %d, uAT %d, uPT %d, scale %f alibi @ 0x%llx\n",
      num_batches,
      beam,
      num_heads,
      1,
      kv_len0,
      kv_len1,
      head_size,
      kUseAlibi,
      kUseBias,
      kIsTraining,
      alibi_padding,
      attn_mask_padding,
      sm_scale,
      (unsigned long long)alibi);
#endif
  // ifmha forward kernel
  using ifmha_forward_op_t =
      ifmha_forward_t<ifmha_policy, T, kUseAlibi, kUseBias, kIsTraining>;

  sycl::nd_range<2> NdRange =
      ifmha_forward_op_t::get_nd_range(num_batches, beam, num_heads);

  auto cgf = DPCPP_Q_CGF(cgh) {
    IfmhaForwardImplKernelFunctor<ifmha_forward_op_t, T> kfn(
        query,
        key0,
        key1,
        value0,
        value1,
        index,
        alibi,
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
        alibi_padding,
        attn_mask_padding);
    cgh.parallel_for<class IfmhaForwardKernel<
        ifmha_policy,
        T,
        kUseAlibi,
        kUseBias,
        kIsTraining>>(NdRange, kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

} // namespace fmha

#define CALL_IMPL_FUNC(P)                                           \
  fmha::ifmha_forward_impl<P, T, kUseAlibi, kUseBias, kIsTraining>( \
      q,                                                            \
      query,                                                        \
      key0,                                                         \
      key1,                                                         \
      value0,                                                       \
      value1,                                                       \
      index,                                                        \
      alibi,                                                        \
      bias,                                                         \
      dropout,                                                      \
      dropout_prob,                                                 \
      sm_scale,                                                     \
      out,                                                          \
      num_batches,                                                  \
      beam,                                                         \
      num_heads,                                                    \
      head_size,                                                    \
      kv_len0,                                                      \
      kv_len1,                                                      \
      alibi_padding,                                                \
      attn_mask_padding)

/// @brief Main execution function for indexed flash mha forward.
template <typename T, bool kUseAlibi, bool kUseBias, bool kIsTraining>
void ifmha_forward(
    sycl::queue& q,
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* alibi,
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
    uint32_t alibi_padding,
    uint32_t attn_mask_padding) {
  // occupancy first
  constexpr int hardware_concurrent_wg = 64;
  if (head_size <= 64) {
    CALL_IMPL_FUNC(ifmha_policy_64x64);
  } else if (head_size <= 128) {
    CALL_IMPL_FUNC(ifmha_policy_128x64);
  } else if (head_size <= 256) {
    CALL_IMPL_FUNC(ifmha_policy_256x64);
  } else {
    TORCH_CHECK(0, "SDP Index fusion kernel requires head_dim <= 256 ...");
    return;
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
    uint32_t alibi_padding,
    uint32_t attn_mask_padding,
    bool is_causal) {
  using T = sycl::half;
  TORCH_CHECK(
      num_queries == 1,
      "SDP Index fusion kernel requires num_queries == 1 so far ...");
  TORCH_CHECK(
      is_causal == false,
      "SDP Index fusion kernel doesn't support causal so far ...");

#define DISPATCH_TEMPLATE(T, USE_ALIBI, USE_BIAS, IS_TRAINING) \
  ifmha_forward<T, USE_ALIBI, USE_BIAS, IS_TRAINING>(          \
      q,                                                       \
      (T*)query,                                               \
      (T*)key,                                                 \
      (T*)key_cache,                                           \
      (T*)value,                                               \
      (T*)value_cache,                                         \
      index,                                                   \
      (T*)alibi,                                               \
      (T*)attn_mask,                                           \
      dropout,                                                 \
      dropout_p,                                               \
      alpha,                                                   \
      (T*)out,                                                 \
      num_batches,                                             \
      beam_width,                                              \
      num_heads,                                               \
      head_dim,                                                \
      num_keys_in,                                             \
      num_keys_out,                                            \
      alibi_padding,                                           \
      attn_mask_padding);

  if (alibi) {
    if (attn_mask) {
      DISPATCH_TEMPLATE(T, true, true, false)
    } else {
      DISPATCH_TEMPLATE(T, true, false, false)
    }
  } else {
    if (attn_mask) {
      DISPATCH_TEMPLATE(T, false, true, false)
    } else {
      DISPATCH_TEMPLATE(T, false, false, false)
    }
  }
}
} // namespace gpu::xetla
