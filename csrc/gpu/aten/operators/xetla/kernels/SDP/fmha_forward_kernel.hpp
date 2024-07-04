
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/

#include <limits>
#include "../../mha.h"
#include "fmha_forward_policy.h"
#include "fmha_utils.h"
namespace gpu::xetla {

namespace fmha {
template <typename T>
struct dispatch_fmha_forward_args_t {
  T* query;
  T* key;
  T* value;
  T* alibi;
  T* attn_mask;
  uint8_t* dropout_mask;
  T* out;
  void* log_sumexp;
  float softmax_scale;
  float dropout_prob;
  int32_t* cu_seqlen_q;
  int32_t* cu_seqlen_k;
  uint32_t num_batches;
  uint32_t num_heads;
  uint32_t num_kv_heads;
  uint32_t head_size;
  uint32_t num_queries;
  uint32_t num_keys;
  uint32_t bias_strideB;
  uint32_t bias_strideN;
  uint32_t bias_strideF;
  uint32_t alibi_padded_block_size;
  uint32_t attn_mask_padded_block_size;
  uint64_t seed_t;
  uint64_t offset_t;
  dispatch_fmha_forward_args_t(const fmha_forward_kernel_args_t& args)
      : query(reinterpret_cast<T*>(args.query)),
        key(reinterpret_cast<T*>(args.key)),
        value(reinterpret_cast<T*>(args.value)),
        alibi(reinterpret_cast<T*>(args.alibi)),
        attn_mask(reinterpret_cast<T*>(args.attn_mask)),
        dropout_mask(reinterpret_cast<uint8_t*>(args.dropout)),
        out(reinterpret_cast<T*>(args.out)),
        log_sumexp(args.log_sumexp),
        softmax_scale(args.alpha),
        dropout_prob(args.dropout_prob),
        cu_seqlen_q(args.cu_seqlen_q),
        cu_seqlen_k(args.cu_seqlen_k),
        num_batches(args.num_batches),
        num_heads(args.num_heads),
        num_kv_heads(args.num_kv_heads),
        head_size(args.head_size),
        num_queries(args.num_queries),
        num_keys(args.num_keys),
        bias_strideB(args.bias_strideB),
        bias_strideN(args.bias_strideN),
        bias_strideF(args.bias_strideF),
        alibi_padded_block_size(args.alibi_padded_block_size),
        attn_mask_padded_block_size(args.attn_mask_padded_block_size),
        seed_t(args.seed_t),
        offset_t(args.offset_t){};
};

template <typename fmha_forward_op_t, typename T>
struct FmhaForwardKernelFunctor {
  KERNEL_MAIN void operator()(sycl::nd_item<3> item) const {
    // init fmha forward op and arguments
    fmha_forward_op_t fmha_fwd_op;
    using accscalar_t = fmha_forward_op_t::accum_t;
    typename fmha_forward_op_t::arguments_t op_args(
        args.query,
        args.key,
        args.value,
        args.alibi,
        args.attn_mask,
        args.dropout_mask,
        args.out,
        (accscalar_t*)args.log_sumexp,
        args.num_batches,
        args.num_heads,
        args.num_kv_heads,
        args.head_size,
        args.num_queries,
        args.num_keys,
        args.bias_strideB,
        args.bias_strideN,
        args.bias_strideF,
        args.cu_seqlen_q,
        args.cu_seqlen_k,
        (accscalar_t)args.softmax_scale,
        (accscalar_t)args.dropout_prob,
        args.alibi_padded_block_size,
        args.attn_mask_padded_block_size,
        args.seed_t,
        args.offset_t);

    // call the functor
    fmha_fwd_op(item, op_args);
  }
  FmhaForwardKernelFunctor(const dispatch_fmha_forward_args_t<T>& args)
      : args(args) {}

 private:
  dispatch_fmha_forward_args_t<T> args;
};

// The launcher of fmha forward kernel
template <
    typename fmha_policy,
    typename T,
    gpu_arch arch_tag,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsCausal,
    bool kSeqLast,
    bool kIsTraining,
    bool kIsDropout,
    bool kIsVarlen>
cgfs_t xetla_fmha_forward_kernel(const dispatch_fmha_forward_args_t<T>& args);

} // namespace fmha
} // namespace gpu::xetla
