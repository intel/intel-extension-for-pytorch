
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/

#include "fmha_forward.hpp"
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
    bool kIsDropout>
cgfs_t xetla_fmha_forward_kernel(const dispatch_fmha_forward_args_t<T>& args) {
#ifdef SDP_DBG
  printf(
      "B, N, Nkv, F, T, H: %u, %u, %u, %u, %u, %u, UseAlibi: %d, UseBias: %d, IsCausal: %d, IsTraining: %d,"
      "IsDropout: %d, alibi @ 0x%llx, uAT %d, uMT %d, strideB %d, strideN %d, strideF %d, dropout_prob %f, kSeqLast %d\n",
      args.num_batches,
      args.num_heads,
      args.num_kv_heads,
      args.num_queries,
      args.num_keys,
      args.head_size,
      kUseAlibi,
      kUseBias,
      kIsCausal,
      kIsTraining,
      kIsDropout,
      (unsigned long long)args.alibi,
      args.alibi_padded_block_size,
      args.attn_mask_padded_block_size,
      args.bias_strideB,
      args.bias_strideN,
      args.bias_strideF,
      args.dropout_prob,
      kSeqLast);
#endif
  // fmha forward kernel
  using fmha_forward_op_t = fmha_forward_t<
      fmha_policy,
      T,
      arch_tag,
      kUseAlibi,
      kUseBias,
      kIsCausal,
      kSeqLast,
      kIsTraining,
      kIsDropout>;

  sycl::nd_range<3> NdRange = fmha_forward_op_t::get_nd_range(
      args.num_batches * args.num_heads, args.num_queries);

  FmhaForwardKernelFunctor<fmha_forward_op_t, T> kfn(args);
  return {[=](sycl::handler& cgh) { cgh.parallel_for(NdRange, kfn); }};
}

} // namespace fmha

/// @brief Main execution function for flash mha forward.
template <
    typename T,
    gpu_arch arch_tag,
    bool kUseAlibi = false,
    bool kUseBias = false,
    bool kIsCausal = false,
    bool kSeqLast = false,
    bool kIsTraining = false,
    bool kIsDropout = false>
class fmha_forward_kernel_policy {
  template <typename fmha_policy, typename... Args>
  static cgfs_t policy(Args&&... args) {
    // check for param pack tricks: https://stackoverflow.com/a/2821244/9817693
    return fmha::xetla_fmha_forward_kernel<
        fmha_policy,
        T,
        arch_tag,
        kUseAlibi,
        kUseBias,
        kIsCausal,
        kSeqLast,
        kIsTraining,
        kIsDropout>(std::forward<Args>(args)...);
  }

 public:
  static cgfs_t run(const fmha::dispatch_fmha_forward_args_t<T>& args) {
#ifdef SDP_DBG
    printf("\n%s\n", __PRETTY_FUNCTION__);
#endif

    // Xetla dropout requires the same policy for fwd and bwd
    if (kIsTraining && kIsDropout && !args.dropout_mask) {
      if constexpr (kIsTraining && kIsDropout && arch_tag == gpu_arch::XeHpc) {
        if (args.head_size <= 64) {
          return policy<fmha_policy_64x64x64>(args);
        } else if (args.head_size <= 128) {
          return policy<fmha_policy_128x128x128>(args);
        } else if (args.head_size <= 256) {
          return policy<fmha_policy_128x128x256>(args);
        } else if (args.head_size <= 512) {
          return policy<fmha_policy_64x128x512>(args);
        } else {
          assert(false);
          return {};
        }
      } else {
        std::cout << "BWD only available on PVC\n";
        return {};
      }
    } else {
      // roughly policy should match (num_queries)x(num_keys)x(head_size)
      if (args.head_size <= 64) {
        if (args.num_queries < 64) {
          // for short query length
          return policy<fmha_policy_8x128x64>(args);
        } else {
          // for long query length
          return policy<fmha_policy_64x128x64>(args);
        }
      } else if (args.head_size <= 128) {
        constexpr bool igpu_wo_dropout =
            arch_tag == gpu_arch::XeLpg && !kIsDropout;
        if (igpu_wo_dropout && args.num_queries == 1) {
          // for extreamly short query length
          if constexpr (igpu_wo_dropout) {
            if (args.num_keys < 512) {
              return policy<stage0<fmha_policy_1x256x128>>(args);
            } else {
              return policy<stage0<fmha_policy_1x512x128>>(args);
            }
          }
          return {};
        } else if (args.num_queries < 64) {
          // for short query length
          if (args.num_keys < 512) {
            return policy<fmha_policy_8x256x128>(args);
          } else {
            return policy<fmha_policy_8x512x128>(args);
          }
        } else {
          // for long query length
          if constexpr (arch_tag == gpu_arch::XeLpg)
            return policy<stage0<fmha_policy_32x128x128>>(args);
          else
            return policy<fmha_policy_64x128x128>(args);
        }
      } else if constexpr (arch_tag == gpu_arch::XeLpg) {
        std::cout << "Larger head_size are experiencing problems on MTL...\n";
        assert(false);
        return {};
      } else if (args.head_size <= 256) {
        if (args.num_queries < 64) {
          // for short query length
          return policy<fmha_policy_8x256x256>(args);
        } else {
          // for long query length
          if (arch_tag != gpu_arch::XeHpc || args.num_keys < 128) {
            return policy<fmha_policy_64x128x256>(args);
          } else {
            if constexpr (arch_tag == gpu_arch::XeHpc)
              return policy<fmha_policy_64x256x256>(args);
          }
        }
      } else if (arch_tag == gpu_arch::XeHpc && args.head_size <= 512) {
        if constexpr (arch_tag == gpu_arch::XeHpc)
          return policy<fmha_policy_64x128x512>(args);
        return {};
      } else {
        assert(false);
        return {};
      }
    }
  };
};

template <typename T, gpu_arch arch_tag, bool... Bs>

cgfs_t dispatch_fmha_forward(
    const fmha::dispatch_fmha_forward_args_t<T>& args) {
  return fmha_forward_kernel_policy<T, arch_tag, Bs...>::run(args);
}

// dispatch different conditions
template <typename T, gpu_arch arch_tag, bool... Bs, typename... Ts>
cgfs_t dispatch_fmha_forward(
    const fmha::dispatch_fmha_forward_args_t<T>& args,
    bool b,
    Ts... ts) {
  if (b) {
    return dispatch_fmha_forward<T, arch_tag, Bs..., true>(args, ts...);
  } else {
    return dispatch_fmha_forward<T, arch_tag, Bs..., false>(args, ts...);
  }
}

// dispatch datatype
template <gpu_arch arch_tag>
cgfs_t _fmha_forward_kernel(
    XetlaType xeType,
    const fmha_forward_kernel_args_t& args) {
  if (xeType == XetlaType::fp16) {
    return dispatch_fmha_forward<fp16, arch_tag>(
        fmha::dispatch_fmha_forward_args_t<fp16>(args),
        args.alibi != nullptr, // is alibi
        args.attn_mask != nullptr, // is is_attn_mask
        args.is_causal,
        args.seq_last,
        args.is_training,
        args.is_dropout);
  } else if constexpr (arch_tag != gpu_arch::XeLpg) {
    return dispatch_fmha_forward<bf16, arch_tag>(
        fmha::dispatch_fmha_forward_args_t<bf16>(args),
        args.alibi != nullptr, // is alibi
        args.attn_mask != nullptr, // is is_attn_mask
        args.is_causal,
        args.seq_last,
        args.is_training,
        args.is_dropout);
  } else {
    printf("No bf16 for igpu!!\n\n");
    return {};
  }
}

// dispatch arch
XETLA_KERNEL_API cgfs_t fmha_forward_kernel(
    gpu_arch arch,
    XetlaType xeType,
    const fmha_forward_kernel_args_t& args) {
  switch (arch) {
#ifdef USE_XETLA_XE_LPG
    case gpu_arch::XeLpg:
      return _fmha_forward_kernel<gpu_arch::XeLpg>(xeType, args);
#endif
#ifdef USE_XETLA_XE_HPG
    case gpu_arch::XeHpg:
      // TODO(Yi): fix XeHpg
      return _fmha_forward_kernel<gpu_arch::XeLpg>(xeType, args);
#endif
#ifdef USE_XETLA_XE_HPC
    case gpu_arch::XeHpc:
      return _fmha_forward_kernel<gpu_arch::XeHpc>(xeType, args);
#endif
    default:
      printf("Unsupported gpu_arch of fmha_forward!!\n\n");
      return {};
  }
}
} // namespace gpu::xetla