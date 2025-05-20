
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/

#include <limits>
#include "../../mha.h"
#include "fmha_forward_kernel.hpp"
#include "fmha_forward_policy.h"
#include "fmha_utils.h"

// 4 configurations for head size
#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 128
#define HEAD_SIZE_LIMIT_2 256
#define HEAD_SIZE_LIMIT_3 512
#define NUM_QUERIES_GREEDY 1
#define NUM_QUERIES_LARGE 64
#define NUM_KEYS_LIMIT_0 128
#define NUM_KEYS_LIMIT_1 512

namespace gpu::xetla {

/// @brief Main execution function for flash mha forward.
template <
    typename T,
    gpu_arch arch_tag,
    bool kUseAlibi = false,
    bool kUseBias = false,
    bool kIsCausal = false,
    bool kSeqLast = false,
    bool kIsTraining = false,
    bool kIsDropout = false,
    bool kIsVarlen = false,
    bool kIsLocal = false>
class fmha_forward_kernel_policy {
  template <typename fmha_policy, typename... Args>
  static cgfs_t policy(Args&&... args) {
    if constexpr (arch_tag == gpu_arch::XeHpc) {
      if (is_chunked_prefill(std::forward<Args>(args)...)) {
        return chunked_prefill_wrapper<fmha_policy>(
            std::forward<Args>(args)...);
      }
    }
    // check for param pack tricks: https://stackoverflow.com/a/2821244/9817693
    return kernel_call<fmha_policy>(std::forward<Args>(args)...);
  }
  template <typename fmha_policy, typename... Args>
  static cgfs_t kernel_call(Args&&... args) {
    // std::cout << "before real call" << std::endl;
    return fmha::xetla_fmha_forward_kernel<
        fmha_policy,
        T,
        arch_tag,
        kUseAlibi,
        kUseBias,
        kIsCausal,
        kSeqLast,
        kIsTraining,
        kIsDropout,
        kIsVarlen,
        kIsLocal>(std::forward<Args>(args)...);
  }

 public:
  static bool is_chunked_prefill(
      const fmha::dispatch_fmha_forward_args_t<T>& args) {
    return args.block_tables != nullptr;
  }

  template <typename fmha_policy>
  static cgfs_t chunked_prefill_wrapper(
      const fmha::dispatch_fmha_forward_args_t<T>& args) {
    // std::cout << "block tables: " << (args.block_tables == nullptr)
    // << std::endl;
    if (args.block_size == 64) {
      // std::cout << "block size 64" << std::endl;
      if (args.head_size <= HEAD_SIZE_LIMIT_0) {
        return kernel_call<fmha_policy_64x64x64>(args);
      } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
        return kernel_call<fmha_policy_64x64x128>(args);
      } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
        return kernel_call<fmha_policy_64x64x256>(args);
      } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
        return kernel_call<fmha_policy_64x64x512>(args);
      } else {
        assert(false);
        return {};
      }
    } else if (args.block_size == 128) {
      // std::cout << "block size 128 " << std::endl;
      if (args.head_size <= HEAD_SIZE_LIMIT_0) {
        return kernel_call<fmha_policy_64x128x64>(args);
      } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
        return kernel_call<fmha_policy_64x128x128>(args);
      } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
        return kernel_call<fmha_policy_64x128x256>(args);
      } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
        return kernel_call<fmha_policy_64x128x512>(args);
      } else {
        assert(false);
        return {};
      }
    } else {
      std::cout << "unsupported block size " << std::endl;
      assert(false);
      return {};
    }
  }

  static cgfs_t run(const fmha::dispatch_fmha_forward_args_t<T>& args) {
#ifdef SDP_DBG
    printf("\n%s\n", __PRETTY_FUNCTION__);
#endif

    if (kIsTraining && kIsDropout && !args.dropout_mask) {
      if constexpr (kIsTraining && kIsDropout && arch_tag == gpu_arch::XeHpc) {
        if (args.head_size <= HEAD_SIZE_LIMIT_0) {
          return policy<fmha_policy_128x128x64>(args);
        } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
          return policy<fmha_policy_128x128x128>(args);
        } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
          return policy<fmha_policy_128x128x256>(args);
        } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
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
      constexpr bool igpu_wo_dropout =
          arch_tag == gpu_arch::XeLpg && !kIsDropout;
      constexpr bool v2_available = igpu_wo_dropout && !kUseAlibi &&
          !kIsCausal && !kIsTraining && !kIsDropout && !kIsVarlen && !kIsLocal;
      // roughly policy should match (num_queries)x(num_keys)x(head_size)
      if (args.head_size <= HEAD_SIZE_LIMIT_0) {
        if (v2_available && args.num_queries == NUM_QUERIES_GREEDY &&
            args.head_size == HEAD_SIZE_LIMIT_0) {
          // for extremely short query length
          if constexpr (v2_available) {
            assert(p.head_size == HEAD_SIZE_LIMIT_0);
            return policy<std::integral_constant<int, HEAD_SIZE_LIMIT_0>>(args);
          }
        } else if (args.num_queries < NUM_QUERIES_LARGE) {
          // for short query length
          return policy<fmha_policy_8x128x64>(args);
        } else {
          // for long query length
          return policy<fmha_policy_64x128x64>(args);
        }
      } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
        if (igpu_wo_dropout && args.num_queries == NUM_QUERIES_GREEDY) {
          // for extremely short query length
          if (args.num_keys >= NUM_KEYS_LIMIT_0 &&
              args.head_size == HEAD_SIZE_LIMIT_1 && v2_available) {
            if constexpr (v2_available) {
              return policy<std::integral_constant<int, HEAD_SIZE_LIMIT_1>>(
                  args);
            }
          } else if constexpr (igpu_wo_dropout) {
            return policy<fmha_policy_1x256x128>(args);
          }
          return {};
        } else if (args.num_queries < NUM_QUERIES_LARGE) {
          // for short query length
          if (args.num_keys < NUM_KEYS_LIMIT_1) {
            return policy<fmha_policy_8x256x128>(args);
          } else {
            return policy<fmha_policy_8x512x128>(args);
          }
        } else {
          // for long query length
          if constexpr (arch_tag == gpu_arch::XeLpg)
            return policy<fmha_policy_32x128x128>(args);
          else
            return policy<fmha_policy_64x128x128>(args);
        }
      } else if constexpr (arch_tag == gpu_arch::XeLpg) {
        std::cout << "Larger head_size are experiencing problems on MTL...\n";
        assert(false);
        return {};
      } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
        if (args.num_queries < NUM_QUERIES_LARGE) {
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
      } else if (
          arch_tag == gpu_arch::XeHpc && args.head_size <= HEAD_SIZE_LIMIT_3) {
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
        args.is_dropout,
        args.is_varlen,
        args.is_local);
  } else if constexpr (arch_tag != gpu_arch::XeLpg) {
    return dispatch_fmha_forward<bf16, arch_tag>(
        fmha::dispatch_fmha_forward_args_t<bf16>(args),
        args.alibi != nullptr, // is alibi
        args.attn_mask != nullptr, // is is_attn_mask
        args.is_causal,
        args.seq_last,
        args.is_training,
        args.is_dropout,
        args.is_varlen,
        args.is_local);
    return {};
  } else {
    printf("bfloat16 is not supported on the XeLpg platform\n");
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
      // TODO(Yi): fix XeHpg; update fmha_forward_configure.cmake when fixed
      return _fmha_forward_kernel<gpu_arch::XeLpg>(xeType, args);
#endif
#ifdef USE_XETLA_XE_HPC
    case gpu_arch::XeHpc:
      return _fmha_forward_kernel<gpu_arch::XeHpc>(xeType, args);
#endif
    default:
      TORCH_CHECK(false, "Unsupported gpu_arch of fmha_forward!!\n\n");
  }
}
} // namespace gpu::xetla
