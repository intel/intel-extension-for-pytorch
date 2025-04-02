/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in wriscalar_tg, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <c10/util/Exception.h>
#include <limits>
#include "../../mha.h"
#include "../xetla_kernel_api.h"
#include "blocked_attention_kernel.hpp"
#include "blocked_attention_policy.hpp"
#include "common/core/common_types.hpp"
#include "paged_attention_kernel.hpp"
#include "paged_attention_policy.hpp"
#include "xetla.hpp"

namespace gpu::xetla {

namespace attention {

// {
template <
    uint32_t max_head_size,
    uint32_t block_size,
    uint32_t wg_size,
    uint32_t max_blocks_per_sg,
    typename arg_type>
struct slice_kv_policy {};

template <
    uint32_t max_head_size,
    uint32_t block_size,
    uint32_t wg_size,
    uint32_t max_blocks_per_sg>
struct slice_kv_policy<
    max_head_size,
    block_size,
    wg_size,
    max_blocks_per_sg,
    paged_attention_fwd_kernel_args_t> {
  using policy = paged_attention_policy_v1<
      max_head_size,
      block_size,
      wg_size,
      max_blocks_per_sg>;
};

template <
    uint32_t max_head_size,
    uint32_t block_size,
    uint32_t wg_size,
    uint32_t max_blocks_per_sg>
struct slice_kv_policy<
    max_head_size,
    block_size,
    wg_size,
    max_blocks_per_sg,
    chunked_prefill_fwd_kernel_args_t> {
  using policy = chunked_prefill_slice_kv_policy<
      max_head_size,
      block_size,
      wg_size,
      max_blocks_per_sg>;
};
// } // namespace attention

template <typename Policy, typename T, typename U, gpu_arch arch_tag>
cgfs_t launch_kernels(paged_attention_fwd_kernel_args_t fwd_args) {
  using kernel = paged_attention_kernel<Policy, T, U, arch_tag>;

  constexpr uint32_t max_computation =
      Policy::block_size * Policy::max_blocks_per_sg * Policy::wg_size;
  TORCH_CHECK(
      fwd_args.max_context_len <= max_computation,
      "max_context_len is too large to compute for paged attention V1");

  sycl::nd_range<3> nd_range =
      kernel::get_nd_range(fwd_args.num_seqs, fwd_args.num_heads, 1);
  return {[=](sycl::handler& cgh) {
    cgh.parallel_for<paged_attention_kernel<Policy, T, U, arch_tag>>(
        nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
          kernel kernel_fn;
          typename kernel::arguments_t args(
              nullptr,
              nullptr,
              reinterpret_cast<T*>(fwd_args.out),
              reinterpret_cast<T*>(fwd_args.query),
              reinterpret_cast<T*>(fwd_args.key_cache),
              reinterpret_cast<T*>(fwd_args.value_cache),
              reinterpret_cast<float*>(fwd_args.alibi_slopes),
              reinterpret_cast<U*>(fwd_args.block_tables),
              reinterpret_cast<U*>(fwd_args.context_lens),
              fwd_args.num_queries_per_tokens,
              fwd_args.sm_scale,
              fwd_args.num_seqs,
              fwd_args.num_heads,
              fwd_args.num_kv_heads,
              fwd_args.head_size,
              fwd_args.max_blocks_per_seq,
              fwd_args.softcap);

          kernel_fn(item, args);
        });
  }};
}

template <typename Policy, typename T, typename U, gpu_arch arch_tag>
cgfs_t launch_kernels(chunked_prefill_fwd_kernel_args_t fwd_args) {
  using kernel = blocked_attention_kernel<Policy, T, arch_tag>;

  int32_t query_groups =
      (fwd_args.max_queries + Policy::block_m - 1) / Policy::block_m;
  sycl::nd_range<3> nd_range = kernel::get_nd_range(
      fwd_args.batch_size, fwd_args.num_heads_q, query_groups, 1);
  return {[=](sycl::handler& cgh) {
    cgh.parallel_for<blocked_attention_kernel<Policy, T, arch_tag>>(
        nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
          kernel kernel_fn;
          typename kernel::arguments_t args(
              reinterpret_cast<T*>(fwd_args.out),
              nullptr,
              nullptr,
              reinterpret_cast<T*>(fwd_args.query),
              reinterpret_cast<T*>(fwd_args.key_cache),
              reinterpret_cast<T*>(fwd_args.value_cache),
              fwd_args.block_table,
              fwd_args.cu_seqlen_q,
              fwd_args.cu_seqlen_k,
              fwd_args.sm_scale,
              fwd_args.num_heads_q * fwd_args.head_size,
              fwd_args.num_heads_k * fwd_args.head_size,
              fwd_args.max_queries,
              fwd_args.max_keys,
              fwd_args.batch_size,
              fwd_args.num_heads_q,
              fwd_args.num_heads_k,
              fwd_args.head_size,
              fwd_args.max_blocks_per_seq,
              1,
              fwd_args.window_size_left,
              fwd_args.window_size_right,
              fwd_args.is_causal,
              fwd_args.is_local,
              fwd_args.softcap);

          kernel_fn(item, args);
        });
  }};
}
} // namespace attention

#define CALL_V1_LAUNCHER(                                   \
    T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, BLOCK_NUM_PER_SG) \
  return attention::launch_kernels<                         \
      typename attention::slice_kv_policy<                  \
          HEAD_SIZE,                                        \
          BLOCK_SIZE,                                       \
          WG_SIZE,                                          \
          BLOCK_NUM_PER_SG,                                 \
          decltype(args)>::policy,                          \
      T,                                                    \
      U,                                                    \
      arch_tag>(args);

#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE) \
  int min_blocks_per_sg =                                                 \
      (args.max_context_len + (WG_SIZE * BLOCK_SIZE) - 1) /               \
      (WG_SIZE * BLOCK_SIZE);                                             \
  if (min_blocks_per_sg <= 2)                                             \
    CALL_V1_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 2)             \
  else if (min_blocks_per_sg <= 4)                                        \
    CALL_V1_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 4)             \
  else if (min_blocks_per_sg <= 8)                                        \
    CALL_V1_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 8)             \
  else {                                                                  \
    TORCH_CHECK(0, "max_context_len is too large to compute");            \
    return {};                                                            \
  }

#define CALL_V1_LAUNCHER_WG_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE)            \
  int min_num_wg = (args.max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE; \
  if (min_num_wg <= 16) {                                                \
    CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, 8)          \
  } else if (min_num_wg <= 32) {                                         \
    CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, 16)         \
  } else {                                                               \
    CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, 32)         \
  }

// Note: only support block_size = 16/32, to reduce compiliation time
#define CALL_V1_LAUNCHER_HEAD_SIZE(T, U, HEAD_SIZE)  \
  switch (args.block_size) {                         \
    case 16: {                                       \
      CALL_V1_LAUNCHER_WG_SIZE(T, U, HEAD_SIZE, 16); \
    }                                                \
    case 32: {                                       \
      CALL_V1_LAUNCHER_WG_SIZE(T, U, HEAD_SIZE, 32); \
    }                                                \
    default: {                                       \
      TORCH_CHECK(0, "Unsupported block size: ");    \
      return {};                                     \
    }                                                \
  }

template <gpu_arch arch_tag, typename T, typename arg_type>
cgfs_t slice_kv_dispatch(arg_type args) {
  using U = int32_t;
  if (args.head_size <= 64) {
    CALL_V1_LAUNCHER_HEAD_SIZE(T, U, 64);
  } else if (args.head_size <= 128) {
    CALL_V1_LAUNCHER_HEAD_SIZE(T, U, 128);
  } else if (args.head_size <= 256) {
    CALL_V1_LAUNCHER_HEAD_SIZE(T, U, 256);
  } else {
    TORCH_CHECK(0, "Unsupported head size");
    return {};
  }
}

template <gpu_arch arch_tag>
cgfs_t _paged_attention_v1(
    XetlaType xeType,
    paged_attention_fwd_kernel_args_t args) {
  if (xeType == XetlaType::fp16) {
    return slice_kv_dispatch<arch_tag, fp16>(args);
  } else if (xeType == XetlaType::bf16) {
    // 2024.1 and below do not support bf16's operator< etc
    if constexpr (
        __INTEL_LLVM_COMPILER >= 20240200 && arch_tag != gpu_arch::XeLpg) {
      return slice_kv_dispatch<arch_tag, bf16>(args);
    } else {
      printf("paged_attention: No bf16 support for current arch!!\n\n");
      return {};
    }
  } else {
    printf("paged_attention: Unsupported dtype!\n\n");
    return {};
  }
}

XETLA_KERNEL_API cgfs_t paged_attention_v1(
    gpu_arch arch_tag,
    XetlaType xeType,
    paged_attention_fwd_kernel_args_t args) {
  switch (arch_tag) {
#if __INTEL_LLVM_COMPILER >= 20240200
#ifdef USE_XETLA_XE_LPG
    case gpu_arch::XeLpg:
      return _paged_attention_v1<gpu_arch::XeLpg>(xeType, args);
#endif
#ifdef USE_XETLA_XE_HPG
    case gpu_arch::XeHpg:
      return _paged_attention_v1<gpu_arch::XeHpg>(xeType, args);
#endif
#endif // __INTEL_LLVM_COMPILER >= 20240200
#ifdef USE_XETLA_XE_HPC
    case gpu_arch::XeHpc:
      return _paged_attention_v1<gpu_arch::XeHpc>(xeType, args);
#endif
    default:
      printf("Unsupported gpu_arch of paged_attention_v1!!\n\n");
      return {};
  }
}

#define CALL_LAUNCHER(T, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, BLOCK_NUM_PER_SG) \
  return attention::launch_blocked_attention<                              \
      blocked_attention_policy<                                            \
          HEAD_SIZE,                                                       \
          BLOCK_SIZE,                                                      \
          WG_SIZE,                                                         \
          BLOCK_NUM_PER_SG>,                                               \
      T>(                                                                  \
      out,                                                                 \
      query,                                                               \
      key_cache,                                                           \
      value_cache,                                                         \
      atoms,                                                               \
      block_tables,                                                        \
      sm_scale,                                                            \
      q_row_stride,                                                        \
      kv_row_stride,                                                       \
      num_atoms,                                                           \
      num_heads,                                                           \
      num_kv_heads,                                                        \
      head_size,                                                           \
      max_blocks_per_seq);

#define CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, BLOCK_SIZE, WG_SIZE)     \
  int min_blocks_per_sg = (max_blocks_per_seq + WG_SIZE - 1) / WG_SIZE; \
  if (min_blocks_per_sg <= 1)                                           \
    CALL_LAUNCHER(T, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 1)                 \
  else if (min_blocks_per_sg <= 2)                                      \
    CALL_LAUNCHER(T, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 2)                 \
  else {                                                                \
    throw std::runtime_error("Unsupported block number per subgroup");  \
  }

#define CALL_LAUNCHER_WG_SIZE(T, HEAD_SIZE, BLOCK_SIZE)    \
  if (max_blocks_per_seq <= 4) {                           \
    CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, BLOCK_SIZE, 2)  \
  } else if (max_blocks_per_seq <= 8) {                    \
    CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, BLOCK_SIZE, 4)  \
  } else if (max_blocks_per_seq <= 16) {                   \
    CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, BLOCK_SIZE, 8)  \
  } else if (max_blocks_per_seq <= 32) {                   \
    CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, BLOCK_SIZE, 16) \
  } else {                                                 \
    CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, BLOCK_SIZE, 32) \
  }

#define CALL_CHUNK_PREFILL_LAUNCHER(T, HEAD_SIZE, BLOCK_SIZE) \
  attention::                                                 \
      launch_chunked_prefill_kernel<T, HEAD_SIZE, BLOCK_SIZE, arch_tag>(args);

#define CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE)              \
  switch (block_size) {                                     \
    case 64: {                                              \
      CALL_CHUNK_PREFILL_LAUNCHER(T, HEAD_SIZE, 64);        \
    }                                                       \
    case 32: {                                              \
      CALL_CHUNK_PREFILL_LAUNCHER(T, HEAD_SIZE, 32);        \
    }                                                       \
    default: {                                              \
      throw std::runtime_error("Unsupported block size: "); \
    }                                                       \
  }

// Note: only support block_size = 128/64, to reduce compiliation time
#define CALL_LAUNCHER_HEAD_SIZE(T, HEAD_SIZE)               \
  switch (block_size) {                                     \
    case 64: {                                              \
      CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, 64);           \
    }                                                       \
    case 32: {                                              \
      CALL_LAUNCHER_BLOCK_SIZE(T, HEAD_SIZE, 32);           \
    }                                                       \
    default: {                                              \
      throw std::runtime_error("Unsupported block size: "); \
    }                                                       \
  }

template <gpu_arch arch_tag>
cgfs_t _chunked_prefill_slice_kv(
    XetlaType xeType,
    chunked_prefill_fwd_kernel_args_t args) {
  // Only support fp16 for now
  if (xeType == XetlaType::fp16) {
    return slice_kv_dispatch<arch_tag, fp16>(args);
  } else {
    printf("chunked_prefill: Unsupported dtype!\n\n");
    return {};
  }
}

XETLA_KERNEL_API cgfs_t chunked_prefill_slice_kv(
    gpu_arch arch_tag,
    XetlaType xeType,
    chunked_prefill_fwd_kernel_args_t args) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return _chunked_prefill_slice_kv<gpu::xetla::gpu_arch::XeHpc>(
          xeType, args);
#endif
    default:
      printf("Unsupported gpu_arch of chunked_prefill_slice_kv!!\n\n");
      return {};
  }
}

#undef CALL_V1_LAUNCHER_BLOCK_SIZE
#undef CALL_V1_LAUNCHER

} // namespace gpu::xetla
