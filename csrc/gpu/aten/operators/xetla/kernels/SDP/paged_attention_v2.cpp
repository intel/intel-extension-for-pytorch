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
#include "paged_attention_kernel.hpp"
#include "paged_attention_policy.hpp"
#include "xetla.hpp"

namespace gpu::xetla {

namespace attention {

template <
    typename T,
    typename U,
    uint32_t HEAD_SIZE,
    uint32_t BLOCK_SIZE,
    gpu_arch arch_tag = gpu_arch::XeHpc>
std::vector<std::function<void(sycl::handler&)>> launch_paged_attention_v2(
    paged_attention_fwd_kernel_args_t fwd_args) {
  using policy = paged_attention_policy_v2<HEAD_SIZE, BLOCK_SIZE>;

  uint32_t max_num_partitions =
      (fwd_args.max_context_len + policy::partition_size - 1) /
      policy::partition_size;

  TORCH_CHECK(
      max_num_partitions > 1,
      "max_context_len must be greater than partition_size when using paged attention v2");

  std::function<void(sycl::handler&)> cgh0 = [=](sycl::handler& cgh) {
    // first kernel
    using kernel = paged_attention_kernel<policy, T, U, arch_tag>;

    sycl::nd_range<3> nd_range = kernel::get_nd_range(
        fwd_args.num_seqs, fwd_args.num_heads, max_num_partitions);

    cgh.parallel_for<paged_attention_kernel<policy, T, U, arch_tag>>(
        nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
          kernel kernel_fn;
          typename kernel::arguments_t args(
              fwd_args.max_logits,
              fwd_args.exp_sums,
              reinterpret_cast<T*>(fwd_args.tmp_out),
              reinterpret_cast<T*>(fwd_args.query),
              reinterpret_cast<T*>(fwd_args.key_cache),
              reinterpret_cast<T*>(fwd_args.value_cache),
              reinterpret_cast<U*>(fwd_args.head_mapping),
              reinterpret_cast<U*>(fwd_args.block_tables),
              reinterpret_cast<U*>(fwd_args.context_lens),
              fwd_args.sm_scale,
              fwd_args.num_seqs,
              fwd_args.num_heads,
              fwd_args.num_kv_heads,
              fwd_args.head_size,
              fwd_args.max_blocks_per_seq);

          kernel_fn(item, args);
        });
  };
  std::function<void(sycl::handler&)> cgh1 = [=](sycl::handler& cgh) {
    // second reduce kernel
    using reduce_kernel = paged_attention_reduce<policy, T, U, arch_tag>;
    sycl::nd_range<3> nd_range =
        reduce_kernel::get_nd_range(fwd_args.num_seqs, fwd_args.num_heads);

    cgh.parallel_for<paged_attention_reduce<policy, T, U, arch_tag>>(
        nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
          reduce_kernel reduce_kernel_fn;
          typename reduce_kernel::arguments_t args(
              reinterpret_cast<T*>(fwd_args.out),
              reinterpret_cast<T*>(fwd_args.tmp_out),
              fwd_args.max_logits,
              fwd_args.exp_sums,
              reinterpret_cast<U*>(fwd_args.context_lens),
              fwd_args.num_seqs,
              fwd_args.num_heads,
              fwd_args.head_size,
              max_num_partitions);

          reduce_kernel_fn(item, args);
        });
  };
  return {cgh0, cgh1};
}
} // namespace attention

#define CALL_V2_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE)                          \
  attention::launch_paged_attention_v2<T, U, HEAD_SIZE, BLOCK_SIZE, arch_tag>( \
      args);

// Note: only support block_size = 16/32, to reduce compiliation time
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE) \
  switch (args.block_size) {                         \
    case 16: {                                       \
      return CALL_V2_LAUNCHER(T, U, HEAD_SIZE, 16);  \
    }                                                \
    case 32: {                                       \
      return CALL_V2_LAUNCHER(T, U, HEAD_SIZE, 32);  \
    }                                                \
    default: {                                       \
      TORCH_CHECK(0, "Unsupported block size: ");    \
      return {};                                     \
    }                                                \
  }

template <gpu::xetla::gpu_arch arch_tag, typename T>
cgfs_t paged_attention_v2_dispatch(
    gpu::xetla::paged_attention_fwd_kernel_args_t args) {
  using U = int32_t;
  if (args.head_size <= 64) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, 64);
  } else if (args.head_size <= 128) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, 128);
  } else if (args.head_size <= 256) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, 256);
  } else {
    TORCH_CHECK(0, "Unsupported head size");
    return {};
  }
}

template <gpu_arch arch_tag>
cgfs_t _paged_attention_v2(
    XetlaType xeType,
    paged_attention_fwd_kernel_args_t args) {
  if (xeType == gpu::xetla::XetlaType::fp16) {
    return paged_attention_v2_dispatch<arch_tag, fp16>(args);
  } else if (xeType == XetlaType::bf16) {
    // 2024.1 and below do not support bf16's operator< etc
    if constexpr (
        __INTEL_LLVM_COMPILER >= 20240200 && arch_tag != gpu_arch::XeLpg) {
      return paged_attention_v2_dispatch<arch_tag, bf16>(args);
    } else {
      printf("paged_attention: No bf16 support for current arch!!\n\n");
      return {};
    }
  } else {
    printf("paged_attention: Unsupported dtype!\n\n");
    return {};
  }
}

XETLA_KERNEL_API cgfs_t paged_attention_v2(
    gpu_arch arch_tag,
    XetlaType xeType,
    paged_attention_fwd_kernel_args_t args) {
  switch (arch_tag) {
#if __INTEL_LLVM_COMPILER >= 20240200
#ifdef USE_XETLA_XE_LPG
    case gpu::xetla::gpu_arch::XeLpg:
      return _paged_attention_v2<gpu::xetla::gpu_arch::XeLpg>(xeType, args);
#endif
#ifdef USE_XETLA_XE_HPG
    case gpu::xetla::gpu_arch::XeHpg:
      return _paged_attention_v2<gpu::xetla::gpu_arch::XeHpg>(xeType, args);
#endif
#endif // __INTEL_LLVM_COMPILER >= 20240200
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return _paged_attention_v2<gpu::xetla::gpu_arch::XeHpc>(xeType, args);
#endif
    default:
      printf("Unsupported gpu_arch of paged_attention_v2!!\n\n");
      return {};
  }
}

#undef CALL_V2_LAUNCHER_BLOCK_SIZE
#undef CALL_V2_LAUNCHER

} // namespace gpu::xetla
