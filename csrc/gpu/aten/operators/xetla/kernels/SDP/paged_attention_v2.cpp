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

#include <utils/DPCPP.h>
#include <limits>
#include "paged_attention_kernel.hpp"
#include "paged_attention_policy.hpp"

#include "xetla.hpp"

namespace gpu::xetla {

namespace attention {

template <typename T, typename U, uint32_t HEAD_SIZE, uint32_t BLOCK_SIZE>
void launch_paged_attention_v2(
    float* max_logits,
    float* exp_sums,
    T* tmp_out,
    sycl::queue& q,
    T* out,
    T* query,
    T* key_cache,
    T* value_cache,
    U* head_mapping,
    U* block_tables,
    U* context_lens,
    float sm_scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t max_blocks_per_seq,
    uint32_t max_context_len) {
  using policy = paged_attention_policy_v2<HEAD_SIZE, BLOCK_SIZE>;

  uint32_t max_num_partitions =
      (max_context_len + policy::partition_size - 1) / policy::partition_size;

  TORCH_CHECK(
      max_num_partitions > 1,
      "max_context_len must be greater than partition_size when using paged attention v2");

  {
    // first kernel
    using kernel = paged_attention_kernel<policy, T, U>;

    sycl::nd_range<3> nd_range =
        kernel::get_nd_range(num_seqs, num_heads, max_num_partitions);

    auto cgh = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for<paged_attention_kernel<policy, T, U>>(
          nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
            kernel kernel_fn;
            typename kernel::arguments_t args(
                max_logits,
                exp_sums,
                tmp_out,
                query,
                key_cache,
                value_cache,
                head_mapping,
                block_tables,
                context_lens,
                sm_scale,
                num_seqs,
                num_heads,
                num_kv_heads,
                head_size,
                max_blocks_per_seq);

            kernel_fn(item, args);
          });
    };
    DPCPP_Q_SUBMIT(q, cgh);
  }
  {
    // second reduce kernel
    using reduce_kernel = paged_attention_reduce<policy, T, U>;
    sycl::nd_range<3> nd_range =
        reduce_kernel::get_nd_range(num_seqs, num_heads);

    auto cgh2 = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for<paged_attention_reduce<policy, T, U>>(
          nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
            reduce_kernel reduce_kernel_fn;
            typename reduce_kernel::arguments_t args(
                out,
                tmp_out,
                max_logits,
                exp_sums,
                context_lens,
                num_seqs,
                num_heads,
                head_size,
                max_num_partitions);

            reduce_kernel_fn(item, args);
          });
    };
    DPCPP_Q_SUBMIT(q, cgh2);
  }
}
} // namespace attention

#define CALL_V2_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE)                       \
  return attention::launch_paged_attention_v2<T, U, HEAD_SIZE, BLOCK_SIZE>( \
      max_logits,                                                           \
      exp_sums,                                                             \
      tmp_out,                                                              \
      q,                                                                    \
      out,                                                                  \
      query,                                                                \
      key_cache,                                                            \
      value_cache,                                                          \
      head_mapping,                                                         \
      block_tables,                                                         \
      context_lens,                                                         \
      sm_scale,                                                             \
      num_seqs,                                                             \
      num_heads,                                                            \
      num_kv_heads,                                                         \
      head_size,                                                            \
      max_blocks_per_seq,                                                   \
      max_context_len);

// Note: only support block_size = 16/32, to reduce compiliation time
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE) \
  switch (block_size) {                              \
    case 16: {                                       \
      CALL_V2_LAUNCHER(T, U, HEAD_SIZE, 16);         \
    }                                                \
    case 32: {                                       \
      CALL_V2_LAUNCHER(T, U, HEAD_SIZE, 32);         \
    }                                                \
    default: {                                       \
      TORCH_CHECK(0, "Unsupported block size: ");    \
    }                                                \
  }

void paged_attention_v2(
    float* max_logits,
    float* exp_sums,
    sycl::half* tmp_out,
    sycl::queue& q,
    sycl::half* out,
    sycl::half* query,
    sycl::half* key_cache,
    sycl::half* value_cache,
    int32_t* head_mapping,
    int32_t* block_tables,
    int32_t* context_lens,
    float sm_scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t block_size,
    uint32_t max_blocks_per_seq,
    uint32_t max_context_len) {
  using T = sycl::half;
  using U = int32_t;
  if (head_size <= 64) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, 64);
  } else if (head_size <= 128) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, 128);
  } else if (head_size <= 256) {
    CALL_V2_LAUNCHER_BLOCK_SIZE(T, U, 256);
  } else {
    TORCH_CHECK(0, "Unsupported head size");
  }
}

#undef CALL_V2_LAUNCHER_BLOCK_SIZE
#undef CALL_V2_LAUNCHER

} // namespace gpu::xetla
