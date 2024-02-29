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

template <typename Policy, typename T, typename U>
void launch_paged_attention_v1(
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
  using kernel = paged_attention_kernel<Policy, T, U>;

  constexpr uint32_t max_comutation =
      Policy::block_size * Policy::max_blocks_per_sg * Policy::wg_size;
  TORCH_CHECK(
      max_context_len <= max_comutation,
      "max_context_len is too large to compute for paged attention V1");

  sycl::nd_range<3> nd_range = kernel::get_nd_range(num_seqs, num_heads, 1);
  auto cgh = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<paged_attention_kernel<Policy, T, U>>(
        nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          kernel kernel_fn;
          typename kernel::arguments_t args(
              /* max_logits */ nullptr,
              /* exp_sums */ nullptr,
              out,
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
} // namespace attention

#define CALL_V1_LAUNCHER(                                   \
    T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, BLOCK_NUM_PER_SG) \
  return attention::launch_paged_attention_v1<              \
      paged_attention_policy_v1<                            \
          HEAD_SIZE,                                        \
          BLOCK_SIZE,                                       \
          WG_SIZE,                                          \
          BLOCK_NUM_PER_SG>,                                \
      T,                                                    \
      U>(                                                   \
      q,                                                    \
      out,                                                  \
      query,                                                \
      key_cache,                                            \
      value_cache,                                          \
      head_mapping,                                         \
      block_tables,                                         \
      context_lens,                                         \
      sm_scale,                                             \
      num_seqs,                                             \
      num_heads,                                            \
      num_kv_heads,                                         \
      head_size,                                            \
      max_blocks_per_seq,                                   \
      max_context_len);

#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE)      \
  int min_blocks_per_sg =                                                      \
      (max_context_len + (WG_SIZE * BLOCK_SIZE) - 1) / (WG_SIZE * BLOCK_SIZE); \
  if (min_blocks_per_sg <= 2)                                                  \
    CALL_V1_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 2)                  \
  else if (min_blocks_per_sg <= 4)                                             \
    CALL_V1_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 4)                  \
  else if (min_blocks_per_sg <= 8)                                             \
    CALL_V1_LAUNCHER(T, U, HEAD_SIZE, BLOCK_SIZE, WG_SIZE, 8)                  \
  else {                                                                       \
    TORCH_CHECK(0, "max_context_len is too large to compute");                 \
  }

#define CALL_V1_LAUNCHER_WG_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE)       \
  int min_num_wg = (max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE; \
  if (min_num_wg <= 16) {                                           \
    CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, 8)     \
  } else if (min_num_wg <= 32) {                                    \
    CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, 16)    \
  } else {                                                          \
    CALL_V1_LAUNCHER_BLOCK_SIZE(T, U, HEAD_SIZE, BLOCK_SIZE, 32)    \
  }

// Note: only support block_size = 16/32, to reduce compiliation time
#define CALL_V1_LAUNCHER_HEAD_SIZE(T, U, HEAD_SIZE)  \
  switch (block_size) {                              \
    case 16: {                                       \
      CALL_V1_LAUNCHER_WG_SIZE(T, U, HEAD_SIZE, 16); \
    }                                                \
    case 32: {                                       \
      CALL_V1_LAUNCHER_WG_SIZE(T, U, HEAD_SIZE, 32); \
    }                                                \
    default: {                                       \
      TORCH_CHECK(0, "Unsupported block size: ");    \
    }                                                \
  }

void paged_attention_v1(
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
    CALL_V1_LAUNCHER_HEAD_SIZE(T, U, 64);
  } else if (head_size <= 128) {
    CALL_V1_LAUNCHER_HEAD_SIZE(T, U, 128);
  } else if (head_size <= 256) {
    CALL_V1_LAUNCHER_HEAD_SIZE(T, U, 256);
  } else {
    TORCH_CHECK(0, "Unsupported head size");
  }
}

#undef CALL_V1_LAUNCHER_BLOCK_SIZE
#undef CALL_V1_LAUNCHER

} // namespace gpu::xetla