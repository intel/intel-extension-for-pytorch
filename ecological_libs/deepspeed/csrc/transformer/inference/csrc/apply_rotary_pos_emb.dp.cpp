/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <dpct/dpct.h>
#include <sycl/sycl.hpp>
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "inference_sycl_layers.h"
#include "memory_access_utils.h"

namespace rot_half {
constexpr int threads = 256;
} // namespace rot_half

template <typename T, int threadsPerHead, int granularity>
class apply_rotary_pos_half {
 private:
  T* mixed_query;
  T* key_layer;
  unsigned rotary_dim;
  unsigned seq_len;
  unsigned seq_offset;
  unsigned num_heads;
  unsigned head_size;
  unsigned total_count;
  float rope_theta;
  int max_out_tokens;

 public:
  apply_rotary_pos_half(
      T* mixed_query,
      T* key_layer,
      unsigned rotary_dim,
      unsigned seq_len,
      unsigned seq_offset,
      unsigned num_heads,
      unsigned head_size,
      unsigned total_count,
      float rope_theta,
      int max_out_tokens)
      : mixed_query(mixed_query),
        rotary_dim(rotary_dim),
        seq_len(seq_len),
        seq_offset(seq_offset),
        num_heads(num_heads),
        head_size(head_size),
        total_count(total_count),
        rope_theta(rope_theta),
        max_out_tokens(max_out_tokens) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int T_per_thread = granularity / sizeof(T);
    constexpr int heads_per_block = rot_half::threads / threadsPerHead;

    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    auto head_group = sycl::ext::oneapi::experimental::this_sub_group();

    const int head_idx = item_ct1.get_group(2) * heads_per_block +
        item_ct1.get_local_id(2) / threadsPerHead;
    const int cur_seq_idx = head_idx % seq_len;
    const int offset = head_idx * head_size;
    const int k_offset =
        (cur_seq_idx + (head_idx / seq_len) * max_out_tokens) * head_size;

    const int seq_idx = cur_seq_idx + seq_offset;
    const int half_dim = rotary_dim >> 1;
    const int half_dim_threads = half_dim / T_per_thread;

    if (head_idx < total_count) {
      /*
      DPCT1007:0: Migration of thread_rank is not supported.
      */
      const int base_neuron_idx =
          head_group.get_local_linear_id() * T_per_thread;

      T q[T_per_thread], k[T_per_thread];
      mem_access::load_global<granularity>(
          q, mixed_query + offset + base_neuron_idx);
      mem_access::load_global<granularity>(
          k, key_layer + k_offset + base_neuron_idx);

#pragma unroll
      for (int i = 0; i < T_per_thread; i++) {
        const int neuron_idx = base_neuron_idx + i;
        if (neuron_idx < rotary_dim) {
          float inv_freq =
              (float)((neuron_idx % half_dim) * 2) / (float)rotary_dim;
          inv_freq = 1.0 / dpct::pow(rope_theta, inv_freq) * (float)seq_idx;

          float rotary_sign = (neuron_idx > (half_dim - 1) ? -1.0 : 1.0);
          float q_rot = conversion::to<float>(q[i]) * rotary_sign;
          float k_rot = conversion::to<float>(k[i]) * rotary_sign;

          const int target_lane = (neuron_idx < half_dim)
              /*
              DPCT1007:1: Migration of thread_rank is not supported.
              */
              ? head_group.get_local_linear_id() + half_dim_threads
              /*
              DPCT1007:2: Migration of thread_rank is not supported.
              */
              : head_group.get_local_linear_id() - half_dim_threads;

          /*
          DPCT1007:5: Migration of cooperative_groups::thread_block_tile::shfl
          is not supported.
          */
          const float q_rot_temp = head_group.shuffle(q_rot, target_lane);
          /*
          DPCT1007:6: Migration of cooperative_groups::thread_block_tile::shfl
          is not supported.
          */
          const float k_rot_temp = head_group.shuffle(k_rot, target_lane);

          q[i] = conversion::to<T>(
              conversion::to<float>(q[i]) * sycl::cos(inv_freq) +
              q_rot_temp * sycl::sin(inv_freq));
          k[i] = conversion::to<T>(
              conversion::to<float>(k[i]) * sycl::cos(inv_freq) +
              k_rot_temp * sycl::sin(inv_freq));
        }
      }

      mem_access::store_global<granularity>(
          mixed_query + offset + base_neuron_idx, q);
      mem_access::store_global<granularity>(
          key_layer + k_offset + base_neuron_idx, k);
    }
  }
};

#define LAUNCH_ROT_POS_EMB_HALF(HEAD_THREADS, ALIGNMENT)                 \
  {                                                                      \
    dpct::has_capability_or_fail(                                        \
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
    apply_rotary_pos_half<T, HEAD_THREADS, ALIGNMENT> fn(                \
        mixed_query,                                                     \
        key_layer,                                                       \
        rotary_dim,                                                      \
        seq_len,                                                         \
        offset,                                                          \
        num_heads,                                                       \
        head_size,                                                       \
        total_count,                                                     \
        rope_theta,                                                      \
        max_out_tokens);                                                 \
    stream->submit([&](sycl::handler& cgh) {                             \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);      \
    });                                                                  \
  }

#define LAUNCH_FOR_ALIGNMENT(ALIGNMENT)     \
  if (threads_per_head == 4) {              \
    LAUNCH_ROT_POS_EMB_HALF(4, ALIGNMENT);  \
  } else if (threads_per_head == 8) {       \
    LAUNCH_ROT_POS_EMB_HALF(8, ALIGNMENT);  \
  } else if (threads_per_head == 16) {      \
    LAUNCH_ROT_POS_EMB_HALF(16, ALIGNMENT); \
  } else if (threads_per_head == 32) {      \
    LAUNCH_ROT_POS_EMB_HALF(32, ALIGNMENT); \
  } else {                                  \
    assert(false);                          \
  }

template <typename T>
void launch_apply_rotary_pos_emb(
    T* mixed_query,
    T* key_layer,
    unsigned head_size,
    unsigned seq_len,
    unsigned rotary_dim,
    unsigned offset,
    unsigned num_heads,
    unsigned batch,
    float rope_theta,
    dpct::queue_ptr stream,
    int max_out_tokens) {
  const int half_dim = rotary_dim >> 1;

  int alignment = sizeof(T);
  if (half_dim % (16 / sizeof(T)) == 0) {
    alignment = 16;
  } else if (half_dim % (8 / sizeof(T)) == 0) {
    alignment = 8;
  } else if (half_dim % (4 / sizeof(T)) == 0) {
    alignment = 4;
  } else {
    assert(false);
  }
  const int T_per_elem = alignment / sizeof(T);

  int total_count = batch * num_heads * seq_len;

  const int padded_head_size = next_pow2(head_size);

  assert(padded_head_size <= hw_warp_size * T_per_elem);

  const int threads_per_head = padded_head_size / T_per_elem;
  const int heads_per_block = rot_half::threads / threads_per_head;

  sycl::range<3> block(1, 1, rot_half::threads);
  sycl::range<3> grid(
      1, 1, (total_count + heads_per_block - 1) / heads_per_block);

  if (alignment == 4) {
    LAUNCH_FOR_ALIGNMENT(4);
  } else if (alignment == 8) {
    LAUNCH_FOR_ALIGNMENT(8);
  } else if (alignment == 16) {
    LAUNCH_FOR_ALIGNMENT(16);
  } else {
    assert(false);
  }
}

#define INSTANTIATE_LAUNCH_ROTARY_POS_EMB(T)    \
  template void launch_apply_rotary_pos_emb<T>( \
      T*,                                       \
      T*,                                       \
      unsigned,                                 \
      unsigned,                                 \
      unsigned,                                 \
      unsigned,                                 \
      unsigned,                                 \
      unsigned,                                 \
      float,                                    \
      dpct::queue_ptr,                          \
      int);

INSTANTIATE_LAUNCH_ROTARY_POS_EMB(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(sycl::half);
