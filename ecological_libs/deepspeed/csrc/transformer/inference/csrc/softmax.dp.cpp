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
#include <limits>
#include "conversion_utils.h"
#include "inference_sycl_layers.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define MAX_REG_SIZE 8

#define minus_infinity -10000.0

void CheckCudaErrorAux(const char* file, unsigned line) {
  /*
  DPCT1010:11: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 err = 0;
  if (err == 0)
    return;
  /*
  DPCT1009:12: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  std::cerr << "syclGetErrorString is not supported" /*syclGetErrorString(err)*/
            << "(" << err << ") at " << file << ":" << line << std::endl;
  throw std::runtime_error("SYCL ERROR!!!\n");
}

#define SYCL_CHECK_ERROR() CheckCudaErrorAux(__FILE__, __LINE__)

template <typename T, int iterations>
class attn_softmax_v2 {
 private:
  mutable T* vals;
  T* mask;
  T* alibi;
  float layer_scale;
  bool triangular;
  bool recompute;
  bool local_attention;
  int window_size;
  int total_count;
  int heads;
  int sequence_length;
  int num_seq;
  int head_offset;
  int mask_stride;
  int mp_size;
  int reduceWidth;

 public:
  attn_softmax_v2(
      T* vals,
      T* mask,
      T* alibi,
      float layer_scale,
      bool triangular,
      bool recompute,
      bool local_attention,
      int window_size,
      int total_count,
      int heads,
      int sequence_length,
      int num_seq,
      int head_offset,
      int mask_stride,
      int mp_size,
      int reduceWidth)
      : vals(vals),
        mask(mask),
        alibi(alibi),
        layer_scale(layer_scale),
        triangular(triangular),
        recompute(recompute),
        local_attention(local_attention),
        window_size(window_size),
        total_count(total_count),
        heads(heads),
        sequence_length(sequence_length),
        num_seq(num_seq),
        head_offset(head_offset),
        mask_stride(mask_stride),
        mp_size(mp_size),
        reduceWidth(reduceWidth) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    sycl::float2 low_data[MAX_REG_SIZE];
    sycl::float2 high_data[MAX_REG_SIZE];
    const T zero_h = conversion::to<T>(0.f);

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = item_ct1.get_local_id(2) % reduceWidth;

    auto& partialSum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[MAX_WARP_NUM]>(sycl::ext::oneapi::experimental::this_group<3>());

    int iter_offset = item_ct1.get_group(2) * (warp_num / reduce_blocks) +
        (wid / reduce_blocks);
    int batch_idx = iter_offset / (num_seq * heads);
    int alibi_offset = batch_idx * heads * mp_size + head_offset;
    int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);

    if (iter_offset < total_count) {
      vals += (iter_offset * sequence_length);

      alibi_offset =
          (alibi_offset + ((iter_offset / num_seq) % heads)) * sequence_length;
      mask_offset = mask_offset * sequence_length;
      int seq_id = iter_offset % num_seq;

      int real_seq_id =
          seq_id + (num_seq == sequence_length ? 0 : sequence_length);
      int window_stride4 =
          (local_attention && (real_seq_id >> 2) > (window_size >> 2))
          ? (real_seq_id >> 2) - (window_size >> 2)
          : 0;
      int window_stride = (local_attention && real_seq_id >= window_size)
          ? real_seq_id - window_size
          : -1;

      float max_val = minus_infinity;
      // if (lane == 0) printf("%d, %d: %d \n", wid, blockIdx.x, mask_offset);
      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane);
        bool check = (data_id >> 2) >= window_stride4;
        bool low_x_check = check && (data_id < sequence_length) &&
            (!triangular || (data_id <= seq_id)) && (data_id > window_stride);
        bool low_y_check = check &&
            ((data_id + reduceWidth) < sequence_length) &&
            (!triangular || ((data_id + reduceWidth) <= seq_id)) &&
            ((data_id + reduceWidth) > window_stride);
        bool high_x_check = check &&
            ((data_id + reduceWidth * 2) < sequence_length) &&
            (!triangular || ((data_id + reduceWidth * 2) <= seq_id)) &&
            ((data_id + reduceWidth * 2) > window_stride);
        bool high_y_check = check &&
            ((data_id + reduceWidth * 3) < sequence_length) &&
            (!triangular || ((data_id + reduceWidth * 3) <= seq_id)) &&
            ((data_id + reduceWidth * 3) > window_stride);

        if (mask && alibi) {
          low_data[i].x() = low_x_check
              ? conversion::to<float>(vals[data_id]) * layer_scale +
                  (conversion::to<float>(alibi[data_id + alibi_offset])) +
                  (conversion::to<float>(mask[data_id + mask_offset]))
              : minus_infinity;
          low_data[i].y() = low_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth]) *
                      layer_scale +
                  (conversion::to<float>(
                      alibi[data_id + alibi_offset + reduceWidth])) +
                  (conversion::to<float>(
                      mask[data_id + mask_offset + reduceWidth]))
              : minus_infinity;
          high_data[i].x() = high_x_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 2]) *
                      layer_scale +
                  (conversion::to<float>(
                      alibi[data_id + alibi_offset + reduceWidth * 2])) +
                  (conversion::to<float>(
                      mask[data_id + mask_offset + reduceWidth * 2]))
              : minus_infinity;
          high_data[i].y() = high_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 3]) *
                      layer_scale +
                  (conversion::to<float>(
                      alibi[data_id + alibi_offset + reduceWidth * 3])) +
                  (conversion::to<float>(
                      mask[data_id + mask_offset + reduceWidth * 3]))
              : minus_infinity;
        } else if (mask) {
          low_data[i].x() = low_x_check
              ? conversion::to<float>(vals[data_id]) * layer_scale +
                  (conversion::to<float>(mask[data_id + mask_offset]))
              : minus_infinity;
          low_data[i].y() = low_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth]) *
                      layer_scale +
                  (conversion::to<float>(
                      mask[data_id + mask_offset + reduceWidth]))
              : minus_infinity;
          high_data[i].x() = high_x_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 2]) *
                      layer_scale +
                  (conversion::to<float>(
                      mask[data_id + mask_offset + reduceWidth * 2]))
              : minus_infinity;
          high_data[i].y() = high_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 3]) *
                      layer_scale +
                  (conversion::to<float>(
                      mask[data_id + mask_offset + reduceWidth * 3]))
              : minus_infinity;
        } else if (alibi) {
          low_data[i].x() = low_x_check
              ? conversion::to<float>(vals[data_id]) * layer_scale +
                  (conversion::to<float>(alibi[data_id + alibi_offset]))
              : minus_infinity;
          low_data[i].y() = low_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth]) *
                      layer_scale +
                  (conversion::to<float>(
                      alibi[data_id + alibi_offset + reduceWidth]))
              : minus_infinity;
          high_data[i].x() = high_x_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 2]) *
                      layer_scale +
                  (conversion::to<float>(
                      alibi[data_id + alibi_offset + reduceWidth * 2]))
              : minus_infinity;
          high_data[i].y() = high_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 3]) *
                      layer_scale +
                  (conversion::to<float>(
                      alibi[data_id + alibi_offset + reduceWidth * 3]))
              : minus_infinity;
        } else {
          low_data[i].x() = low_x_check
              ? conversion::to<float>(vals[data_id]) * layer_scale
              : minus_infinity;
          low_data[i].y() = low_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale
              : minus_infinity;
          high_data[i].x() = high_x_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 2]) *
                  layer_scale
              : minus_infinity;
          high_data[i].y() = high_y_check
              ? conversion::to<float>(vals[data_id + reduceWidth * 3]) *
                  layer_scale
              : minus_infinity;
        }

        // if(lane == 0) printf("%f , %d, %d \n", low_data[i].x, data_id,
        // seq_id);
        max_val = (low_data[i].x() > max_val ? low_data[i].x() : max_val);
        max_val = (low_data[i].y() > max_val ? low_data[i].y() : max_val);
        max_val = (high_data[i].x() > max_val ? high_data[i].x() : max_val);
        max_val = (high_data[i].y() > max_val ? high_data[i].y() : max_val);
      }

      for (int i = 1; i < WARP_SIZE; i *= 2) {
        auto temp = sycl::permute_group_by_xor(
            sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
        max_val = (temp > max_val ? temp : max_val);
      }

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = max_val;
        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane < warp_num)
          max_val = partialSum[lane];

        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int i = 1; i < reduce_blocks; i *= 2) {
          auto temp = sycl::permute_group_by_xor(
              sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
          max_val = (temp > max_val ? temp : max_val);
        }

        /*
        DPCT1007:13: Migration of cooperative_groups::thread_block_tile::shfl is
        not supported.
        */
        max_val = g.shuffle(max_val, item_ct1.get_local_id(2) / WARP_SIZE);
      }
      float sum = 0;
      for (int i = 0; i < iterations; i++) {
        low_data[i].x() = sycl::native::exp(low_data[i].x() - max_val);
        low_data[i].y() = sycl::native::exp(low_data[i].y() - max_val);
        high_data[i].x() = sycl::native::exp(high_data[i].x() - max_val);
        high_data[i].y() = sycl::native::exp(high_data[i].y() - max_val);

        sum +=
            (low_data[i].x() + low_data[i].y() + high_data[i].x() +
             high_data[i].y());
      }

      for (int i = 1; i < WARP_SIZE; i *= 2)
        sum += sycl::permute_group_by_xor(
            sycl::ext::oneapi::experimental::this_sub_group(), sum, i);

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = sum;
        /*
        DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane < warp_num)
          sum = partialSum[lane];

        /*
        DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int i = 1; i < reduce_blocks; i *= 2) {
          sum += sycl::permute_group_by_xor(
              sycl::ext::oneapi::experimental::this_sub_group(), sum, i);
        }

        /*
        DPCT1007:14: Migration of cooperative_groups::thread_block_tile::shfl is
        not supported.
        */
        sum = g.shuffle(sum, item_ct1.get_local_id(2) / WARP_SIZE);
      }
      sum += 1e-6;
      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane);
        if (data_id < sequence_length) {
          vals[data_id] = conversion::to<T>(low_data[i].x() / sum);
          if ((data_id + reduceWidth) < sequence_length)
            vals[data_id + reduceWidth] =
                conversion::to<T>(low_data[i].y() / sum);
          if ((data_id + reduceWidth * 2) < sequence_length)
            vals[data_id + reduceWidth * 2] =
                conversion::to<T>(high_data[i].x() / sum);
          if ((data_id + reduceWidth * 3) < sequence_length)
            vals[data_id + reduceWidth * 3] =
                conversion::to<T>(high_data[i].y() / sum);
        }
      }
    }
  }
};

template <int iterations>
class attn_softmax_v2<float, iterations> {
 private:
  mutable float* vals;
  float* attn_mask;
  float* alibi;
  float layer_scale;
  bool triangular;
  bool recompute;
  bool local_attention;
  int window_size;
  int total_count;
  int heads;
  int sequence_length;
  int num_seq;
  int head_offset;
  int mask_stride;
  int mp_size;
  int reduceWidth;

 public:
  attn_softmax_v2(
      float* vals,
      float* attn_mask,
      float* alibi,
      float layer_scale,
      bool triangular,
      bool recompute,
      bool local_attention,
      int window_size,
      int total_count,
      int heads,
      int sequence_length,
      int num_seq,
      int head_offset,
      int mask_stride,
      int mp_size,
      int reduceWidth)
      : vals(vals),
        attn_mask(attn_mask),
        alibi(alibi),
        layer_scale(layer_scale),
        triangular(triangular),
        recompute(recompute),
        local_attention(local_attention),
        window_size(window_size),
        total_count(total_count),
        heads(heads),
        sequence_length(sequence_length),
        num_seq(num_seq),
        head_offset(head_offset),
        mask_stride(mask_stride),
        mp_size(mp_size),
        reduceWidth(reduceWidth) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    sycl::float4 data[MAX_REG_SIZE];

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = item_ct1.get_local_id(2) % reduceWidth;

    auto& partialSum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[MAX_WARP_NUM]>(sycl::ext::oneapi::experimental::this_group<3>());

    int iter_offset = item_ct1.get_group(2) * (warp_num / reduce_blocks) +
        (wid / reduce_blocks);
    if (iter_offset < total_count) {
      vals += (iter_offset * sequence_length);

      int batch_idx = iter_offset / (num_seq * heads);
      int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);
      mask_offset = mask_offset * sequence_length;
      int seq_id = iter_offset % num_seq;

      int real_seq_id =
          seq_id + (num_seq == sequence_length ? 0 : sequence_length);
      int window_stride4 =
          (local_attention && (real_seq_id >> 2) > (window_size >> 2))
          ? (real_seq_id >> 2) - (window_size >> 2)
          : 0;
      int window_stride = (local_attention && real_seq_id >= window_size)
          ? real_seq_id - window_size
          : -1;

      float max_val = minus_infinity;

      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane);
        bool check = (data_id >> 2) >= window_stride4;
        bool x_check = check && (data_id < sequence_length) &&
            (!triangular || (data_id <= seq_id)) && (data_id > window_stride);
        bool y_check = check && ((data_id + reduceWidth) < sequence_length) &&
            (!triangular || ((data_id + reduceWidth) <= seq_id)) &&
            ((data_id + reduceWidth) > window_stride);
        bool z_check = check &&
            ((data_id + reduceWidth * 2) < sequence_length) &&
            (!triangular || ((data_id + reduceWidth * 2) <= seq_id)) &&
            ((data_id + reduceWidth * 2) > window_stride);
        bool w_check = check &&
            ((data_id + reduceWidth * 3) < sequence_length) &&
            (!triangular || ((data_id + reduceWidth * 3) <= seq_id)) &&
            ((data_id + reduceWidth * 3) > window_stride);

        if (attn_mask) {
          data[i].x() = x_check
              ? vals[data_id] + attn_mask[data_id + mask_offset]
              : minus_infinity;
          data[i].y() = y_check ? vals[data_id + reduceWidth] +
                  attn_mask[data_id + mask_offset + reduceWidth]
                                : minus_infinity;
          data[i].z() = z_check ? vals[data_id + reduceWidth * 2] +
                  attn_mask[data_id + mask_offset + reduceWidth * 2]
                                : minus_infinity;
          data[i].w() = w_check ? vals[data_id + reduceWidth * 3] +
                  attn_mask[data_id + mask_offset + reduceWidth * 3]
                                : minus_infinity;
        } else {
          data[i].x() = x_check ? vals[data_id] : minus_infinity;
          data[i].y() = y_check ? vals[data_id + reduceWidth] : minus_infinity;
          data[i].z() =
              z_check ? vals[data_id + reduceWidth * 2] : minus_infinity;
          data[i].w() =
              w_check ? vals[data_id + reduceWidth * 3] : minus_infinity;
        }

        max_val = (data[i].x() > max_val ? data[i].x() : max_val);
        max_val = (data[i].y() > max_val ? data[i].y() : max_val);
        max_val = (data[i].z() > max_val ? data[i].z() : max_val);
        max_val = (data[i].w() > max_val ? data[i].w() : max_val);
      }

      for (int i = 1; i < WARP_SIZE; i *= 2) {
        auto temp = sycl::permute_group_by_xor(
            sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
        max_val = (temp > max_val ? temp : max_val);
      }

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = max_val;
        /*
        DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane < warp_num)
          max_val = partialSum[lane];

        /*
        DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int i = 1; i < reduce_blocks; i *= 2) {
          auto temp = sycl::permute_group_by_xor(
              sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
          max_val = (temp > max_val ? temp : max_val);
        }

        /*
        DPCT1007:15: Migration of cooperative_groups::thread_block_tile::shfl is
        not supported.
        */
        max_val = g.shuffle(max_val, item_ct1.get_local_id(2) / WARP_SIZE);
      }

      float sum = 0;
      for (int i = 0; i < iterations; i++) {
        data[i].x() = sycl::native::exp(data[i].x() - max_val);
        data[i].y() = sycl::native::exp(data[i].y() - max_val);
        data[i].z() = sycl::native::exp(data[i].z() - max_val);
        data[i].w() = sycl::native::exp(data[i].w() - max_val);

        sum += (data[i].x() + data[i].y() + data[i].z() + data[i].w());
      }

      for (int i = 1; i < WARP_SIZE; i *= 2)
        sum += sycl::permute_group_by_xor(
            sycl::ext::oneapi::experimental::this_sub_group(), sum, i);

      if (reduceWidth > WARP_SIZE) {
        if (lane == 0)
          partialSum[wid] = sum;
        /*
        DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane < warp_num)
          sum = partialSum[lane];

        /*
        DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int i = 1; i < reduce_blocks; i *= 2) {
          sum += sycl::permute_group_by_xor(
              sycl::ext::oneapi::experimental::this_sub_group(), sum, i);
        }

        /*
        DPCT1007:16: Migration of cooperative_groups::thread_block_tile::shfl is
        not supported.
        */
        sum = g.shuffle(sum, item_ct1.get_local_id(2) / WARP_SIZE);
      }
      sum += 1e-6;

      for (int i = 0; i < iterations; i++) {
        int data_id = i * (reduceWidth << 2) + (seq_lane);
        if (data_id < sequence_length) {
          vals[data_id] = data[i].x() / sum;
          if ((data_id + reduceWidth) < sequence_length)
            vals[data_id + reduceWidth] = data[i].y() / sum;
          if ((data_id + reduceWidth * 2) < sequence_length)
            vals[data_id + reduceWidth * 2] = data[i].z() / sum;
          if ((data_id + reduceWidth * 3) < sequence_length)
            vals[data_id + reduceWidth * 3] = data[i].w() / sum;
        }
      }
    }
  }
};

#define LAUNCH_ATTN_SOFTMAX_V2(iterations)                               \
  {                                                                      \
    dpct::has_capability_or_fail(                                        \
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
    stream->submit([&](sycl::handler& cgh) {                             \
      attn_softmax_v2<T, iterations> fn(                                 \
          vals,                                                          \
          mask,                                                          \
          alibi,                                                         \
          layer_scale,                                                   \
          triangular,                                                    \
          recompute,                                                     \
          local_attention,                                               \
          window_size,                                                   \
          total_count,                                                   \
          heads,                                                         \
          sequence_length,                                               \
          num_seq,                                                       \
          head_offset,                                                   \
          mask_stride,                                                   \
          mp_size,                                                       \
          reduce_width);                                                 \
                                                                         \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);      \
    });                                                                  \
  }

template <typename T>
void launch_attn_softmax_v2(
    T* vals,
    T* mask,
    T* alibi,
    float layer_scale,
    bool triangular,
    bool recompute,
    bool local_attention,
    int window_size,
    int batch_size,
    int heads,
    int num_seq,
    int sequence_length,
    int head_offset,
    int mask_stride,
    int mp_size,
    dpct::queue_ptr stream) {
  const int total_count = batch_size * heads * num_seq;

  // Scheduling Overview
  // 4 element unroll with power of 2 `reduce_width` threads to a ceiling of
  // `attn_threads` Each block should be partitioned into as many `reduce_width`
  // blocks as can be fit.
  constexpr int attn_threads = 256;
  constexpr int min_reduce_width = hw_warp_size;
  constexpr int internal_unroll = 4;

  // Handle internal unroll then round to next power of 2. Bump up to minimum
  // granularity.
  const int thread_steps_rounded =
      next_pow2((sequence_length + internal_unroll - 1) / internal_unroll);
  const int thread_steps_schedule = (thread_steps_rounded < min_reduce_width)
      ? min_reduce_width
      : thread_steps_rounded;
  // Bound reduce width to the number of threads
  const int reduce_width = (thread_steps_schedule < attn_threads)
      ? thread_steps_schedule
      : attn_threads;
  // Scale for the excess
  const int iterations = thread_steps_schedule / reduce_width;
  // Should be safe since reduce_width is capped to attn_threads
  const int partitions = attn_threads / reduce_width;

  // Launch params
  sycl::range<3> grid(1, 1, (total_count + partitions - 1) / partitions);
  sycl::range<3> block(1, 1, attn_threads);

  if (sequence_length <= 32768) {
    if (iterations == 1) {
      LAUNCH_ATTN_SOFTMAX_V2(1);
    } else if (iterations == 2) {
      LAUNCH_ATTN_SOFTMAX_V2(2);
    } else if (iterations == 4) {
      LAUNCH_ATTN_SOFTMAX_V2(4);
    } else if (iterations == 8) {
      LAUNCH_ATTN_SOFTMAX_V2(8);
    } else if (iterations == 16) {
      LAUNCH_ATTN_SOFTMAX_V2(16);
    } else if (iterations == 32) {
      LAUNCH_ATTN_SOFTMAX_V2(32);
    } else if (iterations == 64) {
      LAUNCH_ATTN_SOFTMAX_V2(64);
    }
  } else
    throw std::runtime_error("Unsupport Seq_Length!");
}

#define INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(T) \
  template void launch_attn_softmax_v2(       \
      T* vals,                                \
      T* mask,                                \
      T* alibi,                               \
      float layer_scale,                      \
      bool triangular,                        \
      bool recompute,                         \
      bool local_attention,                   \
      int window_size,                        \
      int batch_size,                         \
      int heads,                              \
      int num_seq,                            \
      int sequence_length,                    \
      int head_offset,                        \
      int mask_stride,                        \
      int mp_size,                            \
      dpct::queue_ptr stream);

INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(sycl::half);

/* #define DEF_ATTN_SOFTMAX_V2_HALF(_iter)             \ */
/*   template void attn_softmax_v2<sycl::half, _iter>( \ */
/*       sycl::half * vals,                            \ */
/*       sycl::half * mask,                            \ */
/*       sycl::half * alibi,                           \ */
/*       float layer_scale,                            \ */
/*       bool triangular,                              \ */
/*       bool recompute,                               \ */
/*       bool local_attention,                         \ */
/*       int window_size,                              \ */
/*       int total_count,                              \ */
/*       int heads,                                    \ */
/*       int sequence_length,                          \ */
/*       int num_seq,                                  \ */
/*       int head_offset,                              \ */
/*       int mask_stride,                              \ */
/*       int mp_size,                                  \ */
/*       int reduceWidth) */

/* #define DEF_ATTN_SOFTMAX_V2_BF16(_iter)                              \ */
/*   template void attn_softmax_v2<sycl::ext::oneapi::bfloat16, _iter>( \ */
/*       sycl::ext::oneapi::bfloat16 * vals,                            \ */
/*       sycl::ext::oneapi::bfloat16 * mask,                            \ */
/*       sycl::ext::oneapi::bfloat16 * alibi,                           \ */
/*       float layer_scale,                                             \ */
/*       bool triangular,                                               \ */
/*       bool recompute,                                                \ */
/*       bool local_attention,                                          \ */
/*       int window_size,                                               \ */
/*       int total_count,                                               \ */
/*       int heads,                                                     \ */
/*       int sequence_length,                                           \ */
/*       int num_seq,                                                   \ */
/*       int head_offset,                                               \ */
/*       int mask_stride,                                               \ */
/*       int mp_size,                                                   \ */
/*       int reduceWidth) */

/* #define FOREACH_ITERATIONS(cb) \ */
/*   cb(1);                       \ */
/*   cb(2);                       \ */
/*   cb(4);                       \ */
/*   cb(8);                       \ */
/*   cb(16);                      \ */
/*   cb(32);                      \ */
/*   cb(64) */

/* FOREACH_ITERATIONS(DEF_ATTN_SOFTMAX_V2_HALF); */
/* #ifdef BF16_AVAILABLE */
/* FOREACH_ITERATIONS(DEF_ATTN_SOFTMAX_V2_BF16); */
/* #endif */
