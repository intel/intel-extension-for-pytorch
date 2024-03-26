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
#include "inference_sycl_layers.h"

// only used to avoid compilation error due to lack of definition.
#ifndef BF16_AVAILABLE
using __nv_bfloat162 = sycl::half2;
#endif

// Bias add
template <typename T>
class bias_add_transform_0213 {
 private:
  T* output;
  T* k_cache;
  T* v_cache;
  const T* vals;
  const T* bias;
  int hidden_dim;
  int seq_length;
  unsigned seq_offset;
  int all_tokens;
  int heads;
  int head_stride;
  int num_kv;
  int rotary_dim;
  bool rotate_half;
  bool rotate_every_two;
  int head_ext;
  int max_out_tokens;
  float rope_theta;

 public:
  bias_add_transform_0213(
      T* output,
      T* k_cache,
      T* v_cache,
      const T* vals,
      const T* bias,
      int hidden_dim,
      int seq_length,
      unsigned seq_offset,
      int all_tokens,
      int heads,
      int head_stride,
      int num_kv,
      int rotary_dim,
      bool rotate_half,
      bool rotate_every_two,
      int head_ext,
      int max_out_tokens,
      float rope_theta)
      : output(output),
        k_cache(k_cache),
        v_cache(v_cache),
        vals(vals),
        bias(bias),
        hidden_dim(hidden_dim),
        seq_length(seq_length),
        seq_offset(seq_offset),
        all_tokens(all_tokens),
        heads(heads),
        head_stride(head_stride),
        num_kv(num_kv),
        rotary_dim(rotary_dim),
        rotate_half(rotate_half),
        rotate_every_two(rotate_every_two),
        head_ext(head_ext),
        max_out_tokens(max_out_tokens),
        rope_theta(rope_theta) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    using T2 = typename std::conditional<
        std::is_same<T, sycl::half>::value,
        sycl::half2,
        sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type;
    unsigned half_dim = (rotary_dim << 3) >> 1;
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_group(1); // Sequence ID (0-127)
    int cnt = item_ct1.get_group(0) / head_ext; // Hidden count
    int d2 = item_ct1.get_local_id(1) +
        (item_ct1.get_group(0) % head_ext) * (heads / head_ext); // Head (0-11)
    int d3 = item_ct1.get_local_id(2); // Values (groups of 4)

    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : max_out_tokens);
    int d0_out_stride = hidden_dim * (cnt == 0 ? seq_length : max_out_tokens);

    sycl::float4 vals_arr;
    sycl::float4 output_arr;

    T2* vals_half = reinterpret_cast<T2*>(&vals_arr);
    T2* output_half = reinterpret_cast<T2*>(&output_arr);

    const sycl::float4* vals_vec = reinterpret_cast<const sycl::float4*>(vals);
    sycl::float4* output_vec = reinterpret_cast<sycl::float4*>(
        cnt == 0 ? output : (cnt == 1 ? k_cache : v_cache));

    vals_vec += (d0 * (d1_stride + num_kv * 2 * d2_stride) * seq_length);
    vals_vec += (d1 * (d1_stride + num_kv * 2 * d2_stride));
    vals_vec += (cnt == 0 ? 0 : d1_stride) +
        (cnt == 0 ? 0 : (cnt - 1) * num_kv * d2_stride);
    vals_vec += ((cnt == 0 ? d2 : (d2 / head_stride)) * d2_stride);

    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_out_stride);
    output_vec += (d2 * d2_out_stride);

    unsigned seq_id = d1 + seq_offset;

    int lane = d3 & 0x1f;
    if (cnt < 2 && rotary_dim > 0 && d3 < rotary_dim) {
      sycl::float4 q = vals_vec[d3];
      T2* q_h = reinterpret_cast<T2*>(&q);
      if (rotate_every_two) {
#pragma unroll
        for (int o = 0; o < 4; o++) {
          float inv_freq =
              (float)(((d3 << 2) + o) * 2) / (float)(rotary_dim << 3);
          inv_freq = 1.0 / dpct::pow(rope_theta, inv_freq) * (float)seq_id;
          float q_data[2];
          q_data[0] = conversion::to<float>(q_h[o][0]);
          q_data[1] = conversion::to<float>(q_h[o][1]);
          q_h[o][0] = conversion::to<T>(
              -1.0 * q_data[1] * sycl::sin(inv_freq) +
              q_data[0] * sycl::cos(inv_freq));
          q_h[o][1] = conversion::to<T>(
              q_data[0] * sycl::sin(inv_freq) +
              q_data[1] * sycl::cos(inv_freq));
        }
      }
      output_vec[d3] = q;
    } else
      output_vec[d3] = vals_vec[d3];
  }
};

template <>
class bias_add_transform_0213<float> {
 private:
  float* output;
  float* k_cache;
  float* v_cache;
  const float* vals;
  const float* bias;
  int hidden_dim;
  int seq_length;
  int all_tokens;
  unsigned seq_offset;
  int heads;
  int head_stride;
  int num_kv;
  int rotary_dim;
  bool rotate_half;
  bool rotate_every_two;
  int head_ext;
  int max_out_tokens;
  float rope_theta;

 public:
  bias_add_transform_0213(
      float* output,
      float* k_cache,
      float* v_cache,
      const float* vals,
      const float* bias,
      int hidden_dim,
      int seq_length,
      int all_tokens,
      unsigned seq_offset,
      int heads,
      int head_stride,
      int num_kv,
      int rotary_dim,
      bool rotate_half,
      bool rotate_every_two,
      int head_ext,
      int max_out_tokens,
      float rope_theta)
      : output(output),
        k_cache(k_cache),
        v_cache(v_cache),
        vals(vals),
        bias(bias),
        hidden_dim(hidden_dim),
        seq_length(seq_length),
        seq_offset(seq_offset),
        all_tokens(all_tokens),
        heads(heads),
        head_stride(head_stride),
        num_kv(num_kv),
        rotary_dim(rotary_dim),
        rotate_half(rotate_half),
        rotate_every_two(rotate_every_two),
        head_ext(head_ext),
        max_out_tokens(max_out_tokens),
        rope_theta(rope_theta) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_group(1); // Sequence ID (0-127)
    int cnt = item_ct1.get_group(0) / head_ext; // Hidden count
    int d2 = item_ct1.get_local_id(1) +
        (item_ct1.get_group(0) % head_ext) * (heads / head_ext); // Head (0-11)
    int d3 = item_ct1.get_local_id(2); // Values (groups of 4)

    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : max_out_tokens);
    int d0_out_stride = hidden_dim * (cnt == 0 ? seq_length : max_out_tokens);

    const sycl::float4* vals_vec = reinterpret_cast<const sycl::float4*>(vals);
    sycl::float4* output_vec = reinterpret_cast<sycl::float4*>(
        cnt == 0 ? output : (cnt == 1 ? k_cache : v_cache));

    vals_vec += (d0 * (d1_stride + num_kv * 2 * d2_stride) * seq_length);
    vals_vec += d1 * (d1_stride + num_kv * 2 * d2_stride);
    vals_vec += (cnt == 0 ? 0 : d1_stride) +
        (cnt == 0 ? 0 : (cnt - 1) * num_kv * d2_stride);
    vals_vec += ((cnt == 0 ? d2 : (d2 / head_stride)) * d2_stride);

    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_out_stride);
    output_vec += (d2 * d2_out_stride);

    unsigned seq_id = d1 + seq_offset;
    sycl::float4 inputs = vals_vec[d3];
    int lane = d3 & 0x1f;
    if (cnt < 2 && rotary_dim > 0 && d3 < rotary_dim) {
      sycl::float4 q = vals_vec[d3];
      sycl::float2* q_f = reinterpret_cast<sycl::float2*>(&q);
      if (rotate_every_two) {
#pragma unroll
        for (int o = 0; o < 2; o++) {
          float inv_freq =
              (float)(((d3 << 1) + o) * 2) / (float)(rotary_dim << 2);
          inv_freq = 1.0 / dpct::pow(rope_theta, inv_freq) * (float)seq_id;
          q_f[o].x() =
              (-1.0 * q_f[o].y() * sycl::sin(inv_freq) +
               q_f[o].x() * sycl::cos(inv_freq));
          q_f[o].y() =
              (q_f[o].x() * sycl::sin(inv_freq) +
               q_f[o].y() * sycl::cos(inv_freq));
        }
      }
      output_vec[d3] = q;
    } else
      output_vec[d3] = inputs;
  }
};

#define ATTN_H 3
#define MAX_SEQ_LINE 10

// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(
    float* output,
    float* k_cache,
    float* v_cache,
    const float* vals,
    const float* bias,
    int batch_size,
    int seq_length,
    unsigned seq_offset,
    int all_tokens,
    int hidden_dim,
    int heads,
    int num_kv,
    int rotary_dim,
    bool rotate_half,
    bool rotate_every_two,
    dpct::queue_ptr stream,
    int trans_count,
    int max_out_tokens,
    float rope_theta) {
  hidden_dim >>= 2;
  int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

  sycl::range<3> block_dim(1, (heads / head_ext), hidden_dim / heads);
  sycl::range<3> grid_dim((trans_count * head_ext), seq_length, batch_size);

  bias_add_transform_0213 fn(
      output,
      k_cache,
      v_cache,
      vals,
      bias,
      hidden_dim,
      seq_length,
      seq_offset,
      0,
      heads,
      num_kv > 0 ? (heads / num_kv) : 1,
      num_kv > 0 ? num_kv : heads,
      rotary_dim >> 2,
      rotate_half,
      rotate_every_two,
      head_ext,
      max_out_tokens,
      rope_theta);
  stream->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
}

template <typename T>
void launch_bias_add_transform_0213(
    T* output,
    T* k_cache,
    T* v_cache,
    const T* vals,
    const T* bias,
    int batch_size,
    int seq_length,
    unsigned seq_offset,
    int all_tokens,
    int hidden_dim,
    int heads,
    int num_kv,
    int rotary_dim,
    bool rotate_half,
    bool rotate_every_two,
    dpct::queue_ptr stream,
    int trans_count,
    int max_out_tokens,
    float rope_theta) {
  hidden_dim >>= 3;
  int head_ext = 1; // (hidden_dim - 1) / MAX_THREADS + 1;
  sycl::range<3> block_dim(1, (heads / head_ext), hidden_dim / heads);
  sycl::range<3> grid_dim((trans_count * head_ext), seq_length, batch_size);
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    bias_add_transform_0213 fn(
        output,
        k_cache,
        v_cache,
        vals,
        bias,
        hidden_dim,
        seq_length,
        seq_offset,
        all_tokens,
        heads,
        num_kv > 0 ? (heads / num_kv) : 1,
        num_kv > 0 ? num_kv : heads,
        rotary_dim >> 3,
        rotate_half,
        rotate_every_two,
        head_ext,
        max_out_tokens,
        rope_theta);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
  }
}

#define INSTANTIATE_LAUNCH_BIAS_ADD_TRANSFORM_0213(T) \
  template void launch_bias_add_transform_0213<T>(    \
      T*,                                             \
      T*,                                             \
      T*,                                             \
      const T*,                                       \
      const T*,                                       \
      int,                                            \
      int,                                            \
      unsigned,                                       \
      int,                                            \
      int,                                            \
      int,                                            \
      int,                                            \
      int,                                            \
      bool,                                           \
      bool,                                           \
      dpct::queue_ptr,                                \
      int,                                            \
      int,                                            \
      float)

#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_ADD_TRANSFORM_0213(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_BIAS_ADD_TRANSFORM_0213(sycl::half);

// Bias add

/* void pad_add_transform_0213(float* output, */
/*                                        const float* vals, */
/*                                        int hidden_dim, */
/*                                        int seq_length, */
/*                                        int padded_seq_len, */
/*                                        int heads, */
/*                                        int padded_head_size) */
/* { */
/* } */

template <typename T>
class pad_add_transform_0213 {
 private:
  T* output;
  const T* vals;
  int hidden_dim;
  int seq_length;
  int padded_seq_len;
  int heads;
  int padded_head_size;

 public:
  pad_add_transform_0213(
      T* output,
      const T* vals,
      int hidden_dim,
      int seq_length,
      int padded_seq_len,
      int heads,
      int padded_head_size)
      : output(output),
        vals(vals),
        hidden_dim(hidden_dim),
        seq_length(seq_length),
        padded_seq_len(padded_seq_len),
        heads(heads),
        padded_head_size(padded_head_size) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    using T2 = typename std::conditional<
        std::is_same<T, sycl::half>::value,
        sycl::half2,
        sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type;
    sycl::float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++)
      ZERO_h[i] = zero_h;

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_group(1) * item_ct1.get_local_range(0) +
        item_ct1.get_local_id(0); // Sequence ID (0-127)
    int d2 = item_ct1.get_local_id(1); // Head (0-11)
    int d3 = item_ct1.get_local_id(2); // Values (groups of 4)

    int d2_out_stride = padded_head_size * padded_seq_len;
    int d0_out_stride = heads * d2_out_stride;

    const sycl::float4* vals_vec = reinterpret_cast<const sycl::float4*>(vals);
    sycl::float4* output_vec = reinterpret_cast<sycl::float4*>(output);

    vals_vec += (d0 * d0_stride);
    vals_vec += (d1 * d1_stride);
    vals_vec += (d2 * d2_stride);

    output_vec += (d1 * padded_head_size);
    output_vec += (d0 * d0_out_stride);
    output_vec += (d2 * d2_out_stride);

    if (d3 < d2_stride && d1 < seq_length)
      output_vec[d3] = vals_vec[d3];
    else
      output_vec[d3] = ZERO;
  }
};

// [B S C*H] - > C * [B A S N]
template <>
void launch_pad_add_transform_0213<float>(
    float* output,
    const float* vals,
    int batch_size,
    int hidden_dim,
    int seq_length,
    int padded_seq_len,
    int heads,
    int padded_head_size,
    dpct::queue_ptr stream) {}

template <typename T>
void launch_pad_add_transform_0213(
    T* output,
    const T* vals,
    int batch_size,
    int hidden_dim,
    int seq_length,
    int padded_seq_len,
    int heads,
    int padded_head_size,
    dpct::queue_ptr stream) {
  hidden_dim >>= 3;
  sycl::range<3> block_dim(2, heads, (padded_head_size >> 3));
  sycl::range<3> grid_dim(1, padded_seq_len / 2, batch_size);
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    pad_add_transform_0213 fn(
        output,
        vals,
        hidden_dim,
        seq_length,
        padded_seq_len,
        heads,
        padded_head_size >> 3);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
  }
}

#define INSTANTIATE_LAUNCH_PAD_ADD_TRANSFORM_0213_SIMPLE(T) \
  template void launch_pad_add_transform_0213<T>(           \
      T*, const T*, int, int, int, int, int, int, dpct::queue_ptr);

INSTANTIATE_LAUNCH_PAD_ADD_TRANSFORM_0213_SIMPLE(sycl::half);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_PAD_ADD_TRANSFORM_0213_SIMPLE(sycl::ext::oneapi::bfloat16);
#endif

// Bias add
/* template <typename T> */
/* class bias_add_transform_0213 { */
/* private: */
/*   T* output; */
/*   const T* vals; */
/*   const T* bias; */
/*   int hidden_dim; */
/*   int seq_length; */
/*   int heads; */
/*   int head_ext; */
/* }; */

/* template <> */
/* void bias_add_transform_0213<float>( */
/*     float* output, */
/*     const float* vals, */
/*     const float* bias, */
/*     int hidden_dim, */
/*     int seq_length, */
/*     int heads, */
/*     int head_ext) { */
/*   auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>(); */
/*   int d0_stride = hidden_dim * seq_length; */
/*   int d1_stride = hidden_dim; */
/*   int d2_stride = hidden_dim / heads; */

/*   int d0_out_stride = d0_stride; */
/*   int d1_out_stride = d2_stride; */
/*   int d2_out_stride = d2_stride * seq_length; */

/*   int d0 = item_ct1.get_group(2); // Batch */
/*   int d1 = item_ct1.get_group(1); // Sequence ID (0-127) */
/*   int cnt = item_ct1.get_group(0) / head_ext; // Hidden count */
/*   int d2 = item_ct1.get_local_id(1) + */
/*       (item_ct1.get_group(0) % head_ext) * (heads / head_ext); // Head (0-11)
 */
/*   int d3 = item_ct1.get_local_id(2); // Values (groups of 4) */

/*   const sycl::float4* vals_vec = reinterpret_cast<const sycl::float4*>(vals);
 */
/*   const sycl::float4* bias_vec = reinterpret_cast<const sycl::float4*>(bias);
 */
/*   sycl::float4* output_vec = reinterpret_cast<sycl::float4*>(output); */

/*   sycl::float4 inputs = vals_vec */
/*       [d0 * d0_stride * (item_ct1.get_group_range(0) / head_ext) + */
/*        cnt * d1_stride + */
/*        d1 * d1_stride * (item_ct1.get_group_range(0) / head_ext) + */
/*        d2 * d2_stride + d3]; */
/*   sycl::float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3]; */

/*   sycl::float4 outputs; */
/*   outputs.x() = inputs.x() + biases.x(); */
/*   outputs.y() = inputs.y() + biases.y(); */
/*   outputs.z() = inputs.z() + biases.z(); */
/*   outputs.w() = inputs.w() + biases.w(); */

/*   output_vec */
/*       [cnt * d0_out_stride * item_ct1.get_group_range(2) + d0 * d0_out_stride
 * + */
/*        d1 * d1_out_stride + d2 * d2_out_stride + d3] = outputs; */
/* } */

/* template <typename T> */
/* void bias_add_transform_0213( */
/*     T* output, */
/*     const T* vals, */
/*     const T* bias, */
/*     int hidden_dim, */
/*     int seq_length, */
/*     int heads, */
/*     int head_ext) { */
/*   auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>(); */
/*   using T2 = typename std::conditional< */
/*       std::is_same<T, sycl::half>::value, */
/*       sycl::half2, */
/*       sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type; */
/*   int d0_stride = hidden_dim * seq_length; */
/*   int d1_stride = hidden_dim; */
/*   int d2_stride = hidden_dim / heads; */

/*   int d2_out_stride = d2_stride * seq_length; */

/*   int d0 = item_ct1.get_group(2); // Batch */
/*   int d1 = item_ct1.get_group(1); // Sequence ID (0-127) */
/*   int cnt = item_ct1.get_group(0) / head_ext; // Hidden count */
/*   int d2 = item_ct1.get_local_id(1) + */
/*       (item_ct1.get_group(0) % head_ext) * (heads / head_ext); // Head (0-11)
 */
/*   int d3 = item_ct1.get_local_id(2); // Values (groups of 4) */

/*   sycl::float4 vals_arr; */
/*   sycl::float4 bias_arr; */
/*   sycl::float4 output_arr; */
/*   T2* vals_half = reinterpret_cast<T2*>(&vals_arr); */
/*   T2* bias_half = reinterpret_cast<T2*>(&bias_arr); */
/*   T2* output_half = reinterpret_cast<T2*>(&output_arr); */

/*   const sycl::float4* vals_vec = reinterpret_cast<const sycl::float4*>(vals);
 */
/*   const sycl::float4* bias_vec = reinterpret_cast<const sycl::float4*>(bias);
 */
/*   sycl::float4* output_vec = reinterpret_cast<sycl::float4*>(output); */

/*   vals_vec += (d0 * d0_stride * (item_ct1.get_group_range(0) / head_ext)); */
/*   vals_vec += (d1 * d1_stride * (item_ct1.get_group_range(0) / head_ext)); */
/*   vals_vec += (cnt * d1_stride); */
/*   vals_vec += (d2 * d2_stride); */

/*   bias_vec += (cnt * d1_stride); */
/*   bias_vec += (d2 * d2_stride); */

/*   output_vec += (cnt * d0_stride * item_ct1.get_group_range(2)); */
/*   output_vec += (d1 * d2_stride); */
/*   output_vec += (d0 * d0_stride); */
/*   output_vec += (d2 * d2_out_stride); */

/*   bias_arr = bias_vec[d3]; */
/*   vals_arr = vals_vec[d3]; */

/*   output_half[0] = vals_half[0] + bias_half[0]; */
/*   output_half[1] = vals_half[1] + bias_half[1]; */
/*   output_half[2] = vals_half[2] + bias_half[2]; */
/*   output_half[3] = vals_half[3] + bias_half[3]; */
/*   output_vec[d3] = output_arr; */
/* } */

/*template <typename T> */
/*void bias_add_transform_0213_v2( */
/*    T* output, */
/*    const T* vals, */
/*    const T* bias, */
/*    int hidden_dim, */
/*    int seq_length, */
/*    int heads) { */
/*  auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>(); */
/*  using T2 = typename std::conditional< */
/*      std::is_same<T, sycl::half>::value, */
/*      sycl::half2, */
/*      sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type; */
/*  auto& in_data = */
/*      *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::float4[3072]>(
 */
/*          sycl::ext::oneapi::experimental::this_group<3>()); */

/*  int d0_stride = hidden_dim * seq_length; */
/*  int d1_stride = hidden_dim; */
/*  int d2_stride = hidden_dim / heads; */
/*  int iteration_stride = */
/*      d1_stride * item_ct1.get_local_range(0); // Hidden * 3 / 8 */
/*  int batch_stride = */
/*      d0_stride * item_ct1.get_local_range(0); // Hidden * S * 3 / 8 */

/*  int d0_out_stride = d0_stride; */
/*  int d1_out_stride = d2_stride; */
/*  int d2_out_stride = d2_stride * seq_length; */

/*  int d0 = item_ct1.get_group(2); // Batch */
/*  int d1 = item_ct1.get_group(1); // Sequence ID (0-127) */
/*  int cnt = item_ct1.get_local_id(0); // blockIdx.z; // Hidden count */
/*  int d2 = item_ct1.get_local_id(1); // Head (0-11) */
/*  int d3 = item_ct1.get_local_id(2); // Values (groups of 4) */

/*  sycl::float4 vals_arr[1]; */
/*  sycl::float4 bias_arr[1]; */
/*  sycl::float4 output_arr[1]; */
/*  T2* vals_half = reinterpret_cast<T2*>(vals_arr); */
/*  T2* bias_half = reinterpret_cast<T2*>(bias_arr); */
/*  T2* output_half = reinterpret_cast<T2*>(output_arr); */

/*  const sycl::float4* vals_vec = reinterpret_cast<const sycl::float4*>(vals);
 */
/*  const sycl::float4* bias_vec = reinterpret_cast<const sycl::float4*>(bias);
 */
/*  sycl::float4* output_vec = reinterpret_cast<sycl::float4*>(output); */

/*  int iter_index = cnt * d1_stride + d2 * d2_stride + d3; */
/*  int input_offset = d0 * batch_stride + d1 * (iteration_stride << 1); */
/*  bias_arr[0] = bias_vec[iter_index]; */

/*#pragma unroll */
/*  for (int iter = 0; iter < 2; iter++) { */
/*    int iter_id = iter * iteration_stride + iter_index; */
/*    vals_arr[0] = vals_vec[input_offset + iter_id]; */

/*    output_half[0] = vals_half[0] + bias_half[0]; */
/*    output_half[1] = vals_half[1] + bias_half[1]; */
/*    output_half[2] = vals_half[2] + bias_half[2]; */
/*    output_half[3] = vals_half[3] + bias_half[3]; */

/*    in_data[iter_id] = output_arr[0]; */
/*  } */
/*  /1* */
/*  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with */
/*  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better */
/*  performance if there is no access to global memory. */
/*  *1/ */
/*  item_ct1.barrier(); */

/*  iteration_stride = */
/*      item_ct1.get_local_range(0) * (item_ct1.get_local_range(1) >> 1); */
/*  int matrix_stride = (d0_out_stride * item_ct1.get_group_range(2)); */
/*  int head_count = (d2 >> 1) + cnt * (item_ct1.get_local_range(1) >> 1); */

/*  int out_index = d0 * d0_out_stride + d1 * (d1_out_stride << 1) + d3 + */
/*      (d2 % 2) * d2_stride; */

/*#pragma unroll */
/*  for (int iter = 0; iter < 2; iter++) { */
/*    int iter_row = (iter * iteration_stride) + head_count; */
/*    int iter_offset = (iter_row % item_ct1.get_local_range(1)) * d2_out_stride
 * + */
/*        (iter_row / item_ct1.get_local_range(1)) * matrix_stride; */
/*    output_vec[out_index + iter_offset] = in_data */
/*        [iter_row * d2_stride + d3 + */
/*         (d2 % 2) * (d1_stride * item_ct1.get_local_range(0))]; */
/*  } */
/*} */

template <typename T>
class transform4d_0213 {
 private:
  T* out;
  const T* in;
  int heads;
  int seq_length;
  int hidden_dim;
  int head_ext;

 public:
  transform4d_0213(
      T* out,
      const T* in,
      int heads,
      int seq_length,
      int hidden_dim,
      int head_ext)
      : out(out),
        in(in),
        heads(heads),
        seq_length(seq_length),
        hidden_dim(hidden_dim),
        head_ext(head_ext) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int d0_stride = hidden_dim * (seq_length / head_ext);
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_local_id(1) +
        (item_ct1.get_group(0) % head_ext) * (heads / head_ext); // Head
    int d2 = item_ct1.get_group(0) / head_ext; // Sequence
    int cnt = item_ct1.get_group(1); // Hidden count
    int d3 = item_ct1.get_local_id(2); // Values (groups of 8)

    const sycl::float4* in_vec = reinterpret_cast<const sycl::float4*>(in);
    sycl::float4* out_vec = reinterpret_cast<sycl::float4*>(out);

    in_vec += (cnt * d0_stride * item_ct1.get_group_range(2));
    in_vec += (d0 * d0_stride);
    in_vec += (d2 * d2_stride);
    in_vec += (d1 * d2_stride * seq_length);

    out_vec += (cnt * d1_stride);
    out_vec += (d1 * d2_stride);
    out_vec += (d0 * d0_stride * item_ct1.get_group_range(1));
    out_vec += (d2 * d1_stride * item_ct1.get_group_range(1));

    out_vec[d3] = in_vec[d3];
  }
};

template <>
class transform4d_0213<float> {
 private:
  float* out;
  const float* in;
  int heads;
  int seq_length;
  int hidden_dim;
  int head_ext;

 public:
  transform4d_0213(
      float* out,
      const float* in,
      int heads,
      int seq_length,
      int hidden_dim,
      int head_ext)
      : out(out),
        in(in),
        heads(heads),
        seq_length(seq_length),
        hidden_dim(hidden_dim),
        head_ext(head_ext) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = d0_stride / heads;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = hidden_dim;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_group(1) /
        ((seq_length - 1) / item_ct1.get_local_range(1) + 1); // Head
    int d2 = (item_ct1.get_local_id(1) +
              item_ct1.get_local_range(1) * item_ct1.get_group(1)) %
        seq_length;
    int cnt = item_ct1.get_group(0);
    int d3 = item_ct1.get_local_id(2); // Values (groups of 8)

    if (d2 < seq_length) {
      const sycl::float4* in_vec = reinterpret_cast<const sycl::float4*>(in);
      sycl::float4* out_vec = reinterpret_cast<sycl::float4*>(out);

      sycl::float4 vals_vec = in_vec
          [cnt * d0_stride * item_ct1.get_group_range(2) + d0 * d0_stride +
           d1 * d1_stride + d2 * d2_stride + d3];
      out_vec
          [d0 * d0_out_stride * item_ct1.get_group_range(0) +
           cnt * d2_out_stride + d1 * d1_out_stride +
           d2 * d2_out_stride * item_ct1.get_group_range(0) + d3] = vals_vec;
    }
  }
};

template <typename T>
class transform4d_0213_v2 {
 private:
  T* out;
  const T* in;
  int heads;
  int seq_length;
  int hidden_dim;

 public:
  transform4d_0213_v2(
      T* out,
      const T* in,
      int heads,
      int seq_length,
      int hidden_dim)
      : out(out),
        in(in),
        heads(heads),
        seq_length(seq_length),
        hidden_dim(hidden_dim) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    auto& in_data = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::float4[3072]>(sycl::ext::oneapi::experimental::this_group<3>());

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = item_ct1.get_group(2); // Batch
    int d1 = item_ct1.get_local_id(1); // Head
    int d2 = item_ct1.get_group(1); // Sequence
    int cnt = item_ct1.get_local_id(0); // Hidden count
    int d3 = item_ct1.get_local_id(2); // Values (groups of 8)

    const sycl::float4* in_vec = reinterpret_cast<const sycl::float4*>(in);
    sycl::float4* out_vec = reinterpret_cast<sycl::float4*>(out);

    int input_offset =
        d0 * d0_stride + d2 * (d2_stride << 1) + d3 + (d1 % 2) * d2_stride;
    int head_count = (d1 >> 1) + cnt * (item_ct1.get_local_range(1) >> 1);
    int iteration_stride =
        item_ct1.get_local_range(0) * (item_ct1.get_local_range(1) >> 1);
    int matrix_stride = (d0_stride * item_ct1.get_group_range(2));

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
      int iter_row = iter * iteration_stride + head_count;
      int iter_offset = (iter_row % item_ct1.get_local_range(1)) * d2_stride;

      in_data
          [d3 + iter_offset +
           (iter_row / item_ct1.get_local_range(1) +
            (d1 % 2) * item_ct1.get_local_range(0)) *
               d1_stride] = in_vec
              [input_offset + iter_offset * seq_length +
               (iter_row / item_ct1.get_local_range(1)) * matrix_stride];
    }
    /*
    DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    iteration_stride = d1_stride * item_ct1.get_local_range(0);
    int iter_index = cnt * d1_stride + d1 * d2_stride + d3;
    int output_offset = d0 * d0_stride * item_ct1.get_local_range(0) +
        d2 * (iteration_stride << 1);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
      int iter_id = iter * iteration_stride + iter_index;
      out_vec[output_offset + iter_id] = in_data[iter_id];
    }
  }
};

// 3 * [B A S N] - > [B S C*H]
template <>
void launch_transform4d_0213<float>(
    float* out,
    const float* in,
    int batch_size,
    int heads,
    int seq_length,
    int hidden_dim,
    dpct::queue_ptr stream,
    int trans_count) {
  hidden_dim >>= 2;
  sycl::range<3> grid_dims(
      trans_count, heads * ((seq_length - 1) / 8 + 1), batch_size);
  sycl::range<3> block_dims(1, 8, hidden_dim / heads);
  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
    transform4d_0213<float> fn(out, in, heads, seq_length, hidden_dim, 1);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims), fn);
  }
}

template <typename T>
void launch_transform4d_0213(
    T* out,
    const T* in,
    int batch_size,
    int heads,
    int seq_length,
    int hidden_dim,
    dpct::queue_ptr stream,
    int trans_count) {
  hidden_dim >>= 3;
  int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
  sycl::range<3> grid_dims((seq_length * head_ext), trans_count, batch_size);
  sycl::range<3> block_dims(1, (heads / head_ext), hidden_dim / heads);
  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
    transform4d_0213 fn(out, in, heads, seq_length, hidden_dim, head_ext);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims), fn);
  }
}

#define INSTANTIATE_2B_LAUNCH_TRANSFORM4D(T) \
  template void launch_transform4d_0213<T>(  \
      T*, const T*, int, int, int, int, dpct::queue_ptr, int);

INSTANTIATE_2B_LAUNCH_TRANSFORM4D(sycl::half)
#ifdef BF16_AVAILABLE
INSTANTIATE_2B_LAUNCH_TRANSFORM4D(sycl::ext::oneapi::bfloat16)
#endif
