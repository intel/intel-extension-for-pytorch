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
#include "memory_access_utils.h"

#define MAX_CAP 4
#define MAX_SEQ 2048

// only used to avoid compilation error due to lack of definition.
#ifndef BF16_AVAILABLE
using __nv_bfloat162 = sycl::half2;
#endif

inline float gelu(const float x) {
  constexpr float sqrt_param = 0.79788456080286535587989211986876f;
  constexpr float mul_param = 0.044715;
  return x * 0.5f *
      (1.0f + sycl::tanh(sqrt_param * (x + mul_param * x * x * x)));
}

template <typename T>
class fused_bias_gelu {
 private:
  T* input;
  const T* bias;
  int total_count;
  int intermediate_size;

 public:
  fused_bias_gelu(
      T* input,
      const T* bias,
      int total_count,
      int intermediate_size)
      : input(input),
        bias(bias),
        total_count(total_count),
        intermediate_size(intermediate_size) {}
  void operator()(sycl::nd_item<3>) const {
    // Input restriction: intermediate_size % vals_per_access == 0
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_local_id(2)) *
        values_per_access;

    if (offset < total_count) {
      T data[values_per_access];
      T data_bias[values_per_access];
      mem_access::load_global<granularity>(data, input + offset);
      mem_access::load_global<granularity>(
          data_bias, bias + (offset % intermediate_size), bias != nullptr);

#pragma unroll
      for (int i = 0; i < values_per_access; i++) {
        float data_f = conversion::to<float>(data[i]);
        float bias_f = conversion::to<float>(data_bias[i]);
        data[i] = conversion::to<T>(gelu(data_f + bias_f));
      }

      mem_access::store_global<granularity>(input + offset, data);
    }
  }
};

template <typename T>
void launch_bias_gelu(
    T* input,
    const T* bias,
    int intermediate_size,
    int batch_size,
    dpct::queue_ptr stream) {
  constexpr int threads = 1024;
  constexpr int granularity = 16;

  const int total_count = batch_size * intermediate_size;
  const int elems_per_block = threads * (granularity / sizeof(T));
  sycl::range<3> block_dims(1, 1, threads);
  sycl::range<3> grid_dims(
      1, 1, (total_count + elems_per_block - 1) / elems_per_block);

  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    fused_bias_gelu fn(input, bias, total_count, intermediate_size);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims), fn);
  }
}

#define INSTANTIATE_LAUNCH_BIAS_GELU(T) \
  template void launch_bias_gelu<T>(T*, const T*, int, int, dpct::queue_ptr);

INSTANTIATE_LAUNCH_BIAS_GELU(float)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_GELU(sycl::ext::oneapi::bfloat16)
#endif
INSTANTIATE_LAUNCH_BIAS_GELU(sycl::half)

/*
In-place channels-last bias add
*/
template <typename T>
class fused_bias_add {
 private:
  T* input;
  const T* bias;
  int total_count;
  int intermediate_size;

 public:
  fused_bias_add(
      T* input,
      const T* bias,
      int total_count,
      int intermediate_size)
      : input(input),
        bias(bias),
        total_count(total_count),
        intermediate_size(intermediate_size) {}
  void operator()(sycl::nd_item<3>) const {
    // Input restriction: intermediate_size % vals_per_access == 0
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_local_id(2)) *
        values_per_access;

    if (offset < total_count) {
      T data[values_per_access];
      T data_bias[values_per_access];
      mem_access::load_global<granularity>(data, input + offset);
      mem_access::load_global<granularity>(
          data_bias, bias + (offset % intermediate_size), bias != nullptr);

#pragma unroll
      for (int i = 0; i < values_per_access; i++) {
        float data_f = conversion::to<float>(data[i]);
        float bias_f = conversion::to<float>(data_bias[i]);
        data[i] = conversion::to<T>(data_f + bias_f);
      }

      mem_access::store_global<granularity>(input + offset, data);
    }
  }
};

template <typename T>
void launch_bias_add(
    T* input,
    const T* bias,
    int intermediate_size,
    int batch_size,
    dpct::queue_ptr stream) {
  constexpr int threads = 1024;
  constexpr int granularity = 16;

  const int total_count = batch_size * intermediate_size;
  const int elems_per_block = threads * (granularity / sizeof(T));
  sycl::range<3> block_dims(1, 1, threads);
  sycl::range<3> grid_dims(
      1, 1, (total_count + elems_per_block - 1) / elems_per_block);

  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    fused_bias_add fn(input, bias, total_count, intermediate_size);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims), fn);
  }
}

#define INSTANTIATE_LAUNCH_BIAS_ADD(T) \
  template void launch_bias_add<T>(T*, const T*, int, int, dpct::queue_ptr);

INSTANTIATE_LAUNCH_BIAS_ADD(float)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_ADD(sycl::ext::oneapi::bfloat16)
#endif
INSTANTIATE_LAUNCH_BIAS_ADD(sycl::half)

template <typename T>
class fused_bias_residual {
 private:
  T* residual;
  const T* hidden_state;
  const T* attn;
  const T* bias;
  const T* attn_bias;
  const int total_count;
  const int intermediate_size;
  const float mp_scale;
  const bool preln;

 public:
  fused_bias_residual(
      T* residual,
      const T* hidden_state,
      const T* attn,
      const T* bias,
      const T* attn_bias,
      const int total_count,
      const int intermediate_size,
      const float mp_scale,
      const bool preln)
      : residual(residual),
        hidden_state(hidden_state),
        attn(attn),
        bias(bias),
        attn_bias(attn_bias),
        total_count(total_count),
        intermediate_size(intermediate_size),
        mp_scale(mp_scale),
        preln(preln) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    using T2 = typename std::conditional<
        std::is_same<T, sycl::half>::value,
        sycl::half2,
        sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type;
    sycl::float2* res_fl2_ptr = reinterpret_cast<sycl::float2*>(residual);
    const sycl::float2* hs_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(hidden_state);
    const sycl::float2* attn_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(attn);
    const sycl::float2* bias_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(bias);
    const sycl::float2* attn_bias_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(attn_bias);
    const int offset = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);

    if (offset < total_count) {
      sycl::float2 res_fl2 = res_fl2_ptr[offset];
      const sycl::float2 hs_fl2 = hs_fl2_ptr[offset];
      const sycl::float2 attn_fl2 = attn_fl2_ptr[offset];
      const sycl::float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];
      const sycl::float2 attn_bias_fl2 =
          attn_bias_fl2_ptr[offset % intermediate_size];

      T2* res_half2 = reinterpret_cast<T2*>(&res_fl2);
      const T2* hs_half2 = reinterpret_cast<const T2*>(&hs_fl2);
      const T2* attn_half2 = reinterpret_cast<const T2*>(&attn_fl2);
      const T2* bias_half2 = reinterpret_cast<const T2*>(&bias_fl2);
      const T2* attn_bias_half2 = reinterpret_cast<const T2*>(&attn_bias_fl2);

      sycl::float2 res_low = conversion::to<sycl::float2>(res_half2[0]);
      sycl::float2 res_high = conversion::to<sycl::float2>(res_half2[1]);

      const sycl::float2 hs_low = conversion::to<sycl::float2>(hs_half2[0]);
      const sycl::float2 hs_high = conversion::to<sycl::float2>(hs_half2[1]);

      const sycl::float2 attn_low = conversion::to<sycl::float2>(attn_half2[0]);
      const sycl::float2 attn_high =
          conversion::to<sycl::float2>(attn_half2[1]);

      const sycl::float2 bias_low = conversion::to<sycl::float2>(bias_half2[0]);
      const sycl::float2 bias_high =
          conversion::to<sycl::float2>(bias_half2[1]);

      const sycl::float2 attn_bias_low =
          conversion::to<sycl::float2>(attn_bias_half2[0]);
      const sycl::float2 attn_bias_high =
          conversion::to<sycl::float2>(attn_bias_half2[1]);

      if (preln) {
        // residual = (residual + attention + bias + attention_bias) *
        // mp_scale + hidden_state
        res_low.x() =
            (res_low.x() + attn_low.x() + bias_low.x() + attn_bias_low.x()) *
                mp_scale +
            hs_low.x();
        res_low.y() =
            (res_low.y() + attn_low.y() + bias_low.y() + attn_bias_low.y()) *
                mp_scale +
            hs_low.y();
        res_high.x() = (res_high.x() + attn_high.x() + bias_high.x() +
                        attn_bias_high.x()) *
                mp_scale +
            hs_high.x();
        res_high.y() = (res_high.y() + attn_high.y() + bias_high.y() +
                        attn_bias_high.y()) *
                mp_scale +
            hs_high.y();
      } else {
        // residual += hidden_state + bias
        res_low.x() = (res_low.x() + hs_low.x() + bias_low.x());
        res_low.y() = (res_low.y() + hs_low.y() + bias_low.y());
        res_high.x() = (res_high.x() + hs_high.x() + bias_high.x());
        res_high.y() = (res_high.y() + hs_high.y() + bias_high.y());
      }
      res_half2[0] = conversion::to<T2>(res_low);
      res_half2[1] = conversion::to<T2>(res_high);

      res_fl2_ptr[offset] = res_fl2;
    }
  }
};

template <>
class fused_bias_residual<float> {
 private:
  float* residual;
  const float* hidden_state;
  const float* attn;
  const float* bias;
  const float* attn_bias;
  const int total_count;
  const int intermediate_size;
  const float mp_scale;
  const bool preln;

 public:
  fused_bias_residual(
      float* residual,
      const float* hidden_state,
      const float* attn,
      const float* bias,
      const float* attn_bias,
      const int total_count,
      const int intermediate_size,
      const float mp_scale,
      const bool preln)
      : residual(residual),
        hidden_state(hidden_state),
        attn(attn),
        bias(bias),
        attn_bias(attn_bias),
        total_count(total_count),
        intermediate_size(intermediate_size),
        mp_scale(mp_scale),
        preln(preln) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::float4* res_fl4_ptr = reinterpret_cast<sycl::float4*>(residual);
    const sycl::float4* hs_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(hidden_state);
    const sycl::float4* attn_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(attn);
    const sycl::float4* bias_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(bias);
    const sycl::float4* attn_bias_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(attn_bias);
    const int offset = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);

    if (offset < total_count) {
      sycl::float4 res_fl4 = res_fl4_ptr[offset];
      const sycl::float4 hs_fl4 = hs_fl4_ptr[offset];
      const sycl::float4 attn_fl4 = attn_fl4_ptr[offset];
      const sycl::float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];
      const sycl::float4 attn_bias_fl4 =
          attn_bias_fl4_ptr[offset % intermediate_size];
      if (preln) {
        // residual = (residual + attention + bias + attention_bias) *
        // mp_scale + hidden_state
        res_fl4.x() =
            (res_fl4.x() + attn_fl4.x() + bias_fl4.x() + attn_bias_fl4.x()) *
                mp_scale +
            (hs_fl4.x());
        res_fl4.y() =
            (res_fl4.y() + attn_fl4.y() + bias_fl4.y() + attn_bias_fl4.y()) *
                mp_scale +
            (hs_fl4.y());
        res_fl4.z() =
            (res_fl4.z() + attn_fl4.z() + bias_fl4.z() + attn_bias_fl4.z()) *
                mp_scale +
            (hs_fl4.z());
        res_fl4.w() =
            (res_fl4.w() + attn_fl4.w() + bias_fl4.w() + attn_bias_fl4.w()) *
                mp_scale +
            (hs_fl4.w());
      } else {
        // residual += hidden_state + bias
        res_fl4.x() = res_fl4.x() + hs_fl4.x() + bias_fl4.x();
        res_fl4.y() = res_fl4.y() + hs_fl4.y() + bias_fl4.y();
        res_fl4.z() = res_fl4.z() + hs_fl4.z() + bias_fl4.z();
        res_fl4.w() = res_fl4.w() + hs_fl4.w() + bias_fl4.w();
      }
      res_fl4_ptr[offset] = res_fl4;
    }
  }
};

template <typename T>
void launch_bias_residual(
    T* residual,
    T* hidden_state,
    T* attn,
    T* bias,
    T* attn_bias,
    int batch,
    int hidden_dim,
    int mp_size,
    bool preln,
    dpct::queue_ptr stream) {
  int total_count = batch * hidden_dim / 4;
  sycl::range<3> block_dims(1, 1, 1024);
  sycl::range<3> grid_dims(1, 1, (total_count - 1) / 1024 + 1); // (batch_size);

  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    fused_bias_residual fn(
        residual,
        hidden_state,
        attn,
        bias,
        attn_bias,
        total_count,
        hidden_dim / 4,
        1.0 / mp_size,
        preln);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims), fn);
  }
}

#define INSTANTIATE_LAUNCH_BIAS_RESIDUAL(T) \
  template void launch_bias_residual<T>(    \
      T*, T*, T*, T*, T*, int, int, int, bool, dpct::queue_ptr);

INSTANTIATE_LAUNCH_BIAS_RESIDUAL(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_RESIDUAL(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_BIAS_RESIDUAL(sycl::half);

template <typename T>
class gptj_residual_add {
 private:
  T* residual;
  const T* hidden_state;
  const T* attn;
  const T* bias;
  const T* attn_bias;
  const int total_count;
  const int intermediate_size;
  const float mp_scale;

 public:
  gptj_residual_add(
      T* residual,
      const T* hidden_state,
      const T* attn,
      const T* bias,
      const T* attn_bias,
      const int total_count,
      const int intermediate_size,
      const float mp_scale)
      : residual(residual),
        hidden_state(hidden_state),
        attn(attn),
        bias(bias),
        attn_bias(attn_bias),
        total_count(total_count),
        intermediate_size(intermediate_size),
        mp_scale(mp_scale) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    using T2 = typename std::conditional<
        std::is_same<T, sycl::half>::value,
        sycl::half2,
        sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type;
    sycl::float2* res_fl2_ptr = reinterpret_cast<sycl::float2*>(residual);
    const sycl::float2* hs_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(hidden_state);
    const sycl::float2* attn_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(attn);
    const sycl::float2* bias_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(bias);
    const sycl::float2* attn_bias_fl2_ptr =
        reinterpret_cast<const sycl::float2*>(attn_bias);
    const int offset = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);

    if (offset < total_count) {
      sycl::float2 res_fl2 = res_fl2_ptr[offset];
      const sycl::float2 hs_fl2 = hs_fl2_ptr[offset];
      const sycl::float2 attn_fl2 = attn_fl2_ptr[offset];
      const sycl::float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];

      T2* res_half2 = reinterpret_cast<T2*>(&res_fl2);
      const T2* hs_half2 = reinterpret_cast<const T2*>(&hs_fl2);
      const T2* attn_half2 = reinterpret_cast<const T2*>(&attn_fl2);
      const T2* bias_half2 = reinterpret_cast<const T2*>(&bias_fl2);

      sycl::float2 res_low = conversion::to<sycl::float2>(res_half2[0]);
      sycl::float2 res_high = conversion::to<sycl::float2>(res_half2[1]);

      const sycl::float2 hs_low = conversion::to<sycl::float2>(hs_half2[0]);
      const sycl::float2 hs_high = conversion::to<sycl::float2>(hs_half2[1]);

      const sycl::float2 attn_low = conversion::to<sycl::float2>(attn_half2[0]);
      const sycl::float2 attn_high =
          conversion::to<sycl::float2>(attn_half2[1]);

      const sycl::float2 bias_low = conversion::to<sycl::float2>(bias_half2[0]);
      const sycl::float2 bias_high =
          conversion::to<sycl::float2>(bias_half2[1]);

      if (attn_bias) {
        const sycl::float2 attn_bias_fl2 =
            attn_bias_fl2_ptr[offset % intermediate_size];
        const T2* attn_bias_half2 = reinterpret_cast<const T2*>(&attn_bias_fl2);
        const sycl::float2 attn_bias_low =
            conversion::to<sycl::float2>(attn_bias_half2[0]);
        const sycl::float2 attn_bias_high =
            conversion::to<sycl::float2>(attn_bias_half2[1]);
        // residual += attention_bias
        res_low.x() += attn_bias_low.x();
        res_low.y() += attn_bias_low.y();
        res_high.x() += attn_bias_high.x();
        res_high.y() += attn_bias_high.y();
      }
      // residual = hidden_state + attention + (residual + bias) * mp_scale
      res_low.x() =
          attn_low.x() + hs_low.x() + (res_low.x() + bias_low.x()) * mp_scale;
      res_low.y() =
          attn_low.y() + hs_low.y() + (res_low.y() + bias_low.y()) * mp_scale;
      res_high.x() = attn_high.x() + hs_high.x() +
          (res_high.x() + bias_high.x()) * mp_scale;
      res_high.y() = attn_high.y() + hs_high.y() +
          (res_high.y() + bias_high.y()) * mp_scale;

      res_half2[0] = conversion::to<T2>(res_low);
      res_half2[1] = conversion::to<T2>(res_high);

      res_fl2_ptr[offset] = res_fl2;
    }
  }
};

template <>
class gptj_residual_add<float> {
 private:
  float* residual;
  const float* hidden_state;
  const float* attn;
  const float* bias;
  const float* attn_bias;
  const int total_count;
  const int intermediate_size;
  const float mp_scale;

 public:
  gptj_residual_add(
      float* residual,
      const float* hidden_state,
      const float* attn,
      const float* bias,
      const float* attn_bias,
      const int total_count,
      const int intermediate_size,
      const float mp_scale)
      : residual(residual),
        hidden_state(hidden_state),
        attn(attn),
        bias(bias),
        attn_bias(attn_bias),
        total_count(total_count),
        intermediate_size(intermediate_size),
        mp_scale(mp_scale) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::float4* res_fl4_ptr = reinterpret_cast<sycl::float4*>(residual);
    const sycl::float4* hs_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(hidden_state);
    const sycl::float4* attn_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(attn);
    const sycl::float4* bias_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(bias);
    const sycl::float4* attn_bias_fl4_ptr =
        reinterpret_cast<const sycl::float4*>(attn_bias);
    const int offset = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);

    if (offset < total_count) {
      sycl::float4 res_fl4 = res_fl4_ptr[offset];
      const sycl::float4 hs_fl4 = hs_fl4_ptr[offset];
      const sycl::float4 attn_fl4 = attn_fl4_ptr[offset];
      const sycl::float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];

      if (attn_bias) {
        sycl::float4 attn_bias_fl4 =
            attn_bias_fl4_ptr[offset % intermediate_size];
        // residual += attention_bias
        res_fl4.x() += attn_bias_fl4.x();
        res_fl4.y() += attn_bias_fl4.y();
        res_fl4.z() += attn_bias_fl4.z();
        res_fl4.w() += attn_bias_fl4.w();
      }
      // residual = hidden_state + attention + (residual + bias) * mp_scale
      res_fl4.x() =
          hs_fl4.x() + attn_fl4.x() + (res_fl4.x() + bias_fl4.x()) * mp_scale;
      res_fl4.y() =
          hs_fl4.y() + attn_fl4.y() + (res_fl4.y() + bias_fl4.y()) * mp_scale;
      res_fl4.z() =
          hs_fl4.z() + attn_fl4.z() + (res_fl4.z() + bias_fl4.z()) * mp_scale;
      res_fl4.w() =
          hs_fl4.w() + attn_fl4.w() + (res_fl4.w() + bias_fl4.w()) * mp_scale;

      res_fl4_ptr[offset] = res_fl4;
    }
  }
};

template <typename T>
void launch_gptj_residual_add(
    T* residual,
    T* hidden_state,
    T* attn,
    T* bias,
    T* attn_bias,
    int hidden_dim,
    int batch,
    int mp_size,
    dpct::queue_ptr stream) {
  int total_count = batch * hidden_dim / 4;
  sycl::range<3> block_dims(1, 1, 1024);
  sycl::range<3> grid_dims(1, 1, (total_count - 1) / 1024 + 1); // (batch_size);

  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    gptj_residual_add fn(
        residual,
        hidden_state,
        attn,
        bias,
        attn_bias,
        total_count,
        hidden_dim / 4,
        1.0 / mp_size);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims), fn);
  }
}

#define INSTANTIATE_GPT_RES_ADD(T)           \
  template void launch_gptj_residual_add<T>( \
      T*, T*, T*, T*, T*, int, int, int, dpct::queue_ptr);

INSTANTIATE_GPT_RES_ADD(float);
INSTANTIATE_GPT_RES_ADD(sycl::half);
#ifdef BF16_AVAILABLE
INSTANTIATE_GPT_RES_ADD(sycl::ext::oneapi::bfloat16);
#endif

template <typename T>
class moe_res_matmul {
 private:
  T* residual;
  T* coef;
  T* mlp_out;
  int seq_len;
  int hidden_dim;

 public:
  moe_res_matmul(T* residual, T* coef, T* mlp_out, int seq_len, int hidden_dim)
      : residual(residual),
        coef(coef),
        mlp_out(mlp_out),
        seq_len(seq_len),
        hidden_dim(hidden_dim) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(T);

    T* residual_seq = residual + item_ct1.get_group(2) * hidden_dim;
    T* mlp_out_seq = mlp_out + item_ct1.get_group(2) * hidden_dim;

    for (unsigned tid = item_ct1.get_local_id(2) * vals_per_access;
         tid < hidden_dim;
         tid += item_ct1.get_local_range(2) * vals_per_access) {
      T mlp[vals_per_access];
      T res[vals_per_access];
      T coef1[vals_per_access];
      T coef2[vals_per_access];

      mem_access::load_global<granularity>(mlp, mlp_out_seq + tid);
      mem_access::load_global<granularity>(res, residual_seq + tid);
      mem_access::load_global<granularity>(coef1, coef + tid);
      mem_access::load_global<granularity>(coef2, coef + tid + hidden_dim);

#pragma unroll
      for (int idx = 0; idx < vals_per_access; idx++) {
        mlp[idx] = mlp[idx] * coef2[idx] + res[idx] * coef1[idx];
      }

      mem_access::store_global<granularity>(mlp_out_seq + tid, mlp);
    }
  }
};

template <typename T>
void launch_moe_res_matmul(
    T* residual,
    T* coef,
    T* mlp_out,
    int seq_len,
    int hidden_dim,
    dpct::queue_ptr stream) {
  sycl::range<3> grid_dim(1, 1, seq_len);
  sycl::range<3> block_dim(1, 1, 1024);
  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
    moe_res_matmul fn(residual, coef, mlp_out, seq_len, hidden_dim);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
  }
}

#define INSTANTIATE_LAUNCH_MOE_RES_MATMUL(T) \
  template void launch_moe_res_matmul<T>(T*, T*, T*, int, int, dpct::queue_ptr);

INSTANTIATE_LAUNCH_MOE_RES_MATMUL(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_MOE_RES_MATMUL(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_MOE_RES_MATMUL(sycl::half);

template <typename T>
class pad_data_kernel {
 private:
  T* padded_output;
  T* output;
  int head_size;
  int padded_head_size;

 public:
  pad_data_kernel(
      T* padded_output,
      T* output,
      int head_size,
      int padded_head_size)
      : padded_output(padded_output),
        output(output),
        head_size(head_size),
        padded_head_size(padded_head_size) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    using T2 = typename std::conditional<
        std::is_same<T, sycl::half>::value,
        sycl::half2,
        sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type;
    sycl::float4* padded_output_cast =
        reinterpret_cast<sycl::float4*>(padded_output);
    sycl::float4* output_cast = reinterpret_cast<sycl::float4*>(output);
    int bid = item_ct1.get_group(2) * (item_ct1.get_local_range(1)) +
        item_ct1.get_local_id(1);
    int idx = item_ct1.get_local_id(2);
    padded_output_cast += (bid * padded_head_size);
    output_cast += (bid * head_size);
    sycl::float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++)
      ZERO_h[i] = zero_h;
    if (idx < head_size)
      padded_output_cast[idx] = output_cast[idx];
    else
      padded_output_cast[idx] = ZERO;
  }
};

/* void pad_data_kernel(float* padded_output, */
/*                                 float* output, */
/*                                 int head_size, */
/*                                 int padded_head_size) */
/* { */
/* } */

template <typename T>
void pad_data(
    T* padded_output,
    T* output,
    int bsz,
    int head_size,
    int padded_head_size,
    dpct::queue_ptr stream) {
  sycl::range<3> grid_dim(1, 1, (bsz - 1) / 16 + 1);
  sycl::range<3> block_dim(1, 16, padded_head_size / 8);
  /*
  DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    pad_data_kernel fn(
        padded_output, output, head_size / 8, padded_head_size / 8);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
  }
}

#define INSTANTIATE_PAD_DATA(T) \
  template void pad_data(T*, T*, int, int, int, dpct::queue_ptr stream);

INSTANTIATE_PAD_DATA(float);
INSTANTIATE_PAD_DATA(sycl::half);
#ifdef BF16_AVAILABLE
INSTANTIATE_PAD_DATA(sycl::ext::oneapi::bfloat16);
#endif

template <typename T>
class pad_head_seq_kernel {
 private:
  T* padded_output;
  T* output;
  int seq_len;
  int padded_seq_len;
  int head_size;
  int padded_head_size;

 public:
  pad_head_seq_kernel(
      T* padded_output,
      T* output,
      int seq_len,
      int padded_seq_len,
      int head_size,
      int padded_head_size)
      : padded_output(padded_output),
        output(output),
        seq_len(seq_len),
        padded_seq_len(padded_seq_len),
        head_size(head_size),
        padded_head_size(padded_head_size) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    using T2 = typename std::conditional<
        std::is_same<T, sycl::half>::value,
        sycl::half2,
        sycl::marray<sycl::ext::oneapi::bfloat16, 2>>::type;
    sycl::float4* padded_output_cast =
        reinterpret_cast<sycl::float4*>(padded_output);
    sycl::float4* output_cast = reinterpret_cast<sycl::float4*>(output);
    int bsz = item_ct1.get_group(2);
    int bid = item_ct1.get_group(1) * (item_ct1.get_local_range(1)) +
        item_ct1.get_local_id(1);
    int idx = item_ct1.get_local_id(2);
    padded_output_cast += (bsz * padded_seq_len + bid) * padded_head_size;
    output_cast += (bsz * seq_len + bid) * head_size;
    sycl::float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++)
      ZERO_h[i] = zero_h;

    if (idx < head_size && bid < seq_len)
      padded_output_cast[idx] = output_cast[idx];
    else
      padded_output_cast[idx] = ZERO;
  }
};

/* void pad_head_seq_kernel(float* padded_output, */
/*                                     float* output, */
/*                                     int seq_len, */
/*                                     int padded_seq_len, */
/*                                     int head_size, */
/*                                     int padded_head_size) */
/* { */
/* } */

template <typename T>
void pad_head_seq(
    T* padded_output,
    T* output,
    int bsz,
    int seq_len,
    int padded_seq_len,
    int head_size,
    int padded_head_size,
    dpct::queue_ptr stream) {
  sycl::range<3> grid_dim(1, padded_seq_len / 16, bsz);
  sycl::range<3> block_dim(1, 16, padded_head_size / 8);
  /*
  DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    pad_head_seq_kernel fn(
        padded_output,
        output,
        seq_len,
        padded_seq_len,
        head_size / 8,
        padded_head_size / 8);
    stream->parallel_for(
        sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
  }
}

#define INSTANTIATE_PAD_HEAD_SEQ(T) \
  template void pad_head_seq<T>(    \
      T*, T*, int, int, int, int, int, dpct::queue_ptr);

INSTANTIATE_PAD_HEAD_SEQ(sycl::half);
#ifdef BF16_AVAILABLE
INSTANTIATE_PAD_HEAD_SEQ(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_PAD_HEAD_SEQ(float);

// TODO(cmikeh2): evaluate different GeLU performance
__dpct_inline__ float old_gelu(float val) {
  // 1 / sqrt(2)
  constexpr float rsqrt_2 = 0.707106769084930419922;
  return val * 0.5f * (1.0f + sycl::erf(val * rsqrt_2));
}

namespace fused_geglu {
constexpr int threads = 256;
constexpr int steps = 2;
constexpr int granularity = 16;
} // namespace fused_geglu

__dpct_inline__ float silu(float val) {
  return val / (1.0f + sycl::native::exp(-val));
}

template <typename T, bool useGelu>
class fused_gate_activation {
 private:
  T* output;
  const T* activation;
  const T* bias;
  int base_channels;
  int output_stride;
  int total_elems;

 public:
  fused_gate_activation(
      T* output,
      const T* activation,
      const T* bias,
      int base_channels,
      int output_stride,
      int total_elems)
      : output(output),
        activation(activation),
        bias(bias),
        base_channels(base_channels),
        output_stride(output_stride),
        total_elems(total_elems) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int id = item_ct1.get_group(2) * T_per_block +
        item_ct1.get_local_id(2) * T_per_access;

#pragma unroll
    for (int i = 0; i < fused_geglu::steps; i++) {
      T activation_buffer_1[T_per_access];
      T activation_buffer_2[T_per_access];
      T bias_buffer_1[T_per_access];
      T bias_buffer_2[T_per_access];

      const int iter_id = id + T_per_step * i;
      if (iter_id < total_elems) {
        const int channel_id = iter_id % base_channels;
        const int seq_id = iter_id / base_channels;
        const int seq_offset = seq_id * base_channels * 2;

        mem_access::load_global<fused_geglu::granularity>(
            activation_buffer_1, activation + seq_offset + channel_id);
        mem_access::load_global<fused_geglu::granularity>(
            activation_buffer_2,
            activation + seq_offset + channel_id + base_channels);
        mem_access::load_global<fused_geglu::granularity>(
            bias_buffer_1, bias + channel_id, bias != nullptr);
        mem_access::load_global<fused_geglu::granularity>(
            bias_buffer_2, bias + channel_id + base_channels, bias != nullptr);

        // Since the GeLU is going to happen at float, might as well
        // convert
#pragma unroll
        for (int v = 0; v < T_per_access; v++) {
          T hidden_state = activation_buffer_1[v] + bias_buffer_1[v];
          T pre_gate = activation_buffer_2[v] + bias_buffer_2[v];
          float pre_gate_f = conversion::to<float>(pre_gate);
          float gate_f = (useGelu) ? old_gelu(pre_gate_f) : silu(pre_gate_f);
          T gate = conversion::to<T>(gate_f);
          activation_buffer_1[v] = hidden_state * gate;
        }

        mem_access::store_global<fused_geglu::granularity>(
            output + seq_id * output_stride + channel_id, activation_buffer_1);
      }
    }
  }
};

template <typename T>
void launch_gated_activation(
    T* output,
    const T* activation,
    const T* bias,
    int rows,
    int output_stride,
    int elems_per_row,
    bool use_gelu,
    dpct::queue_ptr stream) {
  /*
  Fused bias GEGLU is a variant of the gated activation functions.
  The input here is a matrix of [batch, seq_len, 2 * intermediate_dim]
  where the second half of the channels act as GeLU gates for the first
  half.
  */

  // Re-derive the above figures
  constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
  constexpr int T_per_step = T_per_access * fused_geglu::threads;
  constexpr int T_per_block = T_per_step * fused_geglu::steps;

  const int base_channels = elems_per_row / 2;
  const int total_elems = base_channels * rows;

  sycl::range<3> block(1, 1, fused_geglu::threads);
  sycl::range<3> grid(1, 1, (total_elems + T_per_block - 1) / T_per_block);

  if (use_gelu) {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    fused_gate_activation<T, true> fn(
        output, activation, bias, base_channels, output_stride, total_elems);
    stream->parallel_for(sycl::nd_range<3>(grid * block, block), fn);
  } else {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    fused_gate_activation<T, false> fn(
        output, activation, bias, base_channels, output_stride, total_elems);
    stream->parallel_for(sycl::nd_range<3>(grid * block, block), fn);
  }
}

#define INSTANTIATE_LAUNCH_GATED_ACTIVATION(T) \
  template void launch_gated_activation(       \
      T*, const T*, const T*, int, int, int, bool, dpct::queue_ptr);

INSTANTIATE_LAUNCH_GATED_ACTIVATION(sycl::half);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_GATED_ACTIVATION(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_GATED_ACTIVATION(float);
