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
#include "reduction_utils.h"

using rop = reduce::ROpType;

namespace rms {
constexpr int granularity = 16;
} // namespace rms

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
class rms_norm {
 private:
  T* output;
  const T* vals;
  const T* gamma;
  float epsilon;
  int elems_per_row;

 public:
  rms_norm(
      T* output,
      const T* vals,
      const T* gamma,
      float epsilon,
      int elems_per_row)
      : output(output),
        vals(vals),
        gamma(gamma),
        epsilon(epsilon),
        elems_per_row(elems_per_row) {}
  void operator()(sycl::nd_item<3>) const {
    constexpr int T_per_load = rms::granularity / sizeof(T);

    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    // X-dimension of the block
    const int block_offset = (tb.get_group_id()[2] *
                              (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id()[1] * elems_per_row);
    const int thread_offset = tb.get_local_id()[2] * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride =
        sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_range(2) *
        T_per_load;

    float var_sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);

      mem_access::load_global<rms::granularity>(
          iteration_buffer,
          input_base + (i * stride),
          thread_offset + (i * stride) < elems_per_row);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        float up_cast = conversion::to<float>(iteration_buffer[j]);
        float sq_val = up_cast * up_cast;
        var_sum = reduce::element<rop::Add, float>(var_sum, sq_val);
      }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float var = var_sum / elems_per_row;
    /*
    DPCT1013:8: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    const T denom = conversion::to<T>(sycl::rsqrt(var + epsilon));

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);
      const int iter_idx = i * stride + thread_offset;
      const bool do_loads = (iter_idx < elems_per_row);

      T gamma_local[T_per_load];

      mem_access::load_global<rms::granularity>(
          gamma_local, gamma + iter_idx, do_loads);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        iteration_buffer[j] *= denom;
        iteration_buffer[j] *= gamma_local[j];
      }

      if (do_loads) {
        mem_access::store_global<rms::granularity>(
            block_output + iter_idx, iteration_buffer);
      }
    }
  }
};

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
class pre_rms_norm {
 private:
  T* output;
  T* res_out;
  const T* vals;
  const T* residual;
  const T* gamma;
  float epsilon;
  int elems_per_row;

 public:
  pre_rms_norm(
      T* output,
      T* res_out,
      const T* vals,
      const T* residual,
      const T* gamma,
      float epsilon,
      int elems_per_row)
      : output(output),
        res_out(res_out),
        vals(vals),
        residual(residual),
        gamma(gamma),
        epsilon(epsilon),
        elems_per_row(elems_per_row) {}
  void operator()(sycl::nd_item<3>) const {
    constexpr int T_per_load = rms::granularity / sizeof(T);

    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    // X-dimension of the block
    const int block_offset = (tb.get_group_id()[2] *
                              (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id()[1] * elems_per_row);
    const int thread_offset = tb.get_local_id()[2] * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride =
        sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_range(2) *
        T_per_load;

    float var_sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;
    T* res_output = res_out + base_offset;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);
      T residual_buffer[T_per_load];

      const int iter_offset = i * stride + thread_offset;
      const bool do_loads = (iter_offset < elems_per_row);

      mem_access::load_global<rms::granularity>(
          iteration_buffer, input_base + (i * stride), do_loads);
      mem_access::load_global<rms::granularity>(
          residual_buffer, residual_base + (i * stride), do_loads);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        iteration_buffer[j] += residual_buffer[j];
        float vals_up_cast = conversion::to<float>(iteration_buffer[j]);

        var_sum = reduce::element<rop::Add, float>(
            var_sum, vals_up_cast * vals_up_cast);
      }

      if (do_loads) {
        mem_access::store_global<rms::granularity>(
            res_output + i * stride, iteration_buffer);
      }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float var = var_sum / elems_per_row;
    /*
    DPCT1013:9: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    const T denom = conversion::to<T>(sycl::rsqrt(var + epsilon));

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);
      const int iter_idx = i * stride + thread_offset;
      const bool do_loads = (iter_idx < elems_per_row);

      T gamma_local[T_per_load];

      mem_access::load_global<rms::granularity>(
          gamma_local, gamma + iter_idx, do_loads);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        iteration_buffer[j] *= denom;
        iteration_buffer[j] *= gamma_local[j];
      }

      if (do_loads) {
        mem_access::store_global<rms::granularity>(
            block_output + iter_idx, iteration_buffer);
      }
    }
  }
};

#define LAUNCH_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)           \
  dpct::has_capability_or_fail(                                        \
      stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
  stream->submit([&](sycl::handler& cgh) {                             \
    rms_norm<T, UNROLL, threadsPerGroup, maxThreads> fn(               \
        norm_output, vals, gamma, epsilon, elems_per_row);             \
                                                                       \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);      \
  });

/*
DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define LAUNCH_PRE_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)       \
  dpct::has_capability_or_fail(                                        \
      stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
  stream->submit([&](sycl::handler& cgh) {                             \
    pre_rms_norm<T, UNROLL, threadsPerGroup, maxThreads> fn(           \
        norm_output,                                                   \
        res_output,                                                    \
        vals,                                                          \
        residual,                                                      \
        gamma,                                                         \
        epsilon,                                                       \
        elems_per_row);                                                \
                                                                       \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);      \
  });

#define LAUNCH_ALL_RMS_NORM(UNROLL, threadsPerGroup, maxThreads) \
  if (pre_norm) {                                                \
    LAUNCH_PRE_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)     \
  } else {                                                       \
    LAUNCH_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)         \
  }

template <typename T>
void launch_rms_norm(
    T* norm_output,
    T* res_output,
    const T* vals,
    const T* residual,
    const T* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    dpct::queue_ptr stream) {
  // 8 for sycl::half, 4 for float
  constexpr int T_per_load = rms::granularity / sizeof(T);
  constexpr int maxThreads = 256;
  constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;

  const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
  const int h_per_step =
      is_subblock_schedule ? T_per_load : T_per_load * internalUnroll;

  // Scheduling concern: may be slightly faster for some inputs to assign
  // multiple stages of warp-sized blocks rather than stepping up to 64/96
  // threads
  const int one_step_threads =
      next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
  const int threads_per_group =
      (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

  const int groups_per_block_max = is_subblock_schedule
      ? (maxThreads + threads_per_group - 1) / threads_per_group
      : 1;
  const int groups_per_block =
      (rows < groups_per_block_max) ? rows : groups_per_block_max;
  const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

  sycl::range<3> block(1, groups_per_block, threads_per_group);
  sycl::range<3> grid(1, 1, groups_launch);

  const int elems_per_step = threads_per_group * h_per_step;
  const int external_unRoll =
      (elems_per_row + elems_per_step - 1) / elems_per_step;

  bool pre_norm = (residual == nullptr) ? false : true;

  if (is_subblock_schedule) {
    // <=128
    if (threads_per_group == 1) {
      LAUNCH_ALL_RMS_NORM(1, 1, maxThreads);
    } else if (threads_per_group == 2) {
      LAUNCH_ALL_RMS_NORM(1, 2, maxThreads);
    } else if (threads_per_group == 4) {
      LAUNCH_ALL_RMS_NORM(1, 4, maxThreads);
    } else if (threads_per_group == 8) {
      LAUNCH_ALL_RMS_NORM(1, 8, maxThreads);
    } else if (threads_per_group == 16) {
      LAUNCH_ALL_RMS_NORM(1, 16, maxThreads);
    }
  } else if (external_unRoll == 1) {
    // 129 - 4096 elems
    // (this can launch with 1-7 warps as well)
    LAUNCH_ALL_RMS_NORM(1 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unRoll == 2) {
    // 4097 - 8192 elems
    LAUNCH_ALL_RMS_NORM(2 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unRoll == 3) {
    // 8193 - 12288 elems
    LAUNCH_ALL_RMS_NORM(3 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unRoll == 4) {
    // 12289 - 16384 elems
    LAUNCH_ALL_RMS_NORM(4 * internalUnroll, maxThreads, maxThreads);
  }
}

#define INSTANTIATE_LAUNCH_RMS_NORM(T) \
  template void launch_rms_norm<T>(    \
      T * norm_output,                 \
      T * res_output,                  \
      const T* vals,                   \
      const T* residual,               \
      const T* gamma,                  \
      float epsilon,                   \
      int rows,                        \
      int elems_per_row,               \
      dpct::queue_ptr stream);

INSTANTIATE_LAUNCH_RMS_NORM(float)
INSTANTIATE_LAUNCH_RMS_NORM(sycl::half)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_RMS_NORM(sycl::ext::oneapi::bfloat16)
#endif
