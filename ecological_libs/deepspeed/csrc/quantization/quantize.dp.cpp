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

#include <sycl/sycl.hpp>
#include <dpct/dpct.h>
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"

/*
Pure quantization kernel with no fusion.
*/
template <int q_bits,
          quantize::Type quant_type,
          int UNROLL,
          int internal_unroll,
          int threads_per_group,
          int max_threads>
class cached_quantization {
private:
  int8_t* __restrict__ output_data;
  float* __restrict__ params;
  const sycl::half* __restrict__ input_data;
  int groups;
  int elems_per_group;
public:
  cached_quantization(int8_t* __restrict__ output_data,
                      float* __restrict__ params,
                      const sycl::half* __restrict__ input_data,
                      int groups,
                      int elems_per_group): output_data(output_data),
                                            params(params),
                                            input_data(input_data),
                                            groups(groups),
                                            elems_per_group(elems_per_group) {}
  void operator()(sycl::nd_item<3>) const {
    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    // Indexing offsets
    const int block_offset =
        (tb.get_group_id()[2] * (max_threads / threads_per_group) * elems_per_group) +
        (tb.get_local_id()[1] * elems_per_group);
    const int elem_offset = tb.get_local_id()[2] * quantize::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = sycl::ext::oneapi::experimental::this_group<3>().get_local_linear_range() *
                       quantize::h_per_load;

    const sycl::half* input_base = input_data + base_offset;  //..

    sycl::half2 local_buffer[UNROLL * internal_unroll * quantize::h2_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // Convenience helper, should resolve to register indices and not realize.
        sycl::half2* iteration_buffer = local_buffer + i * internal_unroll * quantize::h2_per_load;
#pragma unroll
        for (int j = 0; j < internal_unroll; j++) {
            const int iteration = i * internal_unroll + j;
            mem_access::load_global<quantize::granularity>(
                iteration_buffer + j * quantize::h2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
        }
    }

    quantize::
        local_array<quant_type, q_bits, UNROLL * internal_unroll, threads_per_group, max_threads>(
            local_buffer, params, output_data, elems_per_group, groups);
  }
};


/********* Launcher methods ***********/
/*
DPCT1049:47: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define LAUNCH_CACHED_QUANT_CALL(q_bits, quant_type)                                            \
 dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});  \
 cached_quantization<q_bits,                                                                    \
                     quant_type,                                                                \
                     unroll_factor,                                                             \
                     internal_unroll_l,                                                         \
                     threads_per_group,                                                         \
                     max_threads> fn(output_data, params, input_data, groups, elems_per_group); \ 
 stream->submit([&](sycl::handler& cgh) {                                                       \
  cgh.parallel_for(                                                                             \
      sycl::nd_range<3>(grid * block, block), fn);                                              \
 });

#define LAUNCH_CACHED_QUANT(                                                        \
    q_bits, quant_type, unroll_factor_in, internal_unroll_in, threads_per_group_in) \
    const int unroll_factor = unroll_factor_in;                                     \
    const int internal_unroll_l = internal_unroll_in;                               \
    const int threads_per_group = threads_per_group_in;                             \
    if (q_bits == 4) {                                                              \
        if (quant_type == quantize::Type::Asymmetric) {                             \
            LAUNCH_CACHED_QUANT_CALL(4, quantize::Type::Asymmetric)                 \
        } else {                                                                    \
            LAUNCH_CACHED_QUANT_CALL(4, quantize::Type::Symmetric)                  \
        }                                                                           \
    } else {                                                                        \
        if (quant_type == quantize::Type::Asymmetric) {                             \
            LAUNCH_CACHED_QUANT_CALL(8, quantize::Type::Asymmetric)                 \
        } else {                                                                    \
            LAUNCH_CACHED_QUANT_CALL(8, quantize::Type::Symmetric)                  \
        }                                                                           \
    }

void launch_quant(int8_t* output_data,
                  float* params,
                  const sycl::half* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  dpct::queue_ptr stream)
{
    constexpr int max_threads = 256;

    constexpr int internal_unroll = 2;

    const bool is_subblock_schedule = (elems_per_group <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? quantize::h_per_load
                                                : quantize::h_per_load * internal_unroll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_group + h_per_step - 1) / h_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;

    const int groups_per_block =
        is_subblock_schedule ? (max_threads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_launch = (groups_per_block + groups - 1) / groups_per_block;

    sycl::range<3> block(1, groups_per_block, threads_per_group);
    sycl::range<3> grid(1, 1, groups_launch);

    const int elems_per_step = threads_per_group * h_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threads_per_group == 1) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 1);
        } else if (threads_per_group == 2) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 2);
        } else if (threads_per_group == 4) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 4);
        } else if (threads_per_group == 8) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 8);
        } else if (threads_per_group == 16) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 16);
        }
    } else if (external_unroll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, internal_unroll, max_threads);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 2, internal_unroll, max_threads);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 3, internal_unroll, max_threads);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 4, internal_unroll, max_threads);
    }
}
