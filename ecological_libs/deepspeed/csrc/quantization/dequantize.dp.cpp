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
#include "dequantization_utils.h"
#include "memory_access_utils.h"

template <typename T, int numBits, dequantize::Type qType, int unroll, int threads>
class dequantize_kernel {
private:
  T* __restrict__ dequant_data;
  const int8_t* __restrict__ q_data;
  const float* __restrict__ q_params;
  int elems_per_group;
  int total_elems;

public:
  dequantize_kernel(T* __restrict__ dequant_data, 
                    const int8_t* __restrict__ q_data, 
                    const float* __restrict__ q_params, 
                    int elems_per_group, 
                    int total_elems): dequant_data(dequant_data), 
                                      q_data(q_data), 
                                      q_params(q_params), 
                                      elems_per_group(elems_per_group), 
                                      total_elems(total_elems) {}
  void operator()(sycl::nd_item<3>) const
  {
      dequantize::to_global<T, numBits, qType, unroll, threads>(
          dequant_data, q_data, q_params, elems_per_group, total_elems);
  }
};

/*
DPCT1049:47: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define LAUNCH_DEQUANT_KERNEL(num_bits, q_type)                                                    \
  {                                                                                                \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});  \
    dequantize_kernel<T, num_bits, q_type, unroll, threads>                                        \
                                fn(dequant_data, q_data, q_params, elems_per_group, total_elems);  \
    stream->submit([&](sycl::handler& cgh) {                                                       \
      cgh.parallel_for(                                                                            \
          sycl::nd_range<3>(grid * block, block),                                                  \
          fn);                                                                                     \
    });                                                                                            \
  }

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int total_elems,
                              dpct::queue_ptr stream)
{
    constexpr int unroll = 8;
    constexpr int threads = 512;
    constexpr int elems_per_block = unroll * threads * dequantize::granularity / (sizeof(T));

    const sycl::range<3> block(1, 1, threads);
    const sycl::range<3> grid(1, 1, (total_elems + elems_per_block - 1) / elems_per_block);

    // TODO(cmikeh2): It may make sense to tune unroll, there is perf benefit for large
    // problem sizes with this large unroll value.
    if (num_bits == 8 && q_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_KERNEL(8, quantize::Type::Symmetric);
    } else if (num_bits == 8 && q_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_KERNEL(8, quantize::Type::Asymmetric);
    } else if (num_bits == 4 && q_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_KERNEL(4, quantize::Type::Symmetric);
    } else if (num_bits == 4 && q_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_KERNEL(4, quantize::Type::Asymmetric);
    }
}

template void launch_dequantize_kernel(sycl::half* dequant_data,
                                       const int8_t* q_data,
                                       const float* q_params,
                                       quantize::Type q_type,
                                       int num_bits,
                                       int elems_per_group,
                                       int total_elems,
                                       dpct::queue_ptr stream);

template void launch_dequantize_kernel(float* dequant_data,
                                       const int8_t* q_data,
                                       const float* q_params,
                                       quantize::Type q_type,
                                       int num_bits,
                                       int elems_per_group,
                                       int total_elems,
                                       dpct::queue_ptr stream);
