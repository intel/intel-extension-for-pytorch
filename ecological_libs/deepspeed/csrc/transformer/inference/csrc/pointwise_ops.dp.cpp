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
#include "memory_access_utils.h"

namespace pwise {
constexpr int granularity = 16;
constexpr int unroll = 4;
constexpr int threads = 256;
} // namespace pwise

template <typename T>
class vector_add_kernel {
 private:
  T* out;
  const T* a;
  const T* b;
  float gamma;
  int num_elems;

 public:
  vector_add_kernel(T* out, const T* a, const T* b, float gamma, int num_elems)
      : out(out), a(a), b(b), gamma(gamma), num_elems(num_elems) {}
  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int T_per_access = pwise::granularity / sizeof(T);

    const int block_offset =
        item_ct1.get_group(2) * pwise::threads * pwise::unroll * T_per_access;
    const int thread_offset = item_ct1.get_local_id(2) * T_per_access;
    const int total_offset = block_offset + thread_offset;
    constexpr int stride = pwise::threads * T_per_access;

#pragma unroll
    for (int i = 0; i < pwise::unroll; i++) {
      T temp_buf_a[T_per_access], temp_buf_b[T_per_access];

      const int iter_idx = total_offset + i * stride;

      mem_access::load_global<pwise::granularity>(
          temp_buf_a, a + iter_idx, iter_idx < num_elems);
      mem_access::load_global<pwise::granularity>(
          temp_buf_b, b + iter_idx, iter_idx < num_elems);

#pragma unroll
      for (int j = 0; j < T_per_access; j++) {
        float up_cast_a = conversion::to<float>(temp_buf_a[j]);
        float up_cast_b = conversion::to<float>(temp_buf_b[j]);
        temp_buf_a[j] = conversion::to<T>((gamma * up_cast_a) + up_cast_b);
      }

      if (iter_idx < num_elems) {
        mem_access::store_global<pwise::granularity>(
            out + iter_idx, temp_buf_a);
      }
    }
  }
};

template <typename T>
void launch_vector_add(
    T* out,
    const T* a,
    const T* b,
    float gamma,
    int num_elems,
    dpct::queue_ptr stream) {
  constexpr int T_per_access = pwise::granularity / sizeof(T);
  constexpr int T_per_block = pwise::threads * T_per_access * pwise::unroll;

  sycl::range<3> block(1, 1, pwise::threads);
  sycl::range<3> grid(1, 1, (num_elems + T_per_block - 1) / T_per_block);

  {
    dpct::has_capability_or_fail(
        stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});
    stream->parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          vector_add_kernel(out, a, b, gamma, num_elems);
        });
  }
}

#define INSTANTIATE_VECTOR_ADD(T)     \
  template void launch_vector_add<T>( \
      T * out,                        \
      const T* a,                     \
      const T* b,                     \
      float gamma,                    \
      int num_elems,                  \
      dpct::queue_ptr stream);

INSTANTIATE_VECTOR_ADD(float)
INSTANTIATE_VECTOR_ADD(sycl::half)
#ifdef BF16_AVAILABLE
INSTANTIATE_VECTOR_ADD(sycl::ext::oneapi::bfloat16)
#endif
