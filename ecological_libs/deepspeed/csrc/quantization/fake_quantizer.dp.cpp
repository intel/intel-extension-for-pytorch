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
#include <math.h>
#include "custom_sycl_layers.h"
#include "memory_access_utils.h"
#include "conversion_utils.h"

template<typename T>
class fake_quantize_kernel {};

template<>
class fake_quantize_kernel<sycl::half> {
private:
  sycl::half* vals;
  int group_size;
  int num_bits;
public:
  fake_quantize_kernel(sycl::half* vals, int group_size, int num_bits): vals(vals), group_size(group_size), num_bits(num_bits) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    /* int gid = threadIdx.x >> 5; */
    /* int lane = threadIdx.x & 0x1f; */
    /* int warp_num = blockDim.x >> 5; */
    /* int id = threadIdx.x; */

    auto gid = item_ct1.get_local_id(2) >> 5;
    auto lane = item_ct1.get_local_id(2) & 0x1f;
    auto warp_num = item_ct1.get_local_range(2) >> 5;
    auto id = item_ct1.get_local_id(2);

    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(sycl::half);

    sycl::half data[vals_per_access];

    /* int group_id = blockIdx.x; */
    auto group_id = item_ct1.get_group(2);

    int thread_index = id * vals_per_access;
    int reg_count = 0;
    int offset = group_id * group_size;
    float max = -10000.0;
    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += item_ct1.get_local_range(2) * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);

#pragma unroll
        for (int i = 0; i < vals_per_access; i++) {
            if (abs((float)data[i]) > max) max = abs((float)data[i]);
        }
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shuffle_xor(max, i);
        if (max < temp) max = temp;
    }
    /* __shared__ float partialMax[WARP_SIZE]; */
    auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
        sycl::ext::oneapi::experimental::this_group<3>());

    if (lane == 0) partialMax[gid] = max;

    /* b.sync(); */
    item_ct1.barrier();

    if (lane < warp_num) max = partialMax[lane];

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shuffle_down(max, i);
        if (max < temp) max = temp;
    }

    max = g.shuffle(max, 0);

    float q_scale = (float)(1 << num_bits) / (2 * max + 1e-5);
    float q_scale_inv = 1 / q_scale;
    int q_range_max = (1 << (num_bits - 1)) - 1;
    int q_range_min = -(1 << (num_bits - 1));

    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += item_ct1.get_local_range(2) * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);
#pragma unroll
        for (int j = 0; j < vals_per_access; j++) {
            float q_data;
            /* q_data = sycl::half2float(data[j]); */
            q_data = conversion::to<float>(data[j]);
            /* q_data = __float2int_rn(q_data * q_scale); */
            q_data = sycl::vec<float, 1>{(q_data * q_scale)}
                         .convert<int, sycl::rounding_mode::rte>()[0];
            q_data = q_data > (q_range_max) ? (q_range_max)
                                            : (q_data < (q_range_min) ? (q_range_min) : q_data);
            /* data[j] = __float2half_rn(q_data * q_scale_inv); */
            data[j] = conversion::to<sycl::half>(q_data * q_scale_inv);
        }
        mem_access::store_global<granularity>(vals + offset + thread_index, data);
    }
  }

};

template<>
class fake_quantize_kernel<float> {
private:
  float* vals;
  int group_size;
  int num_bits;
public:
  fake_quantize_kernel(float* vals, int group_size, int num_bits): vals(vals), group_size(group_size), num_bits(num_bits) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    auto g = sycl::ext::oneapi::experimental::this_sub_group();

    int gid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;
    int id = item_ct1.get_local_id(2);

    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(float);

    float data[vals_per_access];

    int bid = item_ct1.get_group(2);

    int thread_index = id * vals_per_access;

    int reg_count = 0;

    int offset = bid * group_size;

    float max = -10000.0;

    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += item_ct1.get_local_range(2) * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);

#pragma unroll
        for (int i = 0; i < vals_per_access; i++) {
            if (sycl::fabs(data[i]) > max) max = sycl::fabs(data[i]);
        }
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shuffle_xor(max, i);
        if (max < temp) max = temp;
    }
    auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
        sycl::ext::oneapi::experimental::this_group<3>());

    if (lane == 0) partialMax[gid] = max;

    /*
    DPCT1065:42: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    if (lane < warp_num) max = partialMax[lane];

    /*
    DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shuffle_down(max, i);
        if (max < temp) max = temp;
    }

    max = g.shuffle(max, 0);

    float q_scale = (1 << num_bits) / (2 * max + 1e-5);
    float q_scale_inv = 1 / q_scale;

    int q_range_max = (1 << (num_bits - 1)) - 1;
    int q_range_min = -(1 << (num_bits - 1));

    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += item_ct1.get_local_range(2) * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);
#pragma unroll
        for (int j = 0; j < vals_per_access; j++) {
            float q_data;
            q_data = sycl::vec<float, 1>{(data[j] * q_scale)}
                         .convert<int, sycl::rounding_mode::rte>()[0];
            q_data = q_data > (q_range_max) ? (q_range_max)
                                            : (q_data < (q_range_min) ? (q_range_min) : q_data);
            data[j] = sycl::round(q_data * q_scale_inv);
        }
        mem_access::store_global<granularity>(vals + offset + thread_index, data);
    }
}
};

template <typename T>
void launch_fake_quantize_kernel(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 dpct::queue_ptr stream)
{
    sycl::range<3> grid_dim(1, 1, group_num);
    sycl::range<3> block_dim(1, 1, 1024);

   dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
   fake_quantize_kernel<T> fn(vals, total_count / group_num, num_bits);
   stream->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
    
}

template void launch_fake_quantize_kernel(float* vals,
                                          int total_count,
                                          int group_num,
                                          int num_bits,
                                          dpct::queue_ptr stream);
template void launch_fake_quantize_kernel(sycl::half* vals,
                                          int total_count,
                                          int group_num,
                                          int num_bits,
                                          dpct::queue_ptr stream);

template<typename T>
class sr_fake_quantize_kernel {};

template<>
class sr_fake_quantize_kernel<sycl::half> {
private:
  sycl::half* vals;
  int token_size;
  int token_num;
  int num_bits;
  std::pair<uint64_t, uint64_t> seed;
public:
  sr_fake_quantize_kernel(sycl::half* vals,
                          int token_size,
                          int token_num,
                          int num_bits,
                          std::pair<uint64_t, uint64_t> seed): vals(vals),
                                                               token_size(token_size),
                                                               token_num(token_num),
                                                               num_bits(num_bits),
                                                               seed(seed) {}

  void operator()(sycl::nd_item<3>) const {

    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    /* int gid = threadIdx.x >> 5; */
    auto gid = item_ct1.get_local_id(2) >> 5;
    /* int lane = threadIdx.x & 0x1f; */
    auto lane = item_ct1.get_local_id(2) & 0x1f;
    /* int warp_num = blockDim.x >> 5; */
    auto warp_num = item_ct1.get_local_range(2) >> 5;

    /* int idx = blockIdx.x * blockDim.x + threadIdx.x; */
    auto idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    sycl::float2* vals_cast = reinterpret_cast<sycl::float2*>(vals);

    sycl::half2 data_low[128];
    sycl::half2 data_high[128];

    /* int bid = blockIdx.x; */
    auto bid = item_ct1.get_group(2);

    /* curandStatePhilox4_32_10_t state; */
    /* curand_init(seed.first, idx, seed.second, &state); */
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> state;
    state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(seed.first, {seed.second, (unsigned long)idx * 4});
    
    /* unsigned int tid = threadIdx.x; */
    auto tid = item_ct1.get_local_id(2);
    int reg_count = 0;
    int offset = bid * token_size;
    int group_index = bid * token_size + tid;

    int total_count = token_size * token_num;
    if (group_index < total_count) {
        // float min = 10000.0;
        float max = -10000.0;
        while (tid < token_size) {
            sycl::float2 data = vals_cast[offset + tid];
            sycl::half2* data_h = reinterpret_cast<sycl::half2*>(&data);
            data_low[reg_count] = data_h[0];
            data_high[reg_count] = data_h[1];

            sycl::float2 data_f[2];
            /* data_f[0] = sycl::half22float2(data_h[0]); */
            data_f[0] = conversion::to<sycl::float2>(data_h[0]);
            /* data_f[1] = sycl::half22float2(data_h[1]); */
            data_f[1] = conversion::to<sycl::float2>(data_h[1]);

            if (abs((float)data_f[0].x()) > max) max = abs((float)data_f[0].x());
            if (abs((float)data_f[0].y()) > max) max = abs((float)data_f[0].y());
            if (abs((float)data_f[1].x()) > max) max = abs((float)data_f[1].x());
            if (abs((float)data_f[1].y()) > max) max = abs((float)data_f[1].y());

            /* tid += blockDim.x; */
            tid += item_ct1.get_local_range(2);
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(max, i);
            if (max < temp) max = temp;
        }

        /* __shared__ float partialMax[WARP_SIZE]; */
        auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());

        if (lane == 0) partialMax[gid] = max;

        /* b.sync(); */
        item_ct1.barrier();

        if (lane < warp_num) max = partialMax[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(max, i);
            if (max < temp) max = temp;
        }

        max = g.shuffle(max, 0);

        float q_scale_val = (float)(1 << num_bits) / (max * 2 + 1e-5);
        float high_q = (float)((1 << (num_bits - 1)) - 1);
        float low_q = (float)(-((1 << (num_bits - 1))));

        for (int i = 0; i < reg_count; i++) {
            /* int token_index = i * blockDim.x + threadIdx.x; */
            int token_index = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
            if (token_index < token_size) {
                sycl::float2 data_f[2];
                /* data_f[0] = sycl::half22float2(data_low[i]); */
                data_f[0] = conversion::to<sycl::float2>(data_low[i]);
                /* data_f[1] = sycl::half22float2(data_high[i]); */
                data_f[1] = conversion::to<sycl::float2>(data_high[i]);

                sycl::float2 q_data_int[2];
                q_data_int[0].x() = (float)((int)(data_f[0].x() * q_scale_val));
                q_data_int[0].y() = (float)((int)(data_f[0].y() * q_scale_val));
                q_data_int[1].x() = (float)((int)(data_f[1].x() * q_scale_val));
                q_data_int[1].y() = (float)((int)(data_f[1].y() * q_scale_val));

                // Stochastic rounding
                sycl::float4 rand = state.generate<oneapi::mkl::rng::device::uniform<float>, 4>();

                float q_error[4];
                q_error[0] = abs(data_f[0].x() - (q_data_int[0].x() / q_scale_val)) * q_scale_val;
                q_error[1] = abs(data_f[0].y() - (q_data_int[0].y() / q_scale_val)) * q_scale_val;
                q_error[2] = abs(data_f[1].x() - (q_data_int[1].x() / q_scale_val)) * q_scale_val;
                q_error[3] = abs(data_f[1].y() - (q_data_int[1].y() / q_scale_val)) * q_scale_val;

                q_data_int[0].x() =
                    (rand.x() < q_error[0] && q_data_int[0].x() > low_q && q_data_int[0].x() < high_q)
                        ? (q_data_int[0].x() + (data_f[0].x() > 0 ? 1 : -1))
                        : q_data_int[0].x();
                q_data_int[0].y() =
                    (rand.y() < q_error[1] && q_data_int[0].y() > low_q && q_data_int[0].y() < high_q)
                        ? (q_data_int[0].y() + (data_f[0].y() > 0 ? 1 : -1))
                        : q_data_int[0].y();
                q_data_int[1].x() =
                    (rand.w() < q_error[2] && q_data_int[1].x() > low_q && q_data_int[1].x() < high_q)
                        ? (q_data_int[1].x() + (data_f[1].x() > 0 ? 1 : -1))
                        : q_data_int[1].x();
                q_data_int[1].y() =
                    (rand.z() < q_error[3] && q_data_int[1].y() > low_q && q_data_int[1].y() < high_q)
                        ? (q_data_int[1].y() + (data_f[1].y() > 0 ? 1 : -1))
                        : q_data_int[1].y();

                data_f[0].x() = q_data_int[0].x() / q_scale_val;
                data_f[0].y() = q_data_int[0].y() / q_scale_val;
                data_f[1].x() = q_data_int[1].x() / q_scale_val;
                data_f[1].y() = q_data_int[1].y() / q_scale_val;

                sycl::float2 result;
                sycl::half2* result_h = reinterpret_cast<sycl::half2*>(&result);
                /* result_h[0] = __float22half2_rn(data_f[0]); */
                result_h[0] = conversion::to<sycl::half2>(data_f[0]);
                /* result_h[1] = __float22half2_rn(data_f[1]); */
                result_h[1] = conversion::to<sycl::half2>(data_f[1]);

                vals_cast[offset + token_index] = result;
            }
        }
    }
  }
};

template<>
class sr_fake_quantize_kernel<float> {
private:
  float* vals;
  int token_size;
  int token_num;
  int num_bits;
  std::pair<uint64_t, uint64_t> seed;
public:
  sr_fake_quantize_kernel(float* vals,
                          int token_size,
                          int token_num,
                          int num_bits,
                          std::pair<uint64_t, uint64_t> seed): vals(vals),
                                                               token_size(token_size),
                                                               token_num(token_num),
                                                               num_bits(num_bits),
                                                               seed(seed) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    int gid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;
    int id = item_ct1.get_local_id(2);

    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + id;

    sycl::float4* vals_cast = reinterpret_cast<sycl::float4*>(vals);

    sycl::float4 data[128];

    int bid = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> state;
    /* curand_init(seed.first, idx, seed.second, &state); */
    state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(seed.first, {(unsigned long)idx, seed.second * 4});

    int group_index = bid * token_size + item_ct1.get_local_id(2);
    int reg_count = 0;
    int total_count = token_size * token_num;
    if (group_index < total_count) {
        // float min = 10000.0;
        float max = -10000.0;

        while (tid < token_size) {
            data[reg_count] = vals_cast[group_index];

            if (sycl::fabs(data[reg_count].x()) > max) max = sycl::fabs(data[reg_count].x());
            if (sycl::fabs(data[reg_count].y()) > max) max = sycl::fabs(data[reg_count].y());
            if (sycl::fabs(data[reg_count].z()) > max) max = sycl::fabs(data[reg_count].z());
            if (sycl::fabs(data[reg_count].w()) > max) max = sycl::fabs(data[reg_count].w());

            group_index += item_ct1.get_local_range(2);
            tid += item_ct1.get_local_range(2);
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(max, i);
            if (max < temp) max = temp;
        }
        auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());

        if (lane == 0) partialMax[gid] = max;

        /*
        DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if
        there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane < warp_num) max = partialMax[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(max, i);
            if (max < temp) max = temp;
        }

        max = g.shuffle(max, 0);

        float q_scale_val = (float)(1 << num_bits) / (max * 2 + 1e-5);
        float high_q = (float)((1 << (num_bits - 1)) - 1);
        float low_q = (float)(-((1 << (num_bits - 1))));

        int offset = (bid)*token_size;
        for (int i = 0; i < reg_count; i++) {
            group_index = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
            if (group_index < token_size) {
                sycl::float4 q_data = data[i];

                sycl::float4 q_data_int;
                q_data_int.x() = (float)((int)(q_data.x() * q_scale_val));
                q_data_int.y() = (float)((int)(q_data.y() * q_scale_val));
                q_data_int.w() = (float)((int)(q_data.w() * q_scale_val));
                q_data_int.z() = (float)((int)(q_data.z() * q_scale_val));

                // Stochastic rounding
                sycl::float4 rand = state.generate<oneapi::mkl::rng::device::uniform<float>, 4>();

                float q_error[4];
                q_error[0] = sycl::fabs(q_data.x() - (q_data_int.x() / q_scale_val)) * q_scale_val;
                q_error[1] = sycl::fabs(q_data.y() - (q_data_int.y() / q_scale_val)) * q_scale_val;
                q_error[2] = sycl::fabs(q_data.w() - (q_data_int.w() / q_scale_val)) * q_scale_val;
                q_error[3] = sycl::fabs(q_data.z() - (q_data_int.z() / q_scale_val)) * q_scale_val;

                q_data_int.x() =
                    (rand.x() < q_error[0] && q_data_int.x() > low_q && q_data_int.x() < high_q)
                        ? (q_data_int.x() + (q_data.x() > 0 ? 1 : -1))
                        : q_data_int.x();
                q_data_int.y() =
                    (rand.y() < q_error[1] && q_data_int.y() > low_q && q_data_int.y() < high_q)
                        ? (q_data_int.y() + (q_data.y() > 0 ? 1 : -1))
                        : q_data_int.y();
                q_data_int.w() =
                    (rand.w() < q_error[2] && q_data_int.w() > low_q && q_data_int.w() < high_q)
                        ? (q_data_int.w() + (q_data.w() > 0 ? 1 : -1))
                        : q_data_int.w();
                q_data_int.z() =
                    (rand.z() < q_error[3] && q_data_int.z() > low_q && q_data_int.z() < high_q)
                        ? (q_data_int.z() + (q_data.z() > 0 ? 1 : -1))
                        : q_data_int.z();

                q_data_int.x() /= q_scale_val;
                q_data_int.y() /= q_scale_val;
                q_data_int.w() /= q_scale_val;
                q_data_int.z() /= q_scale_val;

                vals_cast[group_index + offset] = q_data_int;
            }
        }
    }
  }
};


template <typename T>
void launch_sr_fake_quantize_kernel(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    dpct::queue_ptr stream)
{
    sycl::range<3> block_dim(1, 1, 1024);
    sycl::range<3> grid_dim(1, 1, group_num);

    uint64_t inc = total_count / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = TrainingContext::Instance().IncrementOffset(inc);

    /*
    DPCT1049:46: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        sr_fake_quantize_kernel<T> fn(vals, (total_count / group_num) / 4, group_num, num_bits, seed);
        stream->parallel_for(
            sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
    }
}
template void launch_sr_fake_quantize_kernel(float* vals,
                                             int total_count,
                                             int group_num,
                                             int num_bits,
                                             dpct::queue_ptr stream);
template void launch_sr_fake_quantize_kernel(sycl::half* vals,
                                             int total_count,
                                             int group_num,
                                             int num_bits,
                                             dpct::queue_ptr stream);
template<typename T>
class fake_quantize_kernel_asym {};

template<>
class fake_quantize_kernel_asym<sycl::half> {
private:
  sycl::half* vals;
  int group_size;
  int num_bits;
public:
  fake_quantize_kernel_asym(sycl::half* vals, int group_size, int num_bits): vals(vals), 
                                                                             group_size(group_size), 
                                                                             num_bits(num_bits) {}

  void operator()(sycl::nd_item<3>) const {

    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    /* int gid = threadIdx.x >> 5; */
    auto gid = item_ct1.get_local_id(2) >> 5;
    /* int lane = threadIdx.x & 0x1f; */
    auto lane = item_ct1.get_local_id(2) & 0x1f;
    /* int warp_num = blockDim.x >> 5; */
    auto warp_num = item_ct1.get_local_range(2) >> 5;
    /* int id = threadIdx.x; */
    auto id = item_ct1.get_local_id(2);

    sycl::float2* vals_cast = reinterpret_cast<sycl::float2*>(vals);

    sycl::float2 data[MAX_REG];

    /* int group_id = blockIdx.x; */
    auto group_id = item_ct1.get_group(2);

    {
        int group_index = id;
        int reg_count = 0;
        int offset = group_id * group_size;
        float max = -10000.0;
        float min = 10000.0;

        while (group_index < group_size && reg_count < MAX_REG) {
            data[reg_count] = vals_cast[offset + group_index];
            sycl::half* data_h = reinterpret_cast<sycl::half*>(&data[reg_count]);

            if (((float)data_h[0]) > max) max = (float)data_h[0];
            if (((float)data_h[1]) > max) max = (float)data_h[1];
            if (((float)data_h[2]) > max) max = (float)data_h[2];
            if (((float)data_h[3]) > max) max = (float)data_h[3];

            if (((float)data_h[0]) < min) min = (float)data_h[0];
            if (((float)data_h[1]) < min) min = (float)data_h[1];
            if (((float)data_h[2]) < min) min = (float)data_h[2];
            if (((float)data_h[3]) < min) min = (float)data_h[3];

            /* group_index += blockDim.x; */
            group_index += item_ct1.get_local_range(2);
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(max, i);
            if (max < temp) max = temp;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(min, i);
            if (min > temp) min = temp;
        }

        /* __shared__ float partialMax[WARP_SIZE]; */
        auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());
        /* __shared__ float partialMin[WARP_SIZE]; */
        auto& partialMin = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());

        if (lane == 0) partialMax[gid] = max;
        if (lane == 0) partialMin[gid] = min;

        /* b.sync(); */
        item_ct1.barrier();

        if (lane < warp_num) max = partialMax[lane];
        if (lane < warp_num) min = partialMin[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(max, i);
            if (max < temp) max = temp;
        }
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(min, i);
            if (min > temp) min = temp;
        }

        max = g.shuffle(max, 0);
        min = g.shuffle(min, 0);

        float q_scale = ((max - min) + 1e-5) / (float)(1 << num_bits);
        float q_scale_inv = 1 / q_scale;

        for (int i = 0; i < reg_count; i++) {
            /* group_index = i * blockDim.x + id; */
            group_index = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
            if (group_index < group_size) {
                sycl::half2* data_h = reinterpret_cast<sycl::half2*>(&data[i]);
                sycl::float2 q_data[2];
                /* q_data[0] = sycl::half22float2(data_h[0]); */
                q_data[0] = conversion::to<sycl::float2>(data_h[0]);
                /* q_data[1] = sycl::half22float2(data_h[1]); */
                q_data[1] = conversion::to<sycl::float2>(data_h[1]);

                sycl::float2 q_data_int[2];

                q_data_int[0].x() = roundf((q_data[0].x() - min) * q_scale_inv);
                q_data_int[0].y() = roundf((q_data[0].y() - min) * q_scale_inv);
                q_data_int[1].x() = roundf((q_data[1].x() - min) * q_scale_inv);
                q_data_int[1].y() = roundf((q_data[1].y() - min) * q_scale_inv);

                q_data_int[0].x() = q_data_int[0].x() * q_scale + min;
                q_data_int[0].y() = q_data_int[0].y() * q_scale + min;
                q_data_int[1].x() = q_data_int[1].x() * q_scale + min;
                q_data_int[1].y() = q_data_int[1].y() * q_scale + min;

                /* data_h[0] = __float22half2_rn(q_data_int[0]); */
                data_h[0] = conversion::to<sycl::half2>(q_data_int[0]);
                /* data_h[1] = __float22half2_rn(q_data_int[1]); */
                data_h[1] = conversion::to<sycl::half2>(q_data_int[1]);

                vals_cast[offset + group_index] = data[i];
            }
        }
    }
  }
};

template<>
class fake_quantize_kernel_asym<float> {
private:
  float* vals;
  int group_size;
  int num_bits;
public:
  fake_quantize_kernel_asym(float* vals, int group_size, int num_bits): vals(vals), 
                                                                             group_size(group_size), 
                                                                             num_bits(num_bits) {}

  void operator()(sycl::nd_item<3>) const {
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    int gid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;
    int id = item_ct1.get_local_id(2);

    sycl::float4* vals_cast = reinterpret_cast<sycl::float4*>(vals);

    sycl::float4 data[MAX_REG];

    int bid = item_ct1.get_group(2);

    int group_index = bid * group_size + id;
    int reg_count = 0;

    float max = -10000.0;
    float min = 10000.0;

    while (id < group_size && reg_count < MAX_REG) {
        sycl::float4 data_reg = vals_cast[group_index];
        data[reg_count] = data_reg;

        if (data_reg.x() > max) max = data_reg.x();
        if (data_reg.y() > max) max = data_reg.y();
        if (data_reg.w() > max) max = data_reg.w();
        if (data_reg.z() > max) max = data_reg.z();

        if (data_reg.x() < min) min = data_reg.x();
        if (data_reg.y() < min) min = data_reg.y();
        if (data_reg.w() < min) min = data_reg.w();
        if (data_reg.z() < min) min = data_reg.z();

        group_index += item_ct1.get_local_range(2);
        id += item_ct1.get_local_range(2);
        reg_count++;
    }
    id = item_ct1.get_local_id(2);

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shuffle_xor(max, i);
        if (max < temp) max = temp;
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shuffle_xor(min, i);
        if (min > temp) min = temp;
    }

    auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
        sycl::ext::oneapi::experimental::this_group<3>());
    auto& partialMin = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
        sycl::ext::oneapi::experimental::this_group<3>());

    if (lane == 0) partialMax[gid] = max;
    if (lane == 0) partialMin[gid] = min;

    /*
    DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    if (lane < warp_num) max = partialMax[lane];
    if (lane < warp_num) min = partialMin[lane];

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shuffle_down(max, i);
        if (max < temp) max = temp;
    }
#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shuffle_down(min, i);
        if (min > temp) min = temp;
    }

    max = g.shuffle(max, 0);
    min = g.shuffle(min, 0);

    float q_scale = ((max - min) + 1e-5) / (float)(1 << num_bits);
    float q_scale_inv = 1 / q_scale;
    for (int i = 0; i < reg_count; i++) {
        group_index = i * item_ct1.get_local_range(2) + id;
        if (group_index < group_size) {
            sycl::float4 q_data;
            q_data = data[i];

            sycl::float4 q_data_int;
            q_data_int.x() = sycl::round((q_data.x() - min) * q_scale_inv);
            q_data_int.y() = sycl::round((q_data.y() - min) * q_scale_inv);
            q_data_int.w() = sycl::round((q_data.w() - min) * q_scale_inv);
            q_data_int.z() = sycl::round((q_data.z() - min) * q_scale_inv);

            q_data.x() = q_data_int.x() * q_scale + min;
            q_data.y() = q_data_int.y() * q_scale + min;
            q_data.w() = q_data_int.w() * q_scale + min;
            q_data.z() = q_data_int.z() * q_scale + min;

            vals_cast[group_index + bid * group_size] = q_data;
        }
    }
  }
};

template <typename T>
void launch_fake_quantize_kernel_asym(T* vals,
                                      int total_count,
                                      int group_num,
                                      int num_bits,
                                      dpct::queue_ptr stream)
{
    sycl::range<3> grid_dim(1, 1, group_num);
    sycl::range<3> block_dim(1, 1, 1024);

    /*
    DPCT1049:48: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        fake_quantize_kernel_asym<T> fn(vals, (total_count / group_num) / 4, num_bits);
        stream->parallel_for(
            sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
    }
}

template void launch_fake_quantize_kernel_asym(float* vals,
                                               int total_count,
                                               int group_num,
                                               int num_bits,
                                               dpct::queue_ptr stream);
template void launch_fake_quantize_kernel_asym(sycl::half* vals,
                                               int total_count,
                                               int group_num,
                                               int num_bits,
                                               dpct::queue_ptr stream);

void sr_fake_quantize_kernel_asym(sycl::half* vals,
                                  int token_size,
                                  int token_num,
                                  int num_bits,
                                  std::pair<uint64_t, uint64_t> seed)
{

    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    /* int gid = threadIdx.x >> 5; */
    auto gid = item_ct1.get_local_id(2) >> 5;
    /* int lane = threadIdx.x & 0x1f; */
    auto lane = item_ct1.get_local_id(2) & 0x1f;
    /* int warp_num = blockDim.x >> 5; */
    auto warp_num = item_ct1.get_local_range(2) >> 5;

    /* int idx = blockIdx.x * blockDim.x + threadIdx.x; */
    auto idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    sycl::float2* vals_cast = reinterpret_cast<sycl::float2*>(vals);

    sycl::half2 data_low[128];
    sycl::half2 data_high[128];

    /* int bid = blockIdx.x; */
    auto bid = item_ct1.get_group(2);

    /* curandStatePhilox4_32_10_t state; */
    /* curand_init(seed.first, idx, seed.second, &state); */
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> state;
    state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(seed.first, {seed.second, (unsigned long)idx * 4});
    /* unsigned int tid = threadIdx.x; */
    auto tid = item_ct1.get_local_id(2);
    int reg_count = 0;
    int offset = bid * token_size;
    int group_index = bid * token_size + tid;

    int total_count = token_size * token_num;
    if (group_index < total_count) {
        float min = 10000.0;
        float max = -10000.0;
        while (tid < token_size) {
            sycl::float2 data = vals_cast[offset + tid];
            sycl::half2* data_h = reinterpret_cast<sycl::half2*>(&data);
            data_low[reg_count] = data_h[0];
            data_high[reg_count] = data_h[1];

            sycl::float2 data_f[2];
            /* data_f[0] = sycl::half22float2(data_h[0]); */
            data_f[0] = conversion::to<sycl::float2>(data_h[0]);
            /* data_f[1] = sycl::half22float2(data_h[1]); */
            data_f[1] = conversion::to<sycl::float2>(data_h[1]);

            if (((float)data_f[0].x()) > max) max = (float)data_f[0].x();
            if (((float)data_f[0].y()) > max) max = (float)data_f[0].y();
            if (((float)data_f[1].x()) > max) max = (float)data_f[1].x();
            if (((float)data_f[1].y()) > max) max = (float)data_f[1].y();

            if (((float)data_f[0].x()) < min) min = (float)data_f[0].x();
            if (((float)data_f[0].y()) < min) min = (float)data_f[0].y();
            if (((float)data_f[1].x()) < min) min = (float)data_f[1].x();
            if (((float)data_f[1].y()) < min) min = (float)data_f[1].y();

            /* tid += blockDim.x; */
            tid += item_ct1.get_local_range(2);
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(max, i);
            if (max < temp) max = temp;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(min, i);
            if (min > temp) min = temp;
        }

        /* __shared__ float partialMax[WARP_SIZE]; */
        auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());
        /* __shared__ float partialMin[WARP_SIZE]; */
        auto& partialMin = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());

        if (lane == 0) partialMax[gid] = max;
        if (lane == 0) partialMin[gid] = min;

        /* b.sync(); */
        item_ct1.barrier();

        if (lane < warp_num) max = partialMax[lane];
        if (lane < warp_num) min = partialMin[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(max, i);
            if (max < temp) max = temp;
        }
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(min, i);
            if (min > temp) min = temp;
        }

        max = g.shuffle(max, 0);
        min = g.shuffle(min, 0);

        float q_scale_val = ((max - min) + 1e-5) / (float)(1 << num_bits);
        float q_scale_val_inv = 1 / q_scale_val;
        float high_q = (float)((1 << num_bits) - 1);

        for (int i = 0; i < reg_count; i++) {
            /* int token_index = i * blockDim.x + threadIdx.x; */
            int token_index = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
            if (token_index < token_size) {
                sycl::float2 data_f[2];
                /* data_f[0] = sycl::half22float2(data_low[i]); */
                data_f[0] = conversion::to<sycl::float2>(data_low[i]);
                /* data_f[1] = sycl::half22float2(data_high[i]); */
                data_f[1] = conversion::to<sycl::float2>(data_high[i]);

                sycl::float2 q_data_int[2];
                q_data_int[0].x() = (float)((unsigned int)((data_f[0].x() - min) * q_scale_val_inv));
                q_data_int[0].y() = (float)((unsigned int)((data_f[0].y() - min) * q_scale_val_inv));
                q_data_int[1].x() = (float)((unsigned int)((data_f[1].x() - min) * q_scale_val_inv));
                q_data_int[1].y() = (float)((unsigned int)((data_f[1].y() - min) * q_scale_val_inv));

                // Stochastic rounding
                /* float4 rand = curand_uniform4(&state); */
                sycl::float4 rand = state.generate<oneapi::mkl::rng::device::uniform<float>, 4>();

                float q_error[4];
                q_error[0] =
                    abs(data_f[0].x() - ((q_data_int[0].x() * q_scale_val) + min)) * q_scale_val_inv;
                q_error[1] =
                    abs(data_f[0].y() - ((q_data_int[0].y() * q_scale_val) + min)) * q_scale_val_inv;
                q_error[2] =
                    abs(data_f[1].x() - ((q_data_int[1].x() * q_scale_val) + min)) * q_scale_val_inv;
                q_error[3] =
                    abs(data_f[1].y() - ((q_data_int[1].y() * q_scale_val) + min)) * q_scale_val_inv;

                q_data_int[0].x() = (rand.x() < q_error[0] && q_data_int[0].x() < high_q)
                                      ? (q_data_int[0].x() + 1)
                                      : q_data_int[0].x();
                q_data_int[0].y() = (rand.y() < q_error[1] && q_data_int[0].y() < high_q)
                                      ? (q_data_int[0].y() + 1)
                                      : q_data_int[0].y();
                q_data_int[1].x() = (rand.w() < q_error[2] && q_data_int[1].x() < high_q)
                                      ? (q_data_int[1].x() + 1)
                                      : q_data_int[1].x();
                q_data_int[1].y() = (rand.z() < q_error[3] && q_data_int[1].y() < high_q)
                                      ? (q_data_int[1].y() + 1)
                                      : q_data_int[1].y();

                data_f[0].x() = q_data_int[0].x() * q_scale_val + min;
                data_f[0].y() = q_data_int[0].y() * q_scale_val + min;
                data_f[1].x() = q_data_int[1].x() * q_scale_val + min;
                data_f[1].y() = q_data_int[1].y() * q_scale_val + min;

                sycl::float2 result;
                sycl::half2* result_h = reinterpret_cast<sycl::half2*>(&result);
                /* result_h[0] = __float22half2_rn(data_f[0]); */
                result_h[0] = conversion::to<sycl::half2>(data_f[0]);
                /* result_h[1] = __float22half2_rn(data_f[1]); */
                result_h[1] = conversion::to<sycl::half2>(data_f[1]);

                vals_cast[offset + token_index] = result;
            }
        }
    }
}

void sr_fake_quantize_kernel_asym(float* vals,
                                             int token_size,
                                             int token_num,
                                             int num_bits,
                                             std::pair<uint64_t, uint64_t> seed)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    int gid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;
    int id = item_ct1.get_local_id(2);

    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + id;

    sycl::float4* vals_cast = reinterpret_cast<sycl::float4*>(vals);

    sycl::float4 data[128];

    int bid = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> state;
    /* curand_init(seed.first, idx, seed.second, &state); */
    state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(seed.first, {seed.second, (unsigned long)idx * 4});

    int group_index = bid * token_size + item_ct1.get_local_id(2);
    int reg_count = 0;
    int total_count = token_size * token_num;
    if (group_index < total_count) {
        float min = 10000.0;
        float max = -10000.0;

        while (tid < token_size) {
            sycl::float4 data_reg = vals_cast[group_index];
            data[reg_count] = data_reg;
            if (data_reg.x() > max) max = data_reg.x();
            if (data_reg.y() > max) max = data_reg.y();
            if (data_reg.w() > max) max = data_reg.w();
            if (data_reg.z() > max) max = data_reg.z();

            if (data_reg.x() < min) min = data_reg.x();
            if (data_reg.y() < min) min = data_reg.y();
            if (data_reg.w() < min) min = data_reg.w();
            if (data_reg.z() < min) min = data_reg.z();

            group_index += item_ct1.get_local_range(2);
            tid += item_ct1.get_local_range(2);
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(max, i);
            if (max < temp) max = temp;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shuffle_xor(min, i);
            if (min > temp) min = temp;
        }

        auto& partialMax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());
        auto& partialMin = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARP_SIZE]>(
            sycl::ext::oneapi::experimental::this_group<3>());

        if (lane == 0) partialMax[gid] = max;
        if (lane == 0) partialMin[gid] = min;

        /*
        DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if
        there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane < warp_num) max = partialMax[lane];
        if (lane < warp_num) min = partialMin[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(max, i);
            if (max < temp) max = temp;
        }
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shuffle_down(min, i);
            if (min > temp) min = temp;
        }

        max = g.shuffle(max, 0);
        min = g.shuffle(min, 0);

        float q_scale_val = ((max - min) + 1e-5) / (float)(1 << num_bits);
        float high_q = (float)((1 << num_bits) - 1);

        int offset = (bid)*token_size;
        for (int i = 0; i < reg_count; i++) {
            group_index = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
            if (group_index < token_size) {
                sycl::float4 q_data = data[i];

                sycl::float4 q_data_int;
                q_data_int.x() = (float)((int)((q_data.x() - min) / q_scale_val));
                q_data_int.y() = (float)((int)((q_data.y() - min) / q_scale_val));
                q_data_int.w() = (float)((int)((q_data.w() - min) / q_scale_val));
                q_data_int.z() = (float)((int)((q_data.z() - min) / q_scale_val));

                // Stochastic rounding
                sycl::float4 rand = state.generate<oneapi::mkl::rng::device::uniform<float>, 4>();

                float q_error[4];
                q_error[0] =
                    sycl::fabs(q_data.x() - ((q_data_int.x() * q_scale_val) + min)) / q_scale_val;
                q_error[1] =
                    sycl::fabs(q_data.y() - ((q_data_int.y() * q_scale_val) + min)) / q_scale_val;
                q_error[2] =
                    sycl::fabs(q_data.w() - ((q_data_int.w() * q_scale_val) + min)) / q_scale_val;
                q_error[3] =
                    sycl::fabs(q_data.z() - ((q_data_int.z() * q_scale_val) + min)) / q_scale_val;

                q_data_int.x() = (rand.x() < q_error[0] && q_data_int.x() < high_q)
                                     ? (q_data_int.x() + 1)
                                     : q_data_int.x();
                q_data_int.y() = (rand.y() < q_error[1] && q_data_int.y() < high_q)
                                     ? (q_data_int.y() + 1)
                                     : q_data_int.y();
                q_data_int.w() = (rand.w() < q_error[2] && q_data_int.w() < high_q)
                                     ? (q_data_int.w() + 1)
                                     : q_data_int.w();
                q_data_int.z() = (rand.z() < q_error[3] && q_data_int.z() < high_q)
                                     ? (q_data_int.z() + 1)
                                     : q_data_int.z();

                q_data_int.x() = q_data_int.x() * q_scale_val + min;
                q_data_int.y() = q_data_int.y() * q_scale_val + min;
                q_data_int.w() = q_data_int.w() * q_scale_val + min;
                q_data_int.z() = q_data_int.z() * q_scale_val + min;

                vals_cast[group_index + offset] = q_data_int;
            }
        }
    }
}
template <typename T>
void launch_sr_fake_quantize_kernel_asym(T* vals,
                                         int total_count,
                                         int group_num,
                                         int num_bits,
                                         dpct::queue_ptr stream)
{
    sycl::range<3> block_dim(1, 1, 1024);
    sycl::range<3> grid_dim(1, 1, group_num);

    uint64_t inc = total_count / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = TrainingContext::Instance().IncrementOffset(inc);

    /*
    DPCT1049:50: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        sr_fake_quantize_kernel<T> fn(vals, (total_count / group_num) / 4, group_num, num_bits, seed);
        stream->parallel_for(
            sycl::nd_range<3>(grid_dim * block_dim, block_dim), fn);
    }
}
template void launch_sr_fake_quantize_kernel_asym(float* vals,
                                                  int total_count,
                                                  int group_num,
                                                  int num_bits,
                                                  dpct::queue_ptr stream);
template void launch_sr_fake_quantize_kernel_asym(sycl::half* vals,
                                                  int total_count,
                                                  int group_num,
                                                  int num_bits,
                                                  dpct::queue_ptr stream);
