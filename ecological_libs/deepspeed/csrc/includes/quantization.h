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

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.h>
#include "ds_kernel_utils.h"

namespace quantize {

enum class Type { Symmetric, Asymmetric };

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

DS_HD_INLINE bool requires_offset(Type qType) { return qType == Type::Asymmetric; }

}  // namespace quantize

void launch_quant(int8_t* output_data,
                  float* params,
                  const sycl::half* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  dpct::queue_ptr stream);

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int total_elems,
                              dpct::queue_ptr stream);

void launch_swizzled_quant(int8_t* q_data,
                           float* q_scales,
                           const sycl::half* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           dpct::queue_ptr stream);

void launch_dequant_reduce(int8_t* reduced_data,
                           float* reduced_scales,
                           const int8_t* input_data,
                           const float* input_scales,
                           int num_gpus,
                           int num_bits,
                           quantize::Type quant_type,
                           int out_groups,
                           int elems_per_out_group,
                           int elems_per_in_tensor,
                           int groups_per_in_tensor,
                           int elems_per_in_group,
                           dpct::queue_ptr stream);

template <typename T>
void launch_fake_quantize_kernel(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 dpct::queue_ptr stream);
template <typename T>
void launch_sr_fake_quantize_kernel(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    dpct::queue_ptr stream);
template <typename T>
void launch_fake_quantize_kernel_asym(T* vals,
                                      int total_count,
                                      int group_num,
                                      int num_bits,
                                      dpct::queue_ptr stream);
template <typename T>
void launch_sr_fake_quantize_kernel_asym(T* vals,
                                         int total_count,
                                         int group_num,
                                         int num_bits,
                                         dpct::queue_ptr stream);

void launch_dequantize_int4_to_half_experimental(uint8_t* data_in,
                                                 sycl::half* data_out,
                                                 sycl::half* scale_buffer,
                                                 sycl::half* min_val_buffer,
                                                 int num_group,
                                                 int group_size,
                                                 dpct::queue_ptr stream);

void launch_dequantize_int8_to_half_experimental(uint8_t* data_in,
                                                 sycl::half* data_out,
                                                 sycl::half* scale_buffer,
                                                 sycl::half* min_val_buffer,
                                                 int num_group,
                                                 int group_size,
                                                 dpct::queue_ptr stream);
