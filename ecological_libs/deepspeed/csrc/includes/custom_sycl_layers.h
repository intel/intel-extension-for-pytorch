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
#include <dpct/rng_utils.h>

#include <stdio.h>
#include <stdlib.h>

#include "context.h"
/* #include "mkl_wrappers.h" */


#define MAX_THREADS 1024
#define THREADS 256

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32

// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32

#define MAX_REGISTERS 256

#define MAX_REG 256

#define WARP_SIZE_BITS 5

// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(const T* input,
                      const T* bias,
                      T* output,
                      int intermediate_size,
                      int batch_size,
                      dpct::queue_ptr stream);

template <typename T>
void launch_gelu(const T* input,
                 T* output,
                 int intermediate_size,
                 int batch_size,
                 dpct::queue_ptr stream);

template <typename T>
void launch_d_gelu(T* d_output,
                   const T* input,
                   const T* bias,
                   int intermediate_size,
                   int batch_size,
                   dpct::queue_ptr stream);

// Custom fused bias add with layer normalization
template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     dpct::queue_ptr stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means);

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     dpct::queue_ptr stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars);

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* X_data,
                                         const T* vars,
                                         const T* means,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         dpct::queue_ptr stream[2]);
template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* vals_hat,
                                         const T* vars,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         dpct::queue_ptr stream[2],
                                         bool invertible = false,
                                         const T* betta = nullptr);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* X_data,
                               const T* vars,
                               const T* means,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               dpct::queue_ptr stream[2]);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* vals_hat,
                               const T* vars,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               dpct::queue_ptr stream[2],
                               bool invertible = false,
                               const T* betta = nullptr);

template <typename T>
void launch_layerNorm_backward_nreversible(const T* out_grad,
                                           const T* vals,
                                           const T* out_grad_trans,
                                           const T* vals_trans,
                                           const T* means,
                                           const T* vars,
                                           const T* gamma,
                                           T* gamma_grad,
                                           T* betta_grad,
                                           T* inp_grad,
                                           int batch_size,
                                           int hidden_dim,
                                           dpct::queue_ptr stream[2]);

template <typename T>
void Transpose(const T* inp_mat, T* out_mat, int rows, int cols, dpct::queue_ptr stream);

template <typename T>
void launch_attn_softmax_backward(T* out_grad,
                                  const T* soft_inp,
                                  int batch_size,
                                  int heads,
                                  int seq_length,
                                  dpct::queue_ptr stream);

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     dpct::queue_ptr stream);

// Custom softmax with scaling and attention mask addition
template <typename T>
void launch_attn_softmax(T* vals,
                         const T* attn_mask,
                         int batch_size,
                         int heads,
                         int sequence_length,
                         dpct::queue_ptr stream);

template <typename T>
void launch_transform_0213(T* output,
                           const T* vals,
                           int batch_size,
                           int seq_length,
                           int hidden_dim,
                           int heads,
                           dpct::queue_ptr stream);

// Custom bias add
template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    const T* vals,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    int hidden_dim,
                                    int heads,
                                    dpct::queue_ptr stream,
                                    int trans_count);

// 4D transform [0, 1, 2, 3] -> [0, 2, 1, 3]
template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             int batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             dpct::queue_ptr stream,
                             int trans_count);

template <typename T>
void launch_dropout(T* vals,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    dpct::queue_ptr stream);

template <typename T>
void launch_dropout(T* vals_out,
                    const T* vals,
                    uint8_t* mask,
                    int total_count,
                    int dim,
                    float ratio,
                    dpct::queue_ptr stream,
                    bool bwd = false);

template <typename T>
void launch_dropout(T* out,
                    const T* vals,
                    const T* residual,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    dpct::queue_ptr stream);

template <typename T>
void launch_dropout_grad(T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         dpct::queue_ptr stream);

template <typename T>
void launch_dropout_grad(T* vals_out,
                         const T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         dpct::queue_ptr stream);

template <typename T>
void launch_fuse_transpose_bias_kernel(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols,
                                       dpct::queue_ptr stream);

void launch_param_update(const float* input, sycl::half* output, int size, dpct::queue_ptr stream);
void launch_param_update_half(const float* input,
                              sycl::half* output,
                              int size,
                              dpct::queue_ptr stream);

void launch_token_sort(int32_t* indices,
                       int layers,
                       int batch_size,
                       int reserved_size,
                       int original_tokens,
                       dpct::queue_ptr stream);

template <typename T>
void launch_gather_tokens(T* retained_tokens,
                          T* activations,
                          int32_t* gather_indices,
                          int32_t batch_size,
                          int32_t sampled_tokens,
                          int32_t channels,
                          int32_t read_batch_stride,
                          int32_t read_seq_stride,
                          int32_t write_batch_stride,
                          int32_t write_seq_stride,
                          dpct::queue_ptr stream);

template <typename T>
void launch_scatter_tokens(T* all_activations,
                           T* layer_activations,
                           int32_t* gather_indices,
                           int32_t batch_size,
                           int32_t sampled_tokens,
                           int32_t channels,
                           int32_t read_batch_stride,
                           int32_t read_seq_stride,
                           int32_t write_batch_stride,
                           int32_t write_seq_stride,
                           dpct::queue_ptr stream);

template <typename T>
void launch_slice_gpt_mask(T* output_mask,
                           const T* input_mask,
                           int batch_size,
                           int truncated_seq_len,
                           int orig_seq_len,
                           dpct::queue_ptr stream);

template <typename T>
void launch_slice_bert_mask(T* output_mask,
                            const T* input_mask,
                            const int32_t* retained_indices,
                            int32_t layers,
                            int32_t batch_size,
                            int32_t truncated_seq_len,
                            int32_t orig_seq_len,
                            dpct::queue_ptr stream);
