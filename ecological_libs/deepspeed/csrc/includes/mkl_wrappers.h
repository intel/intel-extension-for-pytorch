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
#include <assert.h>
#include <dpct/blas_utils.h>

#include <stdio.h>

int mkl_gemm_ex(dpct::queue_ptr handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   int algo = -1);

int mkl_gemm_ex(dpct::queue_ptr handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const sycl::half* A,
                   const sycl::half* B,
                   sycl::half* C,
                   int algo = 99);

int mkl_strided_batched_gemm(dpct::queue_ptr handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo = -1);

int mkl_strided_batched_gemm(dpct::queue_ptr handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const sycl::half* A,
                                const sycl::half* B,
                                sycl::half* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo = 99);
