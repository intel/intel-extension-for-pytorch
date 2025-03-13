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

#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <sycl/sycl.hpp>
#include <cassert>
#include <vector>
#include "context.h"
#include "ops.h"

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
typedef at::Half half;
typedef at::BFloat16 bfloat16;

#define MAKE_BLOCKWISE8(fname, optim_name, gtype, gbits) \
  void fname##_8bit_blockwise_grad_##gbits(              \
      gtype* p,                                          \
      gtype* g,                                          \
      unsigned char* state1,                             \
      unsigned char* state2,                             \
      float beta1,                                       \
      float beta2,                                       \
      float beta3,                                       \
      float alpha,                                       \
      float eps,                                         \
      int step,                                          \
      float lr,                                          \
      float* quantiles1,                                 \
      float* quantiles2,                                 \
      float* absmax1,                                    \
      float* absmax2,                                    \
      float weight_decay,                                \
      const float gnorm_scale,                           \
      bool skip_zeros,                                   \
      int n) {                                           \
    optimizerStatic8bitBlockwise<gtype, optim_name>(     \
        p,                                               \
        g,                                               \
        state1,                                          \
        state2,                                          \
        beta1,                                           \
        beta2,                                           \
        beta3,                                           \
        alpha,                                           \
        eps,                                             \
        step,                                            \
        lr,                                              \
        quantiles1,                                      \
        quantiles2,                                      \
        absmax1,                                         \
        absmax2,                                         \
        weight_decay,                                    \
        gnorm_scale,                                     \
        skip_zeros,                                      \
        n);                                              \
  }

MAKE_BLOCKWISE8(adam, ADAM, half, fp16)
MAKE_BLOCKWISE8(adam, ADAM, float, fp32)
MAKE_BLOCKWISE8(adam, ADAM, bfloat16, bf16)

#define MAKE_CBLOCKWISE8(fname, optim_name, gtype, gbits) \
  void c##fname##_8bit_blockwise_grad_##gbits(            \
      at::Tensor& p,                                      \
      at::Tensor& g,                                      \
      at::Tensor& state1,                                 \
      at::Tensor& state2,                                 \
      double beta1,                                       \
      double beta2,                                       \
      double beta3,                                       \
      double alpha,                                       \
      double eps,                                         \
      int64_t step,                                       \
      double lr,                                          \
      at::Tensor& quantiles1,                             \
      at::Tensor& quantiles2,                             \
      at::Tensor& absmax1,                                \
      at::Tensor& absmax2,                                \
      double weight_decay,                                \
      const double gnorm_scale,                           \
      bool skip_zeros,                                    \
      int64_t n) {                                        \
    fname##_8bit_blockwise_grad_##gbits(                  \
        p.data_ptr<gtype>(),                              \
        g.data_ptr<gtype>(),                              \
        state1.data_ptr<unsigned char>(),                 \
        state2.data_ptr<unsigned char>(),                 \
        beta1,                                            \
        beta2,                                            \
        beta3,                                            \
        alpha,                                            \
        eps,                                              \
        step,                                             \
        lr,                                               \
        quantiles1.data_ptr<float>(),                     \
        quantiles2.data_ptr<float>(),                     \
        absmax1.data_ptr<float>(),                        \
        absmax2.data_ptr<float>(),                        \
        weight_decay,                                     \
        gnorm_scale,                                      \
        skip_zeros,                                       \
        n);                                               \
  }

MAKE_CBLOCKWISE8(adam, ADAM, half, fp16)
MAKE_CBLOCKWISE8(adam, ADAM, float, fp32)
MAKE_CBLOCKWISE8(adam, ADAM, bfloat16, bf16)

void percentileClipping_g32(float* g, float* gnorm_vec, int step, const int n) {
  percentileClipping<float>(g, gnorm_vec, step, n);
}
void percentileClipping_g16(half* g, float* gnorm_vec, int step, const int n) {
  percentileClipping<half>(g, gnorm_vec, step, n);
}

void cpercentile_clipping_g32(
    at::Tensor& g,
    at::Tensor& gnorm_vec,
    int64_t step,
    const int64_t n) {
  auto g_ptr = g.data_ptr<float>();
  auto gnorm_vec_ptr = gnorm_vec.data_ptr<float>();
  percentileClipping_g32(g_ptr, gnorm_vec_ptr, step, n);
}
void cpercentile_clipping_g16(
    at::Tensor& g,
    at::Tensor& gnorm_vec,
    int64_t step,
    const int64_t n) {
  auto g_ptr = g.data_ptr<half>();
  auto gnorm_vec_ptr = gnorm_vec.data_ptr<float>();
  percentileClipping_g16(g_ptr, gnorm_vec_ptr, step, n);
}

void dequantizeBlockwise_fp32(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    const int n) {
  dequantizeBlockwise<float, General8bit>(code, A, absmax, out, blocksize, n);
}

void dequantizeBlockwise_fp16(
    float* code,
    unsigned char* A,
    float* absmax,
    half* out,
    int blocksize,
    const int n) {
  dequantizeBlockwise<half, General8bit>(code, A, absmax, out, blocksize, n);
}

void dequantizeBlockwise_bf16(
    float* code,
    unsigned char* A,
    float* absmax,
    bfloat16* out,
    int blocksize,
    const int n) {
  dequantizeBlockwise<bfloat16, General8bit>(
      code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_fp32(
    at::Tensor& code,
    at::Tensor& A,
    at::Tensor& absmax,
    at::Tensor& out,
    int64_t blocksize,
    const int64_t n) {
  dequantizeBlockwise_fp32(
      code.data_ptr<float>(),
      A.data_ptr<unsigned char>(),
      absmax.data_ptr<float>(),
      out.data_ptr<float>(),
      blocksize,
      n);
}

void cdequantize_blockwise_fp16(
    at::Tensor& code,
    at::Tensor& A,
    at::Tensor& absmax,
    at::Tensor& out,
    int64_t blocksize,
    const int64_t n) {
  dequantizeBlockwise_fp16(
      code.data_ptr<float>(),
      A.data_ptr<unsigned char>(),
      absmax.data_ptr<float>(),
      out.data_ptr<half>(),
      blocksize,
      n);
}

void cdequantize_blockwise_bf16(
    at::Tensor& code,
    at::Tensor& A,
    at::Tensor& absmax,
    at::Tensor& out,
    int64_t blocksize,
    const int64_t n) {
  dequantizeBlockwise_bf16(
      code.data_ptr<float>(),
      A.data_ptr<unsigned char>(),
      absmax.data_ptr<float>(),
      out.data_ptr<bfloat16>(),
      blocksize,
      n);
}

BNB_LIBRARY_FRAGMENT() {
  BNB_OP_REGISTER(
      "cpercentile_clipping_g32",
      cpercentile_clipping_g32,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cpercentile_clipping_g16",
      cpercentile_clipping_g16,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cadam_8bit_blockwise_grad_fp32",
      cadam_8bit_blockwise_grad_fp32,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cadam_8bit_blockwise_grad_fp16",
      cadam_8bit_blockwise_grad_fp16,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cadam_8bit_blockwise_grad_bf16",
      cadam_8bit_blockwise_grad_bf16,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cdequantize_blockwise_fp32",
      cdequantize_blockwise_fp32,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cdequantize_blockwise_fp16",
      cdequantize_blockwise_fp16,
      c10::DispatchKey::AutogradXPU);
  BNB_OP_REGISTER(
      "cdequantize_blockwise_bf16",
      cdequantize_blockwise_bf16,
      c10::DispatchKey::AutogradXPU);
}