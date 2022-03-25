// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "csrc/aten/cpu/Interaction.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/vec512/bf16/vec/bf16_vec_kernel.h"
#include "csrc/cpu/vec512/int8/vec/int8_vec_kernel.h"
#include "csrc/jit/cpu/kernels/Interaction.h"
#include "csrc/quantization/AutoCast.hpp"

#include <ATen/Parallel.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <algorithm>
#include "csrc/utils/ipex_op_profile.h"

/*
 Custom op to optimize DLRM interaction part
*/

namespace torch_ipex {
namespace cpu {

namespace {

template <typename T>
static inline void cat(
    T* out,
    const std::vector<T*>& in_ptr,
    const std::vector<uint32_t>& feature_sizes,
    int feature_num) {
  size_t offset = 0;
  for (int j = 0; j < feature_num; j++) {
    move_ker(&out[offset], in_ptr[j], feature_sizes[j]);
    offset += feature_sizes[j];
  }
}

template <typename Tout, typename Tin>
static inline void cat_backward(
    const Tin* in,
    std::vector<Tout*>& out_ptr,
    const std::vector<uint32_t>& feature_sizes,
    int in_stride,
    int vector_size,
    int feature_num) {
  size_t offset = 0;
  for (int j = 0; j < feature_num; j++) {
    Tout* outp = out_ptr[j];
    for (int v = 0; v < feature_sizes[j]; v += vector_size) {
      move_ker_load_aligned(
          (Tout*)(outp + v), (Tin*)(&in[offset]), vector_size);
      offset += in_stride;
    }
  }
}

template <typename T>
static inline void flat_triangle(const T* in, T* out, size_t size) {
  size_t offset = 0;
  size_t in_off = size;
  out[offset] = in[in_off];
  offset = 1;
  in_off += size;
  for (int i = 2; i < size; i++) {
    move_ker(&out[offset], &in[in_off], i);
    offset += i;
    in_off += size;
  }
}

template <typename T>
static inline void flat_triangle_backward(const T* in, T* out, size_t size) {
  size_t offset = 0;
  size_t out_off = size;
  out[out_off] = in[offset];
  offset = 1;
  out_off += size;
  for (int i = 2; i < size; i++) {
    move_ker(&out[out_off], &in[offset], i);
    offset += i;
    out_off += size;
  }
}

template <typename T>
static inline void transpose_add(
    T* out,
    const T* in,
    uint32_t vector_nums,
    uint32_t out_stride) {
  T* outp = out;
  uint32_t j_row = 0;
  for (int32_t j = 0; j < vector_nums; j++) {
    const T* k_base = in;
    for (int32_t k = 0; k < vector_nums; k++) {
      outp[k] = in[j_row + k] + k_base[j];
      k_base += vector_nums;
    }
    j_row += vector_nums;
    outp += out_stride;
  }
}

template <typename T>
inline at::Tensor _interaction_forward(const std::vector<at::Tensor>& input) {
  IPEX_RECORD_FUNCTION("_interaction_forward", std::vector<c10::IValue>({}));
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];
  uint32_t input_nums = input.size();
  std::vector<uint32_t> feature_sizes(input_nums);
  std::vector<T*> input_data(input_nums);
  for (int i = 0; i < input_nums; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    auto feature_size = input[i].sizes()[1];
    feature_sizes[i] = feature_size;
    total_feature_size += feature_size;
    input_data[i] = input[i].data_ptr<T>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto out_data_line_len = interact_feature_size + vector_size;
  auto out = at::empty({batch_size, out_data_line_len}, input[0].options());
  auto out_data = out.data_ptr<T>();

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({vector_nums, vector_size});
  std::vector<int64_t> lhs_stride({vector_size, 1});
  std::vector<int64_t> rhs_shape({vector_size, vector_nums});
  std::vector<int64_t> rhs_stride({1, vector_size});
  std::vector<int64_t> res_shape({vector_nums, vector_nums});
  std::vector<int64_t> res_stride({vector_nums, 1});
  ideep::tensor::desc lhs_desc(
      std::move(lhs_shape), mkldnn_dtype, std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(
      std::move(rhs_shape), mkldnn_dtype, std::move(rhs_stride));
  ideep::tensor::desc res_desc(
      std::move(res_shape), mkldnn_dtype, std::move(res_stride));
  auto pd = ideep::matmul_forward::primitive_desc(
      {lhs_desc, rhs_desc, res_desc}, ideep::engine::cpu_engine());

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    T cat_buf[vector_nums * vector_size] __attribute__((aligned(64)));
    T mm_buf[vector_nums * vector_nums] __attribute__((aligned(64)));
    std::vector<T*> input_ptr(input_nums);
    for (uint32_t n = 0; n < input_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_sizes[n]];
    }
    ideep::tensor lhs({lhs_desc, cat_buf});
    ideep::tensor rhs({lhs_desc, cat_buf});
    ideep::tensor res({res_desc, mm_buf});
    auto p = dnnl::matmul(pd);
    for (int64_t i = start; i < end; i++) {
      move_ker(&out_data[i * out_data_line_len], input_ptr[0], vector_size);
      cat<T>(cat_buf, input_ptr, feature_sizes, input_nums);
      p.execute(
          ideep::stream::default_stream(),
          {{DNNL_ARG_SRC, lhs}, {DNNL_ARG_WEIGHTS, rhs}, {DNNL_ARG_DST, res}});
      T* flat_buf = (T*)(&out_data[i * out_data_line_len] + vector_size);
      flat_triangle<T>(mm_buf, flat_buf, vector_nums);
      for (uint32_t n = 0; n < input_nums; n++) {
        input_ptr[n] += feature_sizes[n];
      }
    }
  });

  return out;
}

template <typename T>
inline std::vector<at::Tensor> _interaction_backward(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.is_contiguous());
  IPEX_RECORD_FUNCTION("_interaction_backward", std::vector<c10::IValue>({}));
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];
  uint32_t input_nums = input.size();
  std::vector<uint32_t> feature_sizes(input_nums);
  std::vector<at::Tensor> output(input_nums);
  std::vector<T*> input_data(input_nums);
  std::vector<T*> output_data(input_nums);
  for (int i = 0; i < input_nums; i++) {
    auto feature_size = input[i].sizes()[1];
    output[i] = at::empty({batch_size, feature_size}, input[i].options());
    feature_sizes[i] = feature_size;
    total_feature_size += feature_size;
    input_data[i] = input[i].data_ptr<T>();
    output_data[i] = output[i].data_ptr<T>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto grad_out_data_line_len = interact_feature_size + vector_size;
  auto grad_out_data = grad_out.data_ptr<T>();

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({vector_nums, vector_nums});
  std::vector<int64_t> lhs_stride({vector_nums, 1});
  std::vector<int64_t> rhs_shape({vector_nums, vector_size});
  std::vector<int64_t> rhs_stride({vector_size, 1});
  std::vector<int64_t> res_shape({vector_nums, vector_size});
  std::vector<int64_t> res_stride({vector_size, 1});
  ideep::tensor::desc lhs_desc(
      std::move(lhs_shape), mkldnn_dtype, std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(
      std::move(rhs_shape), mkldnn_dtype, std::move(rhs_stride));
  ideep::tensor::desc res_desc(
      std::move(res_shape), mkldnn_dtype, std::move(res_stride));
  auto pd = ideep::matmul_forward::primitive_desc(
      {lhs_desc, rhs_desc, res_desc}, ideep::engine::cpu_engine());

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    auto mm_elems = vector_nums * vector_nums;
    T grad_mm_buf[mm_elems] __attribute__((aligned(64)));
    zero_ker(grad_mm_buf, mm_elems);
    T sum_buf[mm_elems] __attribute__((aligned(64)));
    T grad_cat_buf[vector_nums * vector_size] __attribute__((aligned(64)));
    T cat_buf[vector_nums * vector_size] __attribute__((aligned(64)));
    std::vector<T*> input_ptr(input_nums);
    std::vector<T*> output_ptr(input_nums);
    T* grad_out_ptr = &grad_out_data[start * grad_out_data_line_len];
    for (uint32_t n = 0; n < input_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_sizes[n]];
      output_ptr[n] = &output_data[n][start * feature_sizes[n]];
    }
    ideep::tensor lhs({lhs_desc, sum_buf});
    ideep::tensor rhs({lhs_desc, cat_buf});
    ideep::tensor res({res_desc, grad_cat_buf});
    auto p = dnnl::matmul(pd);
    for (int64_t i = start; i < end; i++) {
      // Special BMM characteristics in Interaction layer
      //  bmm(A, A'): two inputs are transposed to each other.
      //
      //             A --> (T) --> A'
      //              \         /
      //               \       /
      //                \     /
      //                 (bmm)
      //                   |
      //                   v
      //                  out
      //
      //  For traditional bmm backward propagation.
      //  e.g. gx: {gy, w'}, gw: {x', gy}
      //
      //  Can be expanded and optimized as:
      //  gx: {gy, A}, gA': {A', gy}
      //  gA = gx + (gA')' = {gy, A} + {A', gy}' = {gy + gy', A}
      flat_triangle_backward<T>(
          grad_out_ptr + vector_size, grad_mm_buf, vector_nums);
      // Calculate gy + gy'
      transpose_add(sum_buf, grad_mm_buf, vector_nums, vector_nums);
      // Calculate A
      cat<T>(cat_buf, input_ptr, feature_sizes, input_nums);
      p.execute(
          ideep::stream::default_stream(),
          {{DNNL_ARG_SRC, lhs}, {DNNL_ARG_WEIGHTS, rhs}, {DNNL_ARG_DST, res}});
      cat_backward<T, T>(
          grad_cat_buf,
          output_ptr,
          feature_sizes,
          vector_size,
          vector_size,
          input_nums);
      add_ker(output_ptr[0], grad_out_ptr, vector_size);
      grad_out_ptr += grad_out_data_line_len;
      for (uint32_t n = 0; n < input_nums; n++) {
        input_ptr[n] += feature_sizes[n];
        output_ptr[n] += feature_sizes[n];
      }
    }
  });
  return output;
}

#if defined(CPU_CAPABILITY_AMX)
typedef struct tileconfig_t {
  uint8_t palette_id;
  uint8_t startRow;
  uint8_t reserved[14];
  uint16_t colb[16];
  uint8_t rows[16];
} tileconfig_t;

const uint8_t TILE_M = 16;
const uint8_t TILE_N = 16;
const uint8_t TILE_IK = 64;
const uint8_t TILE_BK = 32;

static tileconfig_t tc = {0};
template <typename res_type, typename src_type>
inline void set_tile_config(
    const uint8_t TILE_M,
    const uint8_t TILE_N,
    const uint8_t TILE_K,
    const uint8_t KPACK) {
  tc.palette_id = 1;
  // tc.startRow = 0;
  // Configure C tiles
  for (int t = 0; t < 4; ++t) {
    tc.rows[t] = (uint8_t)TILE_M;
    tc.colb[t] = (uint16_t)(TILE_N * sizeof(res_type));
  }
  // Configure A tiles
  for (int t = 4; t < 6; ++t) {
    tc.rows[t] = (uint8_t)TILE_M;
    tc.colb[t] = (uint16_t)(TILE_K * sizeof(src_type));
  }
  // Configure B tile. B effectively has 64 rows and 16 columns.
  for (int t = 6; t < 8; ++t) {
    tc.rows[t] = (uint8_t)(TILE_K / KPACK);
    tc.colb[t] = (uint16_t)(TILE_N * KPACK * sizeof(src_type));
  }
}

template <>
inline at::Tensor _interaction_forward<at::BFloat16>(
    const std::vector<at::Tensor>& input) {
  IPEX_RECORD_FUNCTION(
      "_interaction_forward_bfloat16", std::vector<c10::IValue>({}));
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  int32_t vector_size = input[0].sizes()[1];
  int32_t input_nums = input.size();
  std::vector<uint32_t> feature_sizes(input_nums);
  std::vector<at::BFloat16*> input_data(input_nums);
  for (int i = 0; i < input_nums; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    auto feature_size = input[i].sizes()[1];
    feature_sizes[i] = feature_size;
    total_feature_size += feature_size;
    input_data[i] = input[i].data_ptr<at::BFloat16>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto out_data_line_len = interact_feature_size + vector_size;
  auto out = at::empty({batch_size, out_data_line_len}, input[0].options());
  auto out_data = out.data_ptr<at::BFloat16>();

  set_tile_config<float, at::BFloat16>(TILE_M, TILE_N, TILE_BK, 2);

  int32_t _AM = ((vector_nums + 31) >> 5) << 5;
  int32_t _AK = ((vector_size + 63) >> 6) << 6; // align to 64
  int32_t A_Stride = _AK * sizeof(at::BFloat16);
  int32_t B_Stride = _AM * sizeof(at::BFloat16) * 2;
  int32_t C_Stride = _AM * sizeof(float);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    const int32_t vector_len = vector_size * sizeof(at::BFloat16);
    float Cmem[_AM][_AM] __attribute__((aligned(64)));
    at::BFloat16 Amem[_AM][_AK] __attribute__((aligned(64)));
    zero_ker(&Amem[0][0], _AM * _AK);
    at::BFloat16 Bmem[_AK >> 1][_AM][2] __attribute__((aligned(64)));

    _tile_loadconfig((const void*)&tc);

    std::vector<at::BFloat16*> input_ptr(input_nums);
    for (uint32_t n = 0; n < input_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_sizes[n]];
      unsigned char* inp = (unsigned char*)(input_ptr[n]);
      for (int cache_line = 0; cache_line < vector_len; cache_line += 64) {
        _mm_prefetch(inp + cache_line, _MM_HINT_T0);
      }
    }
    for (int64_t i = start; i < end; i++) {
      move_ker(&out_data[i * out_data_line_len], input_ptr[0], vector_size);
      cat<at::BFloat16>(&Amem[0][0], input_ptr, feature_sizes, input_nums);
      for (int k = 0; k < (_AK >> 1); k++) {
        int32_t ak = (k << 1);
        for (int n = 0; n < _AM - 15; n += 16) {
          (*(uint32_t*)Bmem[k][n]) = (*(uint32_t*)(&Amem[n][ak]));
          (*(uint32_t*)Bmem[k][n + 1]) = (*(uint32_t*)(&Amem[n + 1][ak]));
          (*(uint32_t*)Bmem[k][n + 2]) = (*(uint32_t*)(&Amem[n + 2][ak]));
          (*(uint32_t*)Bmem[k][n + 3]) = (*(uint32_t*)(&Amem[n + 3][ak]));
          (*(uint32_t*)Bmem[k][n + 4]) = (*(uint32_t*)(&Amem[n + 4][ak]));
          (*(uint32_t*)Bmem[k][n + 5]) = (*(uint32_t*)(&Amem[n + 5][ak]));
          (*(uint32_t*)Bmem[k][n + 6]) = (*(uint32_t*)(&Amem[n + 6][ak]));
          (*(uint32_t*)Bmem[k][n + 7]) = (*(uint32_t*)(&Amem[n + 7][ak]));
          (*(uint32_t*)Bmem[k][n + 8]) = (*(uint32_t*)(&Amem[n + 8][ak]));
          (*(uint32_t*)Bmem[k][n + 9]) = (*(uint32_t*)(&Amem[n + 9][ak]));
          (*(uint32_t*)Bmem[k][n + 10]) = (*(uint32_t*)(&Amem[n + 10][ak]));
          (*(uint32_t*)Bmem[k][n + 11]) = (*(uint32_t*)(&Amem[n + 11][ak]));
          (*(uint32_t*)Bmem[k][n + 12]) = (*(uint32_t*)(&Amem[n + 12][ak]));
          (*(uint32_t*)Bmem[k][n + 13]) = (*(uint32_t*)(&Amem[n + 13][ak]));
          (*(uint32_t*)Bmem[k][n + 14]) = (*(uint32_t*)(&Amem[n + 14][ak]));
          (*(uint32_t*)Bmem[k][n + 15]) = (*(uint32_t*)(&Amem[n + 15][ak]));
        }
      }
      for (int n = 0; n < _AM; n += 2 * TILE_N) {
        for (int m = 0; m < _AM; m += 2 * TILE_M) {
          _tile_zero(0);
          _tile_zero(2);
          //_tile_zero(1);
          _tile_zero(3);
          for (int k = 0; k < _AK; k += 2 * TILE_BK) {
            int32_t bk = k >> 1;
            int32_t bk1 = (k + TILE_BK) >> 1;
            _tile_loadd(6, Bmem[bk][n], B_Stride);
            _tile_loadd(7, Bmem[bk1][n], B_Stride);
            _tile_loadd(4, &Amem[m][k], A_Stride);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(4, &Amem[m][k + TILE_BK], A_Stride);
            _tile_dpbf16ps(0, 4, 7);
            _tile_loadd(5, &Amem[m + TILE_M][k], A_Stride);
            _tile_loadd(4, &Amem[m + TILE_M][k + TILE_BK], A_Stride);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(2, 4, 7);
            _tile_loadd(6, Bmem[bk][n + TILE_N], B_Stride);
            _tile_dpbf16ps(3, 5, 6);
            _tile_loadd(7, Bmem[bk1][n + TILE_N], B_Stride);
            _tile_dpbf16ps(3, 4, 7);
            if (k == (_AK - 2 * TILE_BK)) {
              _tile_stored(0, &Cmem[m][n], C_Stride);
              _tile_stored(2, &Cmem[m + TILE_M][n], C_Stride);
              _tile_stored(3, &Cmem[m + TILE_M][n + TILE_N], C_Stride);
            }
          }
        }
      }

      for (uint32_t n = 0; n < input_nums; n++) {
        input_ptr[n] += feature_sizes[n];
        unsigned char* inp = (unsigned char*)(input_ptr[n]);
        for (int cache_line = 0; cache_line < vector_len; cache_line += 64) {
          _mm_prefetch(inp + cache_line, _MM_HINT_T0);
        }
      }

      at::BFloat16* flat_buf =
          (at::BFloat16*)(&out_data[i * out_data_line_len] + vector_size);
      size_t offset = 0;
      for (int i = 1; i < vector_nums; i++) {
        move_ker_load_aligned(&flat_buf[offset], Cmem[i], i);
        offset += i;
      }
    }
  });
  return out;
}

template <>
inline std::vector<at::Tensor> _interaction_backward<at::BFloat16>(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.is_contiguous());
  IPEX_RECORD_FUNCTION(
      "_interaction_backward_bfloat16", std::vector<c10::IValue>({}));
  int32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  int32_t vector_size = input[0].sizes()[1];
  int32_t input_nums = input.size();
  std::vector<uint32_t> feature_sizes(input_nums);
  std::vector<at::Tensor> output(input_nums);
  std::vector<at::BFloat16*> input_data(input_nums);
  std::vector<at::BFloat16*> output_data(input_nums);
  for (int i = 0; i < input_nums; i++) {
    auto feature_size = input[i].sizes()[1];
    output[i] = at::empty({batch_size, feature_size}, input[i].options());
    feature_sizes[i] = feature_size;
    total_feature_size += feature_size;
    input_data[i] = input[i].data_ptr<at::BFloat16>();
    output_data[i] = output[i].data_ptr<at::BFloat16>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto grad_out_data_line_len = interact_feature_size + vector_size;
  auto grad_out_data = grad_out.data_ptr<at::BFloat16>();

  int32_t _AM = ((vector_nums + 31) >> 5) << 5; // align to 32
  int32_t _AN = ((vector_size + 31) >> 5) << 5; // align to 32
  int32_t _AK = _AM;
  int32_t A_Stride = _AK * sizeof(at::BFloat16);
  int32_t B_Stride = _AN * sizeof(at::BFloat16) * 2;
  int32_t C_Stride = _AN * sizeof(float);
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    const int32_t vector_len = vector_size * sizeof(at::BFloat16);
    auto mm_elems = vector_nums * vector_nums;
    at::BFloat16 grad_mm_buf[mm_elems] __attribute__((aligned(64)));
    zero_ker(grad_mm_buf, mm_elems);
    at::BFloat16 sum_buf[_AM][_AK] __attribute__((aligned(64)));
    zero_ker(&sum_buf[0][0], _AM * _AK);
    at::BFloat16 cat_buf[_AK][_AN] __attribute__((aligned(64)));
    zero_ker(&cat_buf[0][0], _AK * _AN);
    at::BFloat16 Bmem[_AK / 2][_AN][2] __attribute__((aligned(64)));
    float Cmem[_AM][_AN] __attribute__((aligned(64)));

    _tile_loadconfig((const void*)&tc);

    std::vector<at::BFloat16*> input_ptr(input_nums);
    std::vector<at::BFloat16*> output_ptr(input_nums);
    for (uint32_t n = 0; n < input_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_sizes[n]];
      output_ptr[n] = &output_data[n][start * feature_sizes[n]];
      unsigned char* inp = (unsigned char*)(input_ptr[n]);
      for (uint32_t cache_line = 0; cache_line < vector_len; cache_line += 64) {
        _mm_prefetch(inp + cache_line, _MM_HINT_T0);
      }
    }

    at::BFloat16* grad_out_ptr = &grad_out_data[start * grad_out_data_line_len];
    for (int64_t i = start; i < end; i++) {
      flat_triangle_backward<at::BFloat16>(
          grad_out_ptr + vector_size, grad_mm_buf, vector_nums);
      transpose_add(&sum_buf[0][0], grad_mm_buf, vector_nums, _AK);
      cat<at::BFloat16>(&cat_buf[0][0], input_ptr, feature_sizes, input_nums);
      for (int k = 0; k < (_AK >> 1); ++k) {
        int32_t ak = (k << 1);
        for (int n = 0; n < (_AN - 31); n += 32) {
          auto akx16 = _mm256_load_si256((__m256i const*)(&cat_buf[ak][n]));
          auto ak1x16 =
              _mm256_load_si256((__m256i const*)(&cat_buf[ak + 1][n]));
          auto akx16_1 =
              _mm256_load_si256((__m256i const*)(&cat_buf[ak][n + 16]));
          auto ak1x16_1 =
              _mm256_load_si256((__m256i const*)(&cat_buf[ak + 1][n + 16]));
          auto low_part = _mm256_unpacklo_epi16(akx16, ak1x16);
          auto low_part1 = _mm256_unpacklo_epi16(akx16_1, ak1x16_1);
          auto high_part = _mm256_unpackhi_epi16(akx16, ak1x16);
          auto high_part1 = _mm256_unpackhi_epi16(akx16_1, ak1x16_1);
          auto out0 = _mm256_shuffle_i64x2(low_part, high_part, 0x0);
          auto out1 = _mm256_shuffle_i64x2(low_part, high_part, 0x3);
          auto out2 = _mm256_shuffle_i64x2(low_part1, high_part1, 0x0);
          auto out3 = _mm256_shuffle_i64x2(low_part1, high_part1, 0x3);
          _mm256_store_si256((__m256i*)(&Bmem[k][n][0]), out0);
          _mm256_store_si256((__m256i*)(&Bmem[k][n + 8][0]), out1);
          _mm256_store_si256((__m256i*)(&Bmem[k][n + 16][0]), out2);
          _mm256_store_si256((__m256i*)(&Bmem[k][n + 24][0]), out3);
        }
      }

      for (uint32_t n = 0; n < input_nums; n++) {
        unsigned char* outp = (unsigned char*)(output_ptr[n]);
        for (uint32_t cache_line = 0; cache_line < vector_len;
             cache_line += 64) {
          _mm_prefetch(outp + cache_line, _MM_HINT_T0);
        }
      }
      for (uint32_t cache_line = 0; cache_line < vector_len; cache_line += 64) {
        _mm_prefetch(grad_out_ptr + cache_line, _MM_HINT_T0);
      }

      for (int n = 0; n < _AN; n += 2 * TILE_N) {
        for (int m = 0; m < _AM; m += 2 * TILE_M) {
          _tile_zero(0);
          _tile_zero(1);
          _tile_zero(2);
          _tile_zero(3);
          if (_AK == TILE_BK) {
            _tile_loadd(6, Bmem[0][n], B_Stride);
            _tile_loadd(4, &sum_buf[m][0], A_Stride);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, &Cmem[m][n], C_Stride);
            _tile_loadd(5, &sum_buf[m + TILE_M][0], A_Stride);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, &Cmem[m + TILE_M][n], C_Stride);
            _tile_loadd(7, Bmem[0][n + TILE_N], B_Stride);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, &Cmem[m][n + TILE_N], C_Stride);
            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, &Cmem[m + TILE_M][n + TILE_N], C_Stride);
          } else {
            for (int k = 0; k < _AK; k += TILE_BK) {
              int32_t bk = k >> 1;
              _tile_loadd(6, Bmem[bk][n], B_Stride);
              _tile_loadd(4, &sum_buf[m][k], A_Stride);
              _tile_dpbf16ps(0, 4, 6);
              _tile_loadd(5, &sum_buf[m + TILE_M][k], A_Stride);
              _tile_dpbf16ps(2, 5, 6);
              _tile_loadd(7, Bmem[bk][n + TILE_N], B_Stride);
              _tile_dpbf16ps(1, 4, 7);
              _tile_dpbf16ps(3, 5, 7);
              if (k == _AK - TILE_BK) {
                _tile_stored(0, &Cmem[m][n], C_Stride);
                _tile_stored(2, &Cmem[m + TILE_M][n], C_Stride);
                _tile_stored(1, &Cmem[m][n + TILE_N], C_Stride);
                _tile_stored(3, &Cmem[m + TILE_M][n + TILE_N], C_Stride);
              }
            }
          }
        }
      }

      for (uint32_t n = 0; n < input_nums; n++) {
        input_ptr[n] += feature_sizes[n];
        unsigned char* inp = (unsigned char*)(input_ptr[n]);
        for (int cache_line = 0; cache_line < vector_len; cache_line += 64) {
          _mm_prefetch(inp + cache_line, _MM_HINT_T0);
        }
      }

      cat_backward<at::BFloat16, float>(
          &Cmem[0][0], output_ptr, feature_sizes, _AN, vector_size, input_nums);
      add_ker(output_ptr[0], grad_out_ptr, vector_size);
      grad_out_ptr += grad_out_data_line_len;
      for (uint32_t n = 0; n < input_nums; n++) {
        output_ptr[n] += feature_sizes[n];
      }
    }
  });
  return output;
}
#endif

at::Tensor interaction_forward_kernel_impl(
    const std::vector<at::Tensor>& input) {
  if (input[0].scalar_type() == at::kFloat) {
    for (auto& in : input) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(in.scalar_type() == at::kFloat);
    }
    return _interaction_forward<float>(input);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[0].scalar_type() == at::kBFloat16);
    for (const auto& in : input) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(in.scalar_type() == at::kBFloat16);
    }
    return _interaction_forward<at::BFloat16>(input);
  }
}

std::vector<at::Tensor> interaction_backward_kernel_impl(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input) {
  if (grad_out.scalar_type() == at::kFloat) {
    return _interaction_backward<float>(
        grad_out, torch_ipex::autocast::cpu_cached_cast(at::kFloat, input));
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.scalar_type() == at::kBFloat16);
    // todo: move the autograd registion from python into C++.
    // Performance overhead in training here if you use autocast.
    // Because we save the ctx.arg in python before autocast, we have duplicated
    // cast for the input: here and in autocast of the forward path.
    return _interaction_backward<at::BFloat16>(
        grad_out, torch_ipex::autocast::cpu_cached_cast(at::kBFloat16, input));
  }
}

#if defined(CPU_CAPABILITY_AVX512)
static inline void _interaction_s8s8_scale_s32s8_128(
    int8_t* out,
    size_t M,
    const float* __attribute__((aligned(64))) scales,
    __m512i* convert_to_s16_buf,
    __m512i* cat_buf) {
  auto* a = (const __m512i*)&convert_to_s16_buf[0];
  auto* b = (const __m512i*)&convert_to_s16_buf[4];
  mul_and_sum_s16x128_to_s32x16(cat_buf[0], b, a);
  size_t offset = 1;
  for (int i = 2; i < M; i++) {
    auto* c = (const __m512i*)&convert_to_s16_buf[i * 4];
    int j = 0;
    for (; j < i - 1; j += 2) {
      a = (const __m512i*)&convert_to_s16_buf[j * 4];
      b = (const __m512i*)&convert_to_s16_buf[j * 4 + 4];
      mul_and_sum_s16x128x2_to_s32x16x2(
          cat_buf[offset], cat_buf[offset + 1], c, a, c, b);
      offset += 2;
    }
    for (; j < i; j++) {
      a = (const __m512i*)&convert_to_s16_buf[j * 4];
      mul_and_sum_s16x128_to_s32x16(cat_buf[offset], c, a);
      offset++;
    }
  }

  // Do reduce add with scale
  size_t off = 0;
  for (; off < offset - 15; off += 16) {
    __m512 scale_m512 = _mm512_load_ps((const void*)(scales + off));
    reduce_add_s32x16x16_with_scales(out + off, cat_buf + off, scale_m512);
  }
  __m512 scale_m512 = _mm512_load_ps((const void*)(scales + off));
  auto mask = ((1 << (offset - off)) - 1);
  reduce_add_s32x16x16_with_scales_and_mask_store(
      out + off, mask, cat_buf + off, scale_m512);
}
#endif

static inline void _interaction_s8s8_scale_s32s8(
    int8_t* out,
    const std::vector<int8_t*>& input_addr,
    size_t M,
    size_t K,
    float* scales) {
  size_t offset = 0;
  for (int i = 1; i < M; i++) {
    int8_t* a = input_addr[i];
    for (int j = 0; j < i; j++) {
      int8_t* b = input_addr[j];
      out[offset] = _dot_s8s8_scale_s32s8(a, b, K, scales[offset]);
      offset++;
    }
  }
}

at::Tensor dil_qinteraction_kernel_impl(
    const std::vector<at::Tensor> input,
    double output_scale,
    int64_t o_zp,
    at::ScalarType o_dtype) {
  uint32_t input_size = input.size();
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];

  std::vector<float> in_scales(input_size);
  std::vector<uint32_t> feature_sizes(input_size);
  std::vector<int8_t*> input_data(input_size);
  for (auto i = 0; i < input_size; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    auto cur_input = input[i];
    input_data[i] = reinterpret_cast<int8_t*>(cur_input.data_ptr<at::qint8>());
    in_scales[i] = at::native::q_scale_quant(cur_input);
    feature_sizes[i] = cur_input.sizes()[1];
    total_feature_size += feature_sizes[i];
  }

  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto out_data_line_len = interact_feature_size + vector_size;

  // init output tensor
  at::QuantizerPtr output_quantizer =
      at::make_per_tensor_affine_quantizer(output_scale, /*zp=*/0, at::kQInt8);
  at::Tensor output = at::new_qtensor(
      /*sizes=*/{batch_size, out_data_line_len},
      input[0].options(),
      output_quantizer);
  int8_t* out_data = reinterpret_cast<int8_t*>(output.data_ptr<at::qint8>());
  auto aligned_off = (interact_feature_size >> 4) << 4;
  aligned_off =
      (aligned_off < interact_feature_size) ? (aligned_off + 16) : aligned_off;
  float out_in_scales[aligned_off] __attribute__((aligned(64)));
  size_t offset = 0;
  for (int i = 1; i < vector_nums; i++) {
    for (int j = 0; j < i; j++) {
      auto input_scale = in_scales[i] * in_scales[j];
      out_in_scales[offset] = input_scale / output_scale;
      offset++;
    }
  }

  float dense_scale = in_scales[0] / output_scale;

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    __m512i cat_buf[aligned_off] __attribute__((aligned(64)));
    __m512i convert_to_s16_buf[vector_nums * 4] __attribute__((aligned(64)));
    std::vector<int8_t*> input_addr(vector_nums);
    for (int64_t i = start; i < end; i++) {
      int8_t* out_ptr = &out_data[i * out_data_line_len];
      int8_t* flat_buf = (int8_t*)(out_ptr + vector_size);
      auto row_len = i * vector_size;
#if defined(CPU_CAPABILITY_AVX512)
      if (vector_size == 128) {
        int k = 0;
        for (; k < vector_nums - 1; k += 2) {
          load_s8x128x2_to_s16x128x2(
              &convert_to_s16_buf[k * 4],
              &input_data[k][row_len],
              &input_data[k + 1][row_len]);
        }
        for (; k < vector_nums; k++) {
          load_s8x128_to_s16x128(
              &convert_to_s16_buf[k * 4], &input_data[k][row_len]);
        }
        scale_and_move_ker_128(
            out_ptr, &input_data[0][i * vector_size], dense_scale);
        _interaction_s8s8_scale_s32s8_128(
            flat_buf, vector_nums, out_in_scales, convert_to_s16_buf, cat_buf);
      }
      continue;
#endif
      for (int k = 0; k < vector_nums; k++) {
        input_addr[k] = &input_data[k][row_len];
      }
      scale_and_move_ker(
          out_ptr, &input_data[0][i * vector_size], dense_scale, vector_size);
      _interaction_s8s8_scale_s32s8(
          flat_buf, input_addr, vector_nums, vector_size, out_in_scales);
    }
  });

  return output;
}

} // anonymous namespace

REGISTER_DISPATCH(
    interaction_forward_kernel_stub,
    &interaction_forward_kernel_impl);
REGISTER_DISPATCH(
    interaction_backward_kernel_stub,
    &interaction_backward_kernel_impl);
REGISTER_DISPATCH(dil_qinteraction_kernel_stub, &dil_qinteraction_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
