// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "csrc/aten/cpu/Interaction.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/vec/vec.h"
#include "csrc/jit/cpu/kernels/Interaction.h"

#include <ATen/Parallel.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <algorithm>

/*
 Custom op to optimize DLRM interaction part
*/

namespace torch_ipex {
namespace cpu {

namespace {

using namespace torch_ipex::cpu::kernel;

template <typename T>
static inline void cat(
    T* out,
    const std::vector<T*>& in_ptr,
    int feature_size,
    int out_stride) {
  size_t offset = 0;
  auto feature_nums = in_ptr.size();
  for (int j = 0; j < feature_nums; j++) {
    move_ker(&out[offset], in_ptr[j], feature_size);
    offset += out_stride;
  }
}

template <typename Tout, typename Tin>
static inline void cat_backward(
    const Tin* in,
    std::vector<Tout*>& out_ptr,
    int feature_size,
    int in_stride) {
  size_t offset = 0;
  auto feature_nums = out_ptr.size();
  for (int j = 0; j < feature_nums; j++) {
    move_ker((Tout*)out_ptr[j], (Tin*)(&in[offset]), feature_size);
    offset += in_stride;
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
    uint32_t feature_nums,
    uint32_t out_stride) {
  T* outp = out;
  uint32_t j_row = 0;
  for (int32_t j = 0; j < feature_nums; j++) {
    const T* k_base = in;
    for (int32_t k = 0; k < feature_nums; k++) {
      outp[k] = in[j_row + k] + k_base[j];
      k_base += feature_nums;
    }
    j_row += feature_nums;
    outp += out_stride;
  }
}

template <typename T>
inline at::Tensor _interaction_forward(const std::vector<at::Tensor>& input) {
  RECORD_FUNCTION("_interaction_forward", c10::ArrayRef<c10::IValue>({}));
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t feature_size = input[0].sizes()[1];
  uint32_t feature_nums = input.size();
  std::vector<T*> input_data(feature_nums);
  for (int i = 0; i < feature_nums; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    TORCH_CHECK(
        input[i].sizes()[1] == feature_size,
        "expect all inputs have same feature size");
    input_data[i] = input[i].data_ptr<T>();
  }
  auto interact_feature_size = feature_nums * (feature_nums - 1) / 2;
  auto out_data_line_len = interact_feature_size + feature_size;
  auto out = at::empty({batch_size, out_data_line_len}, input[0].options());
  auto out_data = out.data_ptr<T>();

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({feature_nums, feature_size});
  std::vector<int64_t> lhs_stride({feature_size, 1});
  std::vector<int64_t> rhs_shape({feature_size, feature_nums});
  std::vector<int64_t> rhs_stride({1, feature_size});
  std::vector<int64_t> res_shape({feature_nums, feature_nums});
  std::vector<int64_t> res_stride({feature_nums, 1});
  ideep::tensor::desc lhs_desc(
      std::move(lhs_shape), mkldnn_dtype, std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(
      std::move(rhs_shape), mkldnn_dtype, std::move(rhs_stride));
  ideep::tensor::desc res_desc(
      std::move(res_shape), mkldnn_dtype, std::move(res_stride));

  auto op_attr = dnnl::primitive_attr();
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = ideep::matmul_forward::primitive_desc(
      {lhs_desc, rhs_desc, res_desc}, op_attr, ideep::engine::cpu_engine());

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    T cat_buf[feature_nums * feature_size] __attribute__((aligned(64)));
    T mm_buf[feature_nums * feature_nums] __attribute__((aligned(64)));
    std::vector<T*> input_ptr(feature_nums);
    for (uint32_t n = 0; n < feature_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_size];
    }
    ideep::tensor lhs({lhs_desc, cat_buf});
    ideep::tensor rhs({lhs_desc, cat_buf});
    ideep::tensor res({res_desc, mm_buf});
    ideep::tensor scratchpad(pd.scratchpad_desc());
    auto p = dnnl::matmul(pd);
    for (int64_t i = start; i < end; i++) {
      move_ker(&out_data[i * out_data_line_len], input_ptr[0], feature_size);
      cat<T>(cat_buf, input_ptr, feature_size, feature_size);
      p.execute(
          ideep::stream::default_stream(),
          {{DNNL_ARG_SRC, lhs},
           {DNNL_ARG_WEIGHTS, rhs},
           {DNNL_ARG_DST, res},
           {DNNL_ARG_SCRATCHPAD, scratchpad}});
      T* flat_buf = (T*)(&out_data[i * out_data_line_len] + feature_size);
      flat_triangle<T>(mm_buf, flat_buf, feature_nums);
      for (uint32_t n = 0; n < feature_nums; n++) {
        input_ptr[n] += feature_size;
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
  RECORD_FUNCTION("_interaction_backward", c10::ArrayRef<c10::IValue>({}));
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t feature_size = input[0].sizes()[1];
  uint32_t feature_nums = input.size();
  std::vector<at::Tensor> output(feature_nums);
  std::vector<T*> input_data(feature_nums);
  std::vector<T*> output_data(feature_nums);
  for (int i = 0; i < feature_nums; i++) {
    output[i] = at::empty({batch_size, feature_size}, input[i].options());
    input_data[i] = input[i].data_ptr<T>();
    output_data[i] = output[i].data_ptr<T>();
  }
  auto interact_feature_size = feature_nums * (feature_nums - 1) / 2;
  auto grad_out_data_line_len = interact_feature_size + feature_size;
  auto grad_out_data = grad_out.data_ptr<T>();

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({feature_nums, feature_nums});
  std::vector<int64_t> lhs_stride({feature_nums, 1});
  std::vector<int64_t> rhs_shape({feature_nums, feature_size});
  std::vector<int64_t> rhs_stride({feature_size, 1});
  std::vector<int64_t> res_shape({feature_nums, feature_size});
  std::vector<int64_t> res_stride({feature_size, 1});
  ideep::tensor::desc lhs_desc(
      std::move(lhs_shape), mkldnn_dtype, std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(
      std::move(rhs_shape), mkldnn_dtype, std::move(rhs_stride));
  ideep::tensor::desc res_desc(
      std::move(res_shape), mkldnn_dtype, std::move(res_stride));

  auto op_attr = dnnl::primitive_attr();
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = ideep::matmul_forward::primitive_desc(
      {lhs_desc, rhs_desc, res_desc}, op_attr, ideep::engine::cpu_engine());

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    auto mm_elems = feature_nums * feature_nums;
    T grad_mm_buf[mm_elems] __attribute__((aligned(64)));
    zero_ker(grad_mm_buf, mm_elems);
    T sum_buf[mm_elems] __attribute__((aligned(64)));
    T grad_cat_buf[feature_nums * feature_size] __attribute__((aligned(64)));
    T cat_buf[feature_nums * feature_size] __attribute__((aligned(64)));
    std::vector<T*> input_ptr(feature_nums);
    std::vector<T*> output_ptr(feature_nums);
    T* grad_out_ptr = &grad_out_data[start * grad_out_data_line_len];
    for (uint32_t n = 0; n < feature_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_size];
      output_ptr[n] = &output_data[n][start * feature_size];
    }
    ideep::tensor lhs({lhs_desc, sum_buf});
    ideep::tensor rhs({lhs_desc, cat_buf});
    ideep::tensor res({res_desc, grad_cat_buf});
    ideep::tensor scratchpad(pd.scratchpad_desc());
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
          grad_out_ptr + feature_size, grad_mm_buf, feature_nums);
      // Calculate gy + gy'
      transpose_add(sum_buf, grad_mm_buf, feature_nums, feature_nums);
      // Calculate A
      cat<T>(cat_buf, input_ptr, feature_size, feature_size);
      p.execute(
          ideep::stream::default_stream(),
          {{DNNL_ARG_SRC, lhs},
           {DNNL_ARG_WEIGHTS, rhs},
           {DNNL_ARG_DST, res},
           {DNNL_ARG_SCRATCHPAD, scratchpad}});
      cat_backward<T, T>(grad_cat_buf, output_ptr, feature_size, feature_size);
      add_ker(output_ptr[0], grad_out_ptr, feature_size);
      grad_out_ptr += grad_out_data_line_len;
      for (uint32_t n = 0; n < feature_nums; n++) {
        input_ptr[n] += feature_size;
        output_ptr[n] += feature_size;
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
  RECORD_FUNCTION(
      "_interaction_forward_bfloat16", c10::ArrayRef<c10::IValue>({}));
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  int32_t feature_size = input[0].sizes()[1];
  int32_t feature_nums = input.size();
  std::vector<at::BFloat16*> input_data(feature_nums);
  for (int i = 0; i < feature_nums; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    TORCH_CHECK(
        input[i].sizes()[1] == feature_size,
        "expect all inputs have same feature size");
    input_data[i] = input[i].data_ptr<at::BFloat16>();
  }
  auto interact_feature_size = feature_nums * (feature_nums - 1) / 2;
  auto out_data_line_len = interact_feature_size + feature_size;
  auto out = at::empty({batch_size, out_data_line_len}, input[0].options());
  auto out_data = out.data_ptr<at::BFloat16>();

  set_tile_config<float, at::BFloat16>(TILE_M, TILE_N, TILE_BK, 2);

  int32_t _AM = ((feature_nums + 31) >> 5) << 5;
  int32_t _AK = ((feature_size + 63) >> 6) << 6; // align to 64
  int32_t A_Stride = _AK * sizeof(at::BFloat16);
  int32_t B_Stride = _AM * sizeof(at::BFloat16) * 2;
  int32_t C_Stride = _AM * sizeof(float);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    const int32_t vector_len = feature_size * sizeof(at::BFloat16);
    float Cmem[_AM][_AM] __attribute__((aligned(64)));
    at::BFloat16 Amem[_AM][_AK] __attribute__((aligned(64)));
    zero_ker(&Amem[0][0], _AM * _AK);
    at::BFloat16 Bmem[_AK >> 1][_AM][2] __attribute__((aligned(64)));

    _tile_loadconfig((const void*)&tc);

    std::vector<at::BFloat16*> input_ptr(feature_nums);
    for (uint32_t n = 0; n < feature_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_size];
      unsigned char* inp = (unsigned char*)(input_ptr[n]);
      for (int cache_line = 0; cache_line < vector_len; cache_line += 64) {
        _mm_prefetch(inp + cache_line, _MM_HINT_T0);
      }
    }
    for (int64_t i = start; i < end; i++) {
      move_ker(&out_data[i * out_data_line_len], input_ptr[0], feature_size);
      cat<at::BFloat16>(&Amem[0][0], input_ptr, feature_size, _AK);
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

      for (uint32_t n = 0; n < feature_nums; n++) {
        input_ptr[n] += feature_size;
        unsigned char* inp = (unsigned char*)(input_ptr[n]);
        for (int cache_line = 0; cache_line < vector_len; cache_line += 64) {
          _mm_prefetch(inp + cache_line, _MM_HINT_T0);
        }
      }

      at::BFloat16* flat_buf =
          (at::BFloat16*)(&out_data[i * out_data_line_len] + feature_size);
      size_t offset = 0;
      for (int i = 1; i < feature_nums; i++) {
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
  RECORD_FUNCTION(
      "_interaction_backward_bfloat16", c10::ArrayRef<c10::IValue>({}));
  int32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  int32_t feature_size = input[0].sizes()[1];
  int32_t feature_nums = input.size();
  std::vector<at::Tensor> output(feature_nums);
  std::vector<at::BFloat16*> input_data(feature_nums);
  std::vector<at::BFloat16*> output_data(feature_nums);
  for (int i = 0; i < feature_nums; i++) {
    output[i] = at::empty({batch_size, feature_size}, input[i].options());
    input_data[i] = input[i].data_ptr<at::BFloat16>();
    output_data[i] = output[i].data_ptr<at::BFloat16>();
  }
  auto interact_feature_size = feature_nums * (feature_nums - 1) / 2;
  auto grad_out_data_line_len = interact_feature_size + feature_size;
  auto grad_out_data = grad_out.data_ptr<at::BFloat16>();

  int32_t _AM = ((feature_nums + 31) >> 5) << 5; // align to 32
  int32_t _AN = ((feature_size + 31) >> 5) << 5; // align to 32
  int32_t _AK = _AM;
  int32_t A_Stride = _AK * sizeof(at::BFloat16);
  int32_t B_Stride = _AN * sizeof(at::BFloat16) * 2;
  int32_t C_Stride = _AN * sizeof(float);
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    const int32_t vector_len = feature_size * sizeof(at::BFloat16);
    auto mm_elems = feature_nums * feature_nums;
    at::BFloat16 grad_mm_buf[mm_elems] __attribute__((aligned(64)));
    zero_ker(grad_mm_buf, mm_elems);
    at::BFloat16 sum_buf[_AM][_AK] __attribute__((aligned(64)));
    zero_ker(&sum_buf[0][0], _AM * _AK);
    at::BFloat16 cat_buf[_AK][_AN] __attribute__((aligned(64)));
    zero_ker(&cat_buf[0][0], _AK * _AN);
    at::BFloat16 Bmem[_AK / 2][_AN][2] __attribute__((aligned(64)));
    float Cmem[_AM][_AN] __attribute__((aligned(64)));

    _tile_loadconfig((const void*)&tc);

    std::vector<at::BFloat16*> input_ptr(feature_nums);
    std::vector<at::BFloat16*> output_ptr(feature_nums);
    for (uint32_t n = 0; n < feature_nums; n++) {
      input_ptr[n] = &input_data[n][start * feature_size];
      output_ptr[n] = &output_data[n][start * feature_size];
      unsigned char* inp = (unsigned char*)(input_ptr[n]);
      for (uint32_t cache_line = 0; cache_line < vector_len; cache_line += 64) {
        _mm_prefetch(inp + cache_line, _MM_HINT_T0);
      }
    }

    at::BFloat16* grad_out_ptr = &grad_out_data[start * grad_out_data_line_len];
    for (int64_t i = start; i < end; i++) {
      flat_triangle_backward<at::BFloat16>(
          grad_out_ptr + feature_size, grad_mm_buf, feature_nums);
      transpose_add(&sum_buf[0][0], grad_mm_buf, feature_nums, _AK);
      cat<at::BFloat16>(&cat_buf[0][0], input_ptr, feature_size, _AN);
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

      for (uint32_t n = 0; n < feature_nums; n++) {
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

      for (uint32_t n = 0; n < feature_nums; n++) {
        input_ptr[n] += feature_size;
        unsigned char* inp = (unsigned char*)(input_ptr[n]);
        for (int cache_line = 0; cache_line < vector_len; cache_line += 64) {
          _mm_prefetch(inp + cache_line, _MM_HINT_T0);
        }
      }

      cat_backward<at::BFloat16, float>(
          &Cmem[0][0], output_ptr, feature_size, _AN);
      add_ker(output_ptr[0], grad_out_ptr, feature_size);
      grad_out_ptr += grad_out_data_line_len;
      for (uint32_t n = 0; n < feature_nums; n++) {
        output_ptr[n] += feature_size;
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

#if defined(CPU_CAPABILITY_AMX)
/**
 *  A fast path if feature_nums == 27 and feature_size == 128 while AMX is
 * enabled (require gcc >=11.2) This function: (1) Assume feature num = 27 and
 * padding it to 32 to align memory unit. (2) Assume feature size = 128, using
 * "<< 7" insteadt of "/128" for performance consideration
 *  TODO: generalize this function to work on feature_nums != 27 and
 * feature_size != 128
 */
void interaction_int8_128_27_amx(
    const at::Tensor& output,
    const std::vector<int8_t*> input_data,
    float* out_in_scales,
    const float dense_scale) {
  const uint8_t _S = 26;
  const uint8_t _S1 = 27;
  const uint8_t _M = 28;
  const uint8_t TILE_M = 14;
  const uint8_t TILE_N = TILE_M;
  const uint8_t TILE_K = 64;
  const uint8_t TILE_BROWS = 16;
  const uint8_t _K = 128;
  const uint8_t LOG2_K = 7;
  // setup size and create output
  const int32_t flat_nums = _S1 * _S / 2;
  const size_t ROW = _K + flat_nums; // 128 + 27 * 26/2
  TORCH_INTERNAL_ASSERT(input_data.size() == _S1);
  TORCH_INTERNAL_ASSERT(output.size(1) == ROW);
  tileconfig_t tc = {0};
  tc.palette_id = 1;
  // tc.startRow = 0;
  //  Configure C tiles
  for (int t = 0; t < 4; ++t) {
    tc.rows[t] = (uint8_t)TILE_M;
    tc.colb[t] = (uint16_t)(TILE_N * sizeof(int32_t));
  }
  // Configure A tiles
  for (int t = 4; t < 6; ++t) {
    tc.rows[t] = (uint8_t)TILE_M;
    tc.colb[t] = (uint16_t)(TILE_K * sizeof(int8_t));
  }
  // Configure B tile. B effectively has 64 rows and 16 columns.
  for (int t = 6; t < 8; ++t) {
    tc.rows[t] = (uint8_t)TILE_BROWS;
    tc.colb[t] = (uint16_t)(TILE_N * 4 * sizeof(int8_t));
  }

  int8_t* res = static_cast<int8_t*>(output.data_ptr());
  bool do_dense_scale = (std::abs(dense_scale - 1.0) > 0.0005);
  auto batch_size = output.size(0);
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    int32_t Cmem[_M][_M] __attribute__((aligned(64)));
    int32_t flat_buf[351] __attribute__((aligned(64)));
    int8_t Amem[_M][_K] __attribute__((aligned(64)));
    int8_t Bmem[_K / 4][_M][4] __attribute__((aligned(64)));
    _tile_loadconfig((const void*)&tc);
    int8_t* local_input_data[_S1];
    int64_t bs_offset = start << LOG2_K;
    for (int i = 0; i < _S1; i++) {
      local_input_data[i] = input_data[i] + bs_offset;
    }

    int8_t* output0_ptr = res + start * ROW;
    for (int i = start; i < end; ++i) {
      if (do_dense_scale) {
        scale_and_move_ker_128(output0_ptr, local_input_data[0], dense_scale);
      } else {
        move_ker(output0_ptr, local_input_data[0], _K);
      }
      load_s8x128_store_aligned_ker(Amem[0], local_input_data[0]);
      int8_t *p0, *p1;
      int j = 1;
      for (; j < (_S1 - 1); j += 2) {
        p0 = local_input_data[j];
        p1 = local_input_data[j + 1];
        load_double_s8x128_store_aligned_ker(Amem[j], p0, Amem[j + 1], p1);
      }

#pragma unroll
      for (int k = 0; k < 32; k++) {
        int32_t ak = (k << 2);
        int n;
#pragma unroll
        for (n = 0; n < _M - 7; n += 8) {
          (*(int32_t*)Bmem[k][n]) = (*(int32_t*)(&Amem[n][ak]));
          (*(int32_t*)Bmem[k][n + 1]) = (*(int32_t*)(&Amem[n + 1][ak]));
          (*(int32_t*)Bmem[k][n + 2]) = (*(int32_t*)(&Amem[n + 2][ak]));
          (*(int32_t*)Bmem[k][n + 3]) = (*(int32_t*)(&Amem[n + 3][ak]));
          (*(int32_t*)Bmem[k][n + 4]) = (*(int32_t*)(&Amem[n + 4][ak]));
          (*(int32_t*)Bmem[k][n + 5]) = (*(int32_t*)(&Amem[n + 5][ak]));
          (*(int32_t*)Bmem[k][n + 6]) = (*(int32_t*)(&Amem[n + 6][ak]));
          (*(int32_t*)Bmem[k][n + 7]) = (*(int32_t*)(&Amem[n + 7][ak]));
        }
#pragma unroll
        for (; n < _M; n++) {
          (*(int32_t*)Bmem[k][n]) = (*(int32_t*)(&Amem[n][ak]));
        }
      }

      _tile_zero(0);
      _tile_zero(2);
      _tile_zero(3);

      _tile_loadd(6, Bmem[0][0], _M * 4 * sizeof(int8_t));

      _tile_loadd(4, &Amem[0][0], _K * sizeof(int8_t));
      _tile_dpbssd(0, 4, 6);

      _tile_loadd(5, &Amem[TILE_M][0], _K * sizeof(int8_t));
      _tile_dpbssd(2, 5, 6);

      _tile_loadd(7, Bmem[0][TILE_N], _M * 4 * sizeof(int8_t));
      _tile_dpbssd(3, 5, 7);

      _tile_loadd(6, Bmem[TILE_BROWS][0], _M * 4 * sizeof(int8_t));

      _tile_loadd(4, &Amem[0][TILE_K], _K * sizeof(int8_t));
      _tile_dpbssd(0, 4, 6);
      _tile_stored(0, &Cmem[0][0], _M * sizeof(int32_t));

      _tile_loadd(5, &Amem[TILE_M][TILE_K], _K * sizeof(int8_t));
      _tile_dpbssd(2, 5, 6);
      _tile_stored(2, &Cmem[TILE_M][0], _M * sizeof(int32_t));

      _tile_loadd(7, Bmem[TILE_BROWS][TILE_N], _M * 4 * sizeof(int8_t));
      _tile_dpbssd(3, 5, 7);
      _tile_stored(3, &Cmem[TILE_M][TILE_N], _M * sizeof(int32_t));

      flat_buf[0] = Cmem[1][0];
      flat_buf[1] = Cmem[2][0];
      flat_buf[2] = Cmem[2][1];
      int32_t offset = 3;
#pragma unroll
      for (int i = 3; i < _S1; i++) {
        move_ker((int32_t*)(&flat_buf[offset]), Cmem[i], i);
        offset += i;
      }

      int8_t* outp = output0_ptr + _K;
      int off;
#pragma unroll
      for (off = 0; off < flat_nums - 63; off += 64) {
        scale_int32_and_store_int8_16x4(
            (outp + off), (flat_buf + off), (out_in_scales + off));
      }
      __m512 scale_m512 = _mm512_load_ps((const void*)(out_in_scales + off));
      scale_int32_and_store_int8_16((outp + off), (flat_buf + off), scale_m512);
      off += 16;
      scale_m512 = _mm512_load_ps((const void*)(out_in_scales + off));
      scale_int32_and_store_int8_maskz_16(
          (outp + off), (flat_buf + off), scale_m512, 0x7fff);

      for (int i = 0; i < _S1; i++) {
        local_input_data[i] += _K;
      }
      output0_ptr += ROW;
    }
  });
  return;
}

#endif

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
  int64_t offset = 1;
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
  int64_t off = 0;
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
  uint32_t feature_nums = input.size();
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t feature_size = input[0].sizes()[1];

  std::vector<float> in_scales(feature_nums);
  std::vector<int8_t*> input_data(feature_nums);
  for (auto i = 0; i < feature_nums; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    TORCH_CHECK(
        input[i].sizes()[1] == feature_size,
        "expect all inputs have same feature size");
    input_data[i] = reinterpret_cast<int8_t*>(input[i].data_ptr<at::qint8>());
    in_scales[i] = at::native::q_scale_quant(input[i]);
  }

  auto interact_feature_size = feature_nums * (feature_nums - 1) / 2;
  auto out_data_line_len = interact_feature_size + feature_size;

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
  for (int i = 1; i < feature_nums; i++) {
    for (int j = 0; j < i; j++) {
      auto input_scale = in_scales[i] * in_scales[j];
      out_in_scales[offset] = input_scale / output_scale;
      offset++;
    }
  }

  float dense_scale = in_scales[0] / output_scale;

#if defined(CPU_CAPABILITY_AMX)
  if (feature_nums == 27 && feature_size == 128) {
    // A fast path if feature_nums == 27 and feature_size == 128 while AMX is
    // enabled (require gcc >=11.2)
    interaction_int8_128_27_amx(output, input_data, out_in_scales, dense_scale);
    return output;
  }
#endif

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    __m512i cat_buf[aligned_off] __attribute__((aligned(64)));
    __m512i convert_to_s16_buf[feature_nums * 4] __attribute__((aligned(64)));
    std::vector<int8_t*> input_addr(feature_nums);
    for (int64_t i = start; i < end; i++) {
      int8_t* out_ptr = &out_data[i * out_data_line_len];
      int8_t* flat_buf = (int8_t*)(out_ptr + feature_size);
      auto row_len = i * feature_size;
#if defined(CPU_CAPABILITY_AVX512)
      if (feature_size == 128) {
        int k = 0;
        for (; k < feature_nums - 1; k += 2) {
          load_s8x128x2_to_s16x128x2(
              &convert_to_s16_buf[k * 4],
              &input_data[k][row_len],
              &input_data[k + 1][row_len]);
        }
        for (; k < feature_nums; k++) {
          load_s8x128_to_s16x128(
              &convert_to_s16_buf[k * 4], &input_data[k][row_len]);
        }
        scale_and_move_ker_128(
            out_ptr, &input_data[0][i * feature_size], dense_scale);
        scale_and_move_ker(
            out_ptr,
            &input_data[0][i * feature_size],
            dense_scale,
            feature_size);
        _interaction_s8s8_scale_s32s8_128(
            flat_buf, feature_nums, out_in_scales, convert_to_s16_buf, cat_buf);
      }
      continue;
#endif
      for (int k = 0; k < feature_nums; k++) {
        input_addr[k] = &input_data[k][row_len];
      }
      scale_and_move_ker(
          out_ptr, &input_data[0][i * feature_size], dense_scale, feature_size);
      _interaction_s8s8_scale_s32s8(
          flat_buf, input_addr, feature_nums, feature_size, out_in_scales);
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
