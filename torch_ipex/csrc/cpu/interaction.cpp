// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ExtendOPs.h"
#include "bf16/vec/bf16_vec_kernel.h"
#include "int8/vec/int8_vec_kernel.h"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include "torch_ipex/csrc/cpu/CustomOPs.h"
#include "torch_ipex/csrc/cpu/mkldnn/MKLDNNCommon.h"
#include "torch_ipex/csrc/quantization/AutoCast.hpp"
#include "xsmm/libxsmm_utils.h"
#include <ATen/Parallel.h>
#include <ATen/quantized/Quantizer.h>
#include <algorithm>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>

/*
 Custom op to optimize DLRM interaction part
*/

namespace torch_ipex {

template <typename T>
static inline void cat(const T *in1, const T *in2, T *out, size_t in1_size,
                       size_t in2_size) {
  move_ker(out, in1, in1_size);
  move_ker(&out[in1_size], in2, in2_size);
}

template <typename T>
static inline void cat_backward(const T *in, T *out1, T *out2, size_t out1_size,
                                size_t out2_size) {
  move_ker(out1, in, out1_size);
  move_ker(out2, &in[out1_size], out2_size);
}

template <typename T>
static inline void cat(T *out, const std::vector<T *> &in,
                       const std::vector<uint32_t> &feature_sizes, int64_t bs) {
  size_t offset = 0;
  for (int j = 0; j < feature_sizes.size(); j++) {
    move_ker(&out[offset], &in[j][bs * feature_sizes[j]], feature_sizes[j]);
    offset += feature_sizes[j];
  }
}

template <typename T>
static inline void cat_backward(const T *in, std::vector<T *> &out,
                                const std::vector<uint32_t> &feature_sizes,
                                int64_t bs) {
  size_t offset = 0;
  for (int j = 0; j < feature_sizes.size(); j++) {
    // std::memcpy(&out[j][bs * feature_sizes[j]], &in[offset], feature_sizes[j]
    // * sizeof(T));
    move_ker(&out[j][bs * feature_sizes[j]], &in[offset], feature_sizes[j]);
    offset += feature_sizes[j];
  }
}

template <typename T>
static inline void flat_triangle(const T *in, T *out, size_t size) {
  size_t offset = 0;
  for (int i = 1; i < size; i++) {
    move_ker(&out[offset], &in[i * size], i);
    offset += i;
  }
}

template <typename T>
static inline void flat_triangle_backward(const T *in, T *out, size_t size) {
  size_t offset = 0;
  for (int i = 0; i < size * size; i++) {
    out[i] = 0.f;
  }
  for (int i = 1; i < size; i++) {
    move_ker(&out[i * size], &in[offset], i);
    offset += i;
  }
}

template <typename T>
static inline void transpose_add(T *out, const T *in, uint32_t vector_nums) {
  for (int32_t j = 0; j < vector_nums; j++) {
    for (int32_t k = 0; k < vector_nums; k++) {
      out[j * vector_nums + k] =
          in[j * vector_nums + k] + in[k * vector_nums + j];
    }
  }
}

template <typename T>
inline at::Tensor _interaction_forward(const std::vector<at::Tensor> &input) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("_interaction_forward", std::vector<c10::IValue>({}));
#endif
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];
  std::vector<uint32_t> feature_sizes(input.size());
  std::vector<T *> input_data(input.size());
  for (int i = 0; i < input.size(); i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    feature_sizes[i] = input[i].sizes()[1];
    total_feature_size += input[i].sizes()[1];
    input_data[i] = input[i].data_ptr<T>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto out = at::empty({batch_size, interact_feature_size + vector_size},
                       input[0].options());
  auto out_data = out.data_ptr<T>();

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({vector_nums, vector_size});
  std::vector<int64_t> lhs_stride({vector_size, 1});
  std::vector<int64_t> rhs_shape({vector_size, vector_nums});
  std::vector<int64_t> rhs_stride({1, vector_size});
  std::vector<int64_t> res_shape({vector_nums, vector_nums});
  std::vector<int64_t> res_stride({vector_nums, 1});
  ideep::tensor::desc lhs_desc(std::move(lhs_shape), mkldnn_dtype,
                               std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(std::move(rhs_shape), mkldnn_dtype,
                               std::move(rhs_stride));
  ideep::tensor::desc res_desc(std::move(res_shape), mkldnn_dtype,
                               std::move(res_stride));
  auto pd = ideep::matmul_forward::primitive_desc(
      {lhs_desc, rhs_desc, res_desc}, ideep::engine::cpu_engine());

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    T cat_buf[vector_nums * vector_size];
    ideep::tensor lhs({lhs_desc, cat_buf});
    ideep::tensor rhs({lhs_desc, cat_buf});
    T mm_buf[vector_nums * vector_nums];
    ideep::tensor res({res_desc, mm_buf});
    T flat_buf[interact_feature_size];
    auto p = dnnl::matmul(pd);
    for (int64_t i = start; i < end; i++) {
      cat<T>(cat_buf, input_data, feature_sizes, i);
      p.execute(
          ideep::stream::default_stream(),
          {{DNNL_ARG_SRC, lhs}, {DNNL_ARG_WEIGHTS, rhs}, {DNNL_ARG_DST, res}});
      flat_triangle<T>(mm_buf, flat_buf, vector_nums);
      cat<T>(&input_data[0][i * vector_size], flat_buf,
             &out_data[i * (interact_feature_size + vector_size)], vector_size,
             interact_feature_size);
    }
  });

  return out;
}

template <typename T>
inline std::vector<at::Tensor>
_interaction_backward(const at::Tensor &grad_out,
                      const std::vector<at::Tensor> &input) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.is_contiguous());
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("_interaction_backward", std::vector<c10::IValue>({}));
#endif
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];
  std::vector<uint32_t> feature_sizes(input.size());
  std::vector<at::Tensor> output(input.size());
  std::vector<T *> input_data(input.size());
  std::vector<T *> output_data(input.size());
  for (int i = 0; i < input.size(); i++) {
    auto feature_size = input[i].sizes()[1];
    feature_sizes[i] = feature_size;
    total_feature_size += feature_size;
    output[i] = at::empty({batch_size, feature_size}, input[i].options());
    input_data[i] = input[i].data_ptr<T>();
    output_data[i] = output[i].data_ptr<T>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto grad_out_data = grad_out.data_ptr<T>();

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({vector_nums, vector_nums});
  std::vector<int64_t> lhs_stride({vector_nums, 1});
  std::vector<int64_t> rhs_shape({vector_nums, vector_size});
  std::vector<int64_t> rhs_stride({vector_size, 1});
  std::vector<int64_t> res_shape({vector_nums, vector_size});
  std::vector<int64_t> res_stride({vector_size, 1});
  ideep::tensor::desc lhs_desc(std::move(lhs_shape), mkldnn_dtype,
                               std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(std::move(rhs_shape), mkldnn_dtype,
                               std::move(rhs_stride));
  ideep::tensor::desc res_desc(std::move(res_shape), mkldnn_dtype,
                               std::move(res_stride));
  auto pd = ideep::matmul_forward::primitive_desc(
      {lhs_desc, rhs_desc, res_desc}, ideep::engine::cpu_engine());

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    T grad_input0_buf[vector_size];
    T grad_flat_buf[interact_feature_size];
    T grad_mm_buf[vector_nums * vector_nums];
    T sum_buf[vector_nums * vector_nums];
    T grad_cat_buf[vector_nums * vector_size];
    T cat_buf[vector_nums * vector_size];
    ideep::tensor lhs({lhs_desc, sum_buf});
    ideep::tensor rhs({lhs_desc, cat_buf});
    ideep::tensor res({res_desc, grad_cat_buf});
    auto p = dnnl::matmul(pd);
    for (int64_t i = start; i < end; i++) {
      cat_backward<T>(&grad_out_data[i * (interact_feature_size + vector_size)],
                      grad_input0_buf, grad_flat_buf, vector_size,
                      interact_feature_size);
      flat_triangle_backward<T>(grad_flat_buf, grad_mm_buf, vector_nums);

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

      // Calculate A
      cat<T>(cat_buf, input_data, feature_sizes, i);
      // Calculate gy + gy'
      transpose_add(sum_buf, grad_mm_buf, vector_nums);
      p.execute(
          ideep::stream::default_stream(),
          {{DNNL_ARG_SRC, lhs}, {DNNL_ARG_WEIGHTS, rhs}, {DNNL_ARG_DST, res}});
      cat_backward<T>(grad_cat_buf, output_data, feature_sizes, i);
      add_ker(&output_data[0][i * vector_size], grad_input0_buf, vector_size);
    }
  });
  return output;
}

at::Tensor
AtenIpexTypeExt::interaction_forward(const std::vector<at::Tensor> &input) {
  if (input[0].scalar_type() == at::kFloat) {
    for (auto &in : input) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(in.scalar_type() == at::kFloat);
    }
    return _interaction_forward<float>(input);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[0].scalar_type() == at::kBFloat16);
    for (const auto &in : input) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(in.scalar_type() == at::kBFloat16);
    }
    return _interaction_forward<at::BFloat16>(input);
  }
}

std::vector<at::Tensor>
AtenIpexTypeExt::interaction_backward(const at::Tensor &grad_out,
                                      const std::vector<at::Tensor> &input) {
  if (grad_out.scalar_type() == at::kFloat) {
    return _interaction_backward<float>(
        grad_out, torch_ipex::autocast::cpu_cached_cast(at::kFloat, input));
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.scalar_type() == at::kBFloat16);
    // todo: move the autograd registion from python into C++.
    // Performance overhead in training here if you use autocast.
    // Because we save the ctx.arg in python before autocast, we have duplicated
    // cast for the input: here and in autocast of the forward path.
#if defined(ENABLE_AUTOCAST_VERBOSE)
    torch_ipex::autocast::verbose::OpNameGuard op_name("interaction_backward");
#endif
    return _interaction_backward<at::BFloat16>(
        grad_out, torch_ipex::autocast::cpu_cached_cast(at::kBFloat16, input));
  }
}

namespace cpu {
static inline void _interaction_s8s8_scale_s32s8_128(
    int8_t *out, size_t M, const float *__attribute__((aligned(64))) scales,
    __m512i *convert_to_s16_buf, __m512i *cat_buf) {
  auto *a = (const __m512i *)&convert_to_s16_buf[0];
  auto *b = (const __m512i *)&convert_to_s16_buf[4];
  mul_and_sum_s16x128_to_s32x16(cat_buf[0], b, a);
  size_t offset = 1;
  for (int i = 2; i < M; i++) {
    auto *c = (const __m512i *)&convert_to_s16_buf[i * 4];
    int j = 0;
    for (; j < i - 1; j += 2) {
      a = (const __m512i *)&convert_to_s16_buf[j * 4];
      b = (const __m512i *)&convert_to_s16_buf[j * 4 + 4];
      mul_and_sum_s16x128x2_to_s32x16x2(cat_buf[offset], cat_buf[offset + 1], c,
                                        a, c, b);
      offset += 2;
    }
    for (; j < i; j++) {
      a = (const __m512i *)&convert_to_s16_buf[j * 4];
      mul_and_sum_s16x128_to_s32x16(cat_buf[offset], c, a);
      offset++;
    }
  }

  // Do reduce add with scale
  size_t off = 0;
  for (; off < offset - 15; off += 16) {
    __m512 scale_m512 = _mm512_load_ps((const void *)(scales + off));
    reduce_add_s32x16x16_with_scales(out + off, cat_buf + off, scale_m512);
  }
  __m512 scale_m512 = _mm512_load_ps((const void *)(scales + off));
  auto mask = ((1 << (offset - off)) - 1);
  reduce_add_s32x16x16_with_scales_and_mask_store(out + off, mask,
                                                  cat_buf + off, scale_m512);
}

static inline void
_interaction_s8s8_scale_s32s8(int8_t *out,
                              const std::vector<int8_t *> &input_addr, size_t M,
                              size_t K, float *scales) {
  size_t offset = 0;
  for (int i = 1; i < M; i++) {
    int8_t *a = input_addr[i];
    for (int j = 0; j < i; j++) {
      int8_t *b = input_addr[j];
      out[offset] = _dot_s8s8_scale_s32s8(a, b, K, scales[offset]);
      offset++;
    }
  }
}

at::Tensor AtenIpexJITDev::dil_qinteraction(const std::vector<at::Tensor> input,
                                            double output_scale, int64_t o_zp,
                                            at::ScalarType o_dtype) {
  uint32_t input_size = input.size();
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];

  std::vector<float> in_scales(input_size);
  std::vector<uint32_t> feature_sizes(input_size);
  std::vector<int8_t *> input_data(input_size);
  for (auto i = 0; i < input_size; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    auto cur_input = input[i];
    input_data[i] = reinterpret_cast<int8_t *>(cur_input.data_ptr<at::qint8>());
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
  at::Tensor output = at::new_qtensor(/*sizes=*/{batch_size, out_data_line_len},
                                      input[0].options(), output_quantizer);
  int8_t *out_data = reinterpret_cast<int8_t *>(output.data_ptr<at::qint8>());
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
    std::vector<int8_t *> input_addr(vector_nums);
    for (int64_t i = start; i < end; i++) {
      int8_t *out_ptr = &out_data[i * out_data_line_len];
      int8_t *flat_buf = (int8_t *)(out_ptr + vector_size);
      auto row_len = i * vector_size;
      if (vector_size == 128) {
        int k = 0;
        for (; k < vector_nums - 1; k += 2) {
          load_s8x128x2_to_s16x128x2(&convert_to_s16_buf[k * 4],
                                     &input_data[k][row_len],
                                     &input_data[k + 1][row_len]);
        }
        for (; k < vector_nums; k++) {
          load_s8x128_to_s16x128(&convert_to_s16_buf[k * 4],
                                 &input_data[k][row_len]);
        }
        scale_and_move_ker_128(out_ptr, &input_data[0][i * vector_size],
                               dense_scale);
        _interaction_s8s8_scale_s32s8_128(flat_buf, vector_nums, out_in_scales,
                                          convert_to_s16_buf, cat_buf);
      } else {
        for (int k = 0; k < vector_nums; k++) {
          input_addr[k] = &input_data[k][row_len];
        }
        scale_and_move_ker(out_ptr, &input_data[0][i * vector_size],
                           dense_scale, vector_size);
        _interaction_s8s8_scale_s32s8(flat_buf, input_addr, vector_nums,
                                      vector_size, out_in_scales);
      }
    }
  });

  return output;
}

} // namespace cpu
} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      torch::schema("torch_ipex::interaction_forward(Tensor[] input) -> Tensor",
                    c10::AliasAnalysisKind::PURE_FUNCTION),
      torch_ipex::AtenIpexTypeExt::interaction_forward);
  m.def(torch::schema("torch_ipex::interaction_backward(Tensor grad_out, "
                      "Tensor[] input) -> Tensor[]",
                      c10::AliasAnalysisKind::PURE_FUNCTION),
        torch_ipex::AtenIpexTypeExt::interaction_backward);
}
}

namespace torch_ipex {
namespace autocast {

at::Tensor interaction_forward(const std::vector<at::Tensor> &input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::interaction_forward", "")
                       .typed<decltype(interaction_forward)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("interaction_forward");
#endif

  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::interaction_forward(input);
  }

  auto type = promote_type(get_autocast_dtype(), input);
  return op.call(cpu_cached_cast(type, input));
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("interaction_forward", torch_ipex::autocast::interaction_forward);
}

} // namespace autocast
} // namespace torch_ipex
