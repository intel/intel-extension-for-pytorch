// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ExtendOPs.h"
#include <torch/csrc/autograd/function.h>
#include "xsmm/libxsmm_utils.h"
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include "bf16/vec/bf16_vec_kernel.h"

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
    //std::memcpy(&out[j][bs * feature_sizes[j]], &in[offset], feature_sizes[j] * sizeof(T));
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

static inline void mm_backward(float *out, const float *in1, const float *in2,
                               uint32_t vector_nums, uint32_t vector_size,
                               libxsmm_smmfunction mm_ker) {
  // Calculate gy + gy'
  float sum_buf[vector_nums * vector_nums];
  for (int32_t j = 0; j < vector_nums; j++) {
    for (int32_t k = 0; k < vector_nums; k++) {
      sum_buf[j * vector_nums + k] =
          in1[j * vector_nums + k] + in1[k * vector_nums + j];
    }
  }
  // mm backward
  mm_ker(in2, sum_buf, out);
}

static inline void mm_backward(at::BFloat16 *out, const at::BFloat16 *in1,
                               const at::BFloat16 *in2, uint32_t vector_nums,
                               uint32_t vector_size,
                               libxsmm_smmfunction mm_ker) {
  float tmp_in1[vector_nums * vector_nums];
  float tmp_in2[vector_nums * vector_size];
  float tmp_out[vector_nums * vector_size];

  cvt_bf16_to_fp32(tmp_in1, in1, vector_nums * vector_nums);
  cvt_bf16_to_fp32(tmp_in2, in2, vector_nums * vector_size);
  // Calculate gy + gy'
  for (int32_t j = 0; j < vector_nums; j++) {
    for (int32_t k = 0; k < vector_nums; k++) {
      tmp_in1[j * vector_nums + k] += tmp_in1[k * vector_nums + j];
    }
  }
  // mm backward w/ fp32
  mm_ker(tmp_in2, tmp_in1, tmp_out);
  cvt_fp32_to_bf16(out, tmp_out, vector_nums * vector_size);
}

template <typename T>
inline at::Tensor _interaction_forward(const std::vector<at::Tensor> &input) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("_interaction_forward", std::vector<c10::IValue>({}));
#endif
    printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
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
  auto tr_vector_size = sizeof(T) == 4 ? vector_size : vector_size / 2;
  auto out = at::empty({batch_size, interact_feature_size + vector_size},
                       input[0].options());
  auto out_data = out.data_ptr<T>();

  auto mm_kernel = get_mm_kernel<T>(vector_nums, vector_nums, vector_size);
  auto tr_kernel = get_tr_kernel(tr_vector_size, vector_nums, vector_nums);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    T cat_buf[vector_nums * vector_size];
    T tr_buf[vector_nums * vector_size];
    T mm_buf[vector_nums * vector_nums];
    T flat_buf[interact_feature_size];
    for (int64_t i = start; i < end; i++) {
      cat<T>(cat_buf, input_data, feature_sizes, i);
      tr_kernel(cat_buf, &tr_vector_size, tr_buf, &vector_nums);
      mm_kernel((xsmm_dtype<T> *)tr_buf, (xsmm_dtype<T> *)cat_buf,
                (xsmm_dtype<T> *)mm_buf);
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
  RECORD_FUNCTION("_interaction_backward",
                  std::vector<c10::IValue>({}));
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

  auto mm_kernel = get_mm_kernel<float>(vector_nums, vector_size, vector_nums);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    T grad_input0_buf[vector_size];
    T grad_flat_buf[interact_feature_size];
    T grad_mm_buf[vector_nums * vector_nums];
    T grad_cat_buf[vector_nums * vector_size];
    T cat_buf[vector_nums * vector_size];
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
      mm_backward(grad_cat_buf, grad_mm_buf, cat_buf, vector_nums, vector_size,
                  mm_kernel);
      cat_backward<T>(grad_cat_buf, output_data, feature_sizes, i);
      add_ker(grad_input0_buf, &output_data[0][i * vector_size], vector_size);
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
    return _interaction_backward<float>(grad_out, input);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.scalar_type() == at::kBFloat16);
    return _interaction_backward<at::BFloat16>(grad_out, input);
  }
}
} // namespace torch_ipex

namespace {
static auto dispatch =
    torch::RegisterOperators()
        .op("torch_ipex::interaction_forward", &torch_ipex::AtenIpexTypeExt::interaction_forward)
        .op("torch_ipex::interaction_backward", &torch_ipex::AtenIpexTypeExt::interaction_backward);
}
