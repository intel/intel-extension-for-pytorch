// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ExtendOPs.h"
#include <torch/csrc/autograd/function.h>
#include "xsmm/libxsmm_utils.h"
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include "bf16/vec/bf16_vec_kernel.h"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include "torch_ipex/csrc/cpu/mkldnn/MKLDNNCommon.h"

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
  ideep::tensor::desc lhs_desc(std::move(lhs_shape), mkldnn_dtype, std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(std::move(rhs_shape), mkldnn_dtype, std::move(rhs_stride));
  ideep::tensor::desc res_desc(std::move(res_shape), mkldnn_dtype, std::move(res_stride));
  auto pd = ideep::matmul_forward::primitive_desc({lhs_desc, rhs_desc, res_desc}, ideep::engine::cpu_engine());

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
        {{DNNL_ARG_SRC, lhs},
        {DNNL_ARG_WEIGHTS, rhs},
        {DNNL_ARG_DST, res}}
      );
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

  auto mkldnn_dtype = cpu::get_mkldnn_dtype(input[0].scalar_type());
  std::vector<int64_t> lhs_shape({vector_nums, vector_nums});
  std::vector<int64_t> lhs_stride({vector_nums, 1});
  std::vector<int64_t> rhs_shape({vector_nums, vector_size});
  std::vector<int64_t> rhs_stride({vector_size, 1});
  std::vector<int64_t> res_shape({vector_nums, vector_size});
  std::vector<int64_t> res_stride({vector_size, 1});
  ideep::tensor::desc lhs_desc(std::move(lhs_shape), mkldnn_dtype, std::move(lhs_stride));
  ideep::tensor::desc rhs_desc(std::move(rhs_shape), mkldnn_dtype, std::move(rhs_stride));
  ideep::tensor::desc res_desc(std::move(res_shape), mkldnn_dtype, std::move(res_stride));
  auto pd = ideep::matmul_forward::primitive_desc({lhs_desc, rhs_desc, res_desc}, ideep::engine::cpu_engine());

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
        {{DNNL_ARG_SRC, lhs},
        {DNNL_ARG_WEIGHTS, rhs},
        {DNNL_ARG_DST, res}}
      );
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
    // todo: move the autograd registion from python into C++.
    // Performance overhead in training here if you use autocast.
    // Because we save the ctx.arg in python before autocast, we have duplicated cast for the input: here and in autocast of the forward path.
#if defined(ENABLE_AUTOCAST_VERBOSE)
  torch_ipex::autocast::verbose::OpNameGuard op_name("interaction_backward");
#endif
    return _interaction_backward<at::BFloat16>(grad_out, torch_ipex::autocast::cpu_cached_cast(at::kBFloat16, input));
  }
}
} // namespace torch_ipex

namespace {
static auto dispatch =
    torch::RegisterOperators()
        .op("torch_ipex::interaction_forward", &torch_ipex::AtenIpexTypeExt::interaction_forward)
        .op("torch_ipex::interaction_backward", &torch_ipex::AtenIpexTypeExt::interaction_backward);
}

namespace torch_ipex {
namespace autocast {

at::Tensor interaction_forward(const std::vector<at::Tensor> &input){
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::interaction_forward", "")
    .typed<decltype(interaction_forward)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("interaction_forward");
#endif
  auto type = promote_type(at::kBFloat16, input);
  return op.call(cpu_cached_cast(type, input));
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m){
  m.impl("interaction_forward", torch_ipex::autocast::interaction_forward);
}

}}
