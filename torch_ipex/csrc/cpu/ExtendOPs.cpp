#include <algorithm>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/record_function.h>
#include "ExtendOPs.h"
#include "bf16/vec/bf16_vec_kernel.h"
#include "dil/dil.hpp"
#include "aten/aten.hpp"
#include "xsmm/libxsmm_utils.h"
#include "../utils.h"
#include "DevOPs.h"
#include "CustomOPs.h"

namespace torch_ipex {

void AtenIpexTypeExt::packed_add_(at::Tensor & top_half, at::Tensor & bot_half, const at::Tensor & grad, float alpha) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bot_half.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bot_half.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half.sizes() == bot_half.sizes());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bot_half.is_contiguous());

  RECORD_FUNCTION("packed_add_", std::vector<c10::IValue>({top_half, bot_half, grad, alpha}), torch::autograd::Node::peek_at_next_sequence_nr());
  if (grad.is_sparse()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half.dim() == 2);
    auto sparse_nnz = grad._nnz();
    auto sparse_dim = grad.sparse_dim();
    auto values = grad._values();
    auto indices = grad._indices();
    auto entry_range = top_half.size(0);
    auto feature_size = values.stride(0);
    auto indices_accessor = indices.accessor<int64_t, 2>();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    auto value_ptr = values.data_ptr<at::BFloat16>();
    auto top_half_ptr = top_half.data_ptr<at::BFloat16>();
    auto bot_half_ptr = bot_half.data_ptr<at::BFloat16>();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(value_ptr != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half_ptr != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bot_half_ptr != nullptr);

    std::vector<int64_t> sparse_stride(sparse_dim);
    for (int64_t d = 0; d < sparse_dim; d++) {
      sparse_stride[d] = top_half.stride(d);
    }

    int32_t max_threads = at::get_num_threads();
    max_threads = (entry_range < max_threads) ? entry_range : max_threads;
    int64_t avg_size = entry_range / max_threads;
    int64_t tail_size = entry_range % max_threads;
    std::vector<int64_t> chunk_size(max_threads, avg_size);
    std::transform(chunk_size.begin(), chunk_size.begin() + tail_size, chunk_size.begin(),
        [](int64_t a) -> int64_t { return a + 1; });
    std::vector<int64_t> acc_chunk_size(max_threads + 1);
    for (int64_t i = 1; i < max_threads + 1; i++) {
      acc_chunk_size[i] = acc_chunk_size[i - 1] + chunk_size[i - 1];
    }

    at::parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
      for (int64_t c = start; c < end; c++) {
        int64_t chunk_begin = acc_chunk_size[c];
        int64_t chunk_end = acc_chunk_size[c + 1];
        for (int64_t n = 0; n < sparse_nnz; n++) {
          int64_t chunk_offset = indices_accessor[0][n];
          if (chunk_offset >= chunk_begin && chunk_offset < chunk_end) {
            int64_t table_offset = 0;
            for (int64_t d = 0; d < sparse_dim; d++) {
              table_offset += sparse_stride[d] * indices_accessor[d][n];
            }
            auto value_index = value_ptr + n * feature_size;
            auto top_half_index = top_half_ptr + table_offset;
            auto bot_half_index = bot_half_ptr + table_offset;
            packed_bf16_add_ker(top_half_index, bot_half_index, value_index, feature_size, alpha);
          }
        }
      }
    });
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad.is_contiguous());
    //TODO: vector implementation basing on vector size
    union packed_bf16 {
      unsigned short s[2];
      float f;
    };

    auto len = top_half.numel();
    auto value_ptr = grad.data_ptr<at::BFloat16>();
    auto top_half_ptr = (unsigned short *)top_half.data_ptr();
    auto bot_half_ptr = (unsigned short *)bot_half.data_ptr();

    at::parallel_for(0, len, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        packed_bf16 p16;
        p16.s[0] = bot_half_ptr[i];
        p16.s[1] = top_half_ptr[i];
        p16.f += alpha * (float)(value_ptr[i]);
        bot_half_ptr[i] = p16.s[0];
        top_half_ptr[i] = p16.s[1];
      }
    });
  }
}

template<typename T>
static inline void cat(const T *in1, const T *in2, T *out, size_t in1_size, size_t in2_size) {
  std::memcpy(out, in1, in1_size * sizeof(T));
  std::memcpy(&out[in1_size], in2, in2_size * sizeof(T));
}

template<typename T>
static inline void cat_backward(const T *in, T *out1, T *out2, size_t out1_size, size_t out2_size) {
  std::memcpy(out1, in, out1_size * sizeof(T));
  std::memcpy(out2, &in[out1_size], out2_size * sizeof(T));
}

template<typename T>
static inline void cat(T *out, const std::vector<T *> &in, const std::vector<uint32_t> &feature_sizes, int64_t bs) {
  size_t offset = 0;
  for (int j = 0; j < feature_sizes.size(); j++) {
    std::memcpy(&out[offset], &in[j][bs * feature_sizes[j]], feature_sizes[j] * sizeof(T));
    offset += feature_sizes[j];
  }
}

template<typename T>
static inline void cat_backward(const T *in, std::vector<T *> &out, const std::vector<uint32_t> &feature_sizes, int64_t bs) {
  size_t offset = 0;
  for (int j = 0; j < feature_sizes.size(); j++) {
    std::memcpy(&out[j][bs * feature_sizes[j]], &in[offset], feature_sizes[j] * sizeof(T));
    offset += feature_sizes[j];
  }
}

template<typename T>
static inline void flat_triangle(const T *in, T *out, size_t size) {
  size_t offset = 0;
  for (int i = 1; i < size; i++) {
    std::memcpy(&out[offset], &in[i * size], i * sizeof(T));
    offset += i;
  }
}

template<typename T>
static inline void flat_triangle_backward(const T *in, T *out, size_t size) {
  size_t offset = 0;
  for (int i = 0; i < size * size; i++) { out[i] = 0.f; }
  for (int i = 1; i < size; i++) {
    std::memcpy(&out[i * size], &in[offset], i * sizeof(T));
    offset += i;
  }
}

template<typename T>
static inline void add(const T *in, T *out, size_t size) {
  #pragma omp simd
  for (size_t i = 0; i < size; i++) { out[i] += in[i]; }
}

static inline void mm_backward(float *out, const float *in1, const float *in2,
    uint32_t vector_nums, uint32_t vector_size, libxsmm_smmfunction mm_ker) {
  // Calculate gy + gy'
  float sum_buf[vector_nums * vector_nums];
  for (int32_t j = 0; j < vector_nums; j++) {
    for (int32_t k = 0; k < vector_nums; k++) {
      sum_buf[j * vector_nums + k] = in1[j * vector_nums + k] + in1[k * vector_nums + j];
    }
  }
  // mm backward
  mm_ker(in2, sum_buf, out);
}

static inline void mm_backward(at::BFloat16 *out, const at::BFloat16 *in1, const at::BFloat16 *in2,
    uint32_t vector_nums, uint32_t vector_size, libxsmm_smmfunction mm_ker) {
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

template<typename T>
inline at::Tensor _interaction_forward(const std::vector<at::Tensor> & input) {
  RECORD_FUNCTION("_interaction_forward", std::vector<c10::IValue>({input}), torch::autograd::Node::peek_at_next_sequence_nr());
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];
  std::vector<uint32_t> feature_sizes(input.size());
  std::vector<T *> input_data(input.size());
  for (int i = 0; i < input.size(); i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].device().is_dpcpp());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[i].dim() == 2);
    feature_sizes[i] = input[i].sizes()[1];
    total_feature_size += input[i].sizes()[1];
    input_data[i] = input[i].data_ptr<T>();
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto tr_vector_size = sizeof(T) == 4 ? vector_size : vector_size / 2;
  auto out = at::empty({batch_size, interact_feature_size + vector_size}, input[0].options());
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
      mm_kernel((xsmm_dtype<T> *)tr_buf, (xsmm_dtype<T> *)cat_buf, (xsmm_dtype<T> *)mm_buf);
      flat_triangle<T>(mm_buf, flat_buf, vector_nums);
      cat<T>(&input_data[0][i * vector_size], flat_buf,
             &out_data[i * (interact_feature_size + vector_size)],
             vector_size, interact_feature_size);
    }
  });

  return out;
}

template<typename T>
inline std::vector<at::Tensor> _interaction_backward(const at::Tensor & grad_out, const std::vector<at::Tensor> & input) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.is_contiguous());
  RECORD_FUNCTION("_interaction_backward", std::vector<c10::IValue>({grad_out, input}), torch::autograd::Node::peek_at_next_sequence_nr());
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
                      grad_input0_buf, grad_flat_buf, vector_size, interact_feature_size);
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
      mm_backward(grad_cat_buf, grad_mm_buf, cat_buf, vector_nums, vector_size, mm_kernel);
      cat_backward<T>(grad_cat_buf, output_data, feature_sizes, i);
      add<T>(grad_input0_buf, &output_data[0][i * vector_size], vector_size);
    }
  });
  return output;
}

at::Tensor AtenIpexTypeExt::interaction_forward(const std::vector<at::Tensor> & input) {
  if (input[0].scalar_type() == at::kFloat) {
    for (const auto &in : input) { TORCH_INTERNAL_ASSERT_DEBUG_ONLY(in.scalar_type() == at::kFloat); }
    return _interaction_forward<float>(input);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input[0].scalar_type() == at::kBFloat16);
    for (const auto &in : input) { TORCH_INTERNAL_ASSERT_DEBUG_ONLY(in.scalar_type() == at::kBFloat16); }
    return _interaction_forward<at::BFloat16>(input);
  }
}

std::vector<at::Tensor> AtenIpexTypeExt::interaction_backward(const at::Tensor & grad_out, const std::vector<at::Tensor> & input) {
  if (grad_out.scalar_type() == at::kFloat) {
    return _interaction_backward<float>(grad_out, input);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.scalar_type() == at::kBFloat16);
    return _interaction_backward<at::BFloat16>(grad_out, input);
  }
}

#if 0
template<typename T>
static inline at::Tensor _embedding_bag_forward(const at::Tensor &weights, const at::Tensor &inputs, const at::Tensor &offsets) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weights.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(offsets.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.dim() == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weights.dim() == 2);
  RECORD_FUNCTION("_embedding_bag_forward", std::vector<c10::IValue>({weights, inputs, offsets}), torch::autograd::Node::peek_at_next_sequence_nr());
  auto batch_size = offsets.size(0);
  auto num_input = inputs.size(0);
  auto vector_size = weights.size(1);
  auto weights_data = weights.data_ptr<T>();
  auto inputs_data = inputs.data_ptr<int64_t>();
  auto offsets_data = offsets.data_ptr<int64_t>();
  auto output = at::empty({batch_size, vector_size}, weights.options());
  auto output_data = output.data_ptr<T>();

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      auto inputs_start = offsets_data[i];
      auto inputs_end = (i < batch_size - 1) ? offsets_data[i + 1] : num_input;
      // TODO: add acc_t support for bag size larger than 1
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs_end - inputs_start == 1);
      auto out_data_ptr = &output_data[i * vector_size];
      #pragma omp simd
      for (int64_t v = 0; v < vector_size; v++) out_data_ptr[v] = 0.0;
      for (int64_t s = inputs_start; s < inputs_end; s++) {
        auto weight_data_ptr = &weights_data[inputs_data[s] * vector_size];
        add_ker((T *)out_data_ptr, (T *)weight_data_ptr, vector_size);
      }
    }
  });
  return output;
}

template<typename T>
static inline at::Tensor _embedding_bag_backward(const at::Tensor &grad_out,
    const at::Tensor &weights, const at::Tensor &inputs, const at::Tensor offsets) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.dim() == 1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.dim() == 2);
  RECORD_FUNCTION("_embedding_bag_backward", std::vector<c10::IValue>({grad_out, weights, inputs, offsets}), torch::autograd::Node::peek_at_next_sequence_nr());
  auto batch_size = offsets.size(0);
  auto num_input = inputs.size(0);
  auto vector_size = weights.size(1);
  auto offsets_data = offsets.data_ptr<int64_t>();
  auto values = at::empty({num_input, vector_size}, weights.options());
  auto values_data = values.data_ptr<T>();
  if (grad_out.is_contiguous()) {
    auto grad_data = grad_out.data_ptr<T>();
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        auto inputs_start = offsets_data[i];
        auto inputs_end = (i < batch_size - 1) ? offsets_data[i + 1] : num_input;
        auto grad_data_ptr = &grad_data[i * vector_size];
        for (int64_t s = inputs_start; s < inputs_end; s++) {
          auto value_data_ptr = &values_data[s * vector_size];
          std::memcpy(value_data_ptr, grad_data_ptr, vector_size * sizeof(T));
        }
      }
    });
  } else {
    auto grad_out_accessor = grad_out.accessor<T, 2>();
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        auto inputs_start = offsets_data[i];
        auto inputs_end = (i < batch_size - 1) ? offsets_data[i + 1] : num_input;
        auto grad_accessor = grad_out_accessor[i];
        for (int64_t s = inputs_start; s < inputs_end; s++) {
          auto value_data_ptr = &values_data[s * vector_size];
          #pragma omp simd
          for (int64_t v = 0; v < vector_size; v++)
            value_data_ptr[v] = grad_accessor[v];
        }
      }
    });
  }
  // TODO:
  auto indices = inputs.reshape({{1, -1}});
  return at::_sparse_coo_tensor_unsafe(indices, values, weights.sizes()); 
}

at::Tensor AtenIpexTypeExt::embedding_bag_forward(const at::Tensor &weights, const at::Tensor &inputs, const at::Tensor &offsets) {
  if (weights.scalar_type() == at::kFloat) {
    return _embedding_bag_forward<float>(weights, inputs, offsets);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weights.scalar_type() == at::kBFloat16);
    return _embedding_bag_forward<at::BFloat16>(weights, inputs, offsets);
  }
}

at::Tensor AtenIpexTypeExt::embedding_bag_backward(const at::Tensor &grad_out,
    const at::Tensor &weights, const at::Tensor &inputs, const at::Tensor &offsets) {
  if (grad_out.scalar_type() == at::kFloat) {
    return _embedding_bag_backward<float>(grad_out, weights, inputs, offsets);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_out.scalar_type() == at::kBFloat16);
    return _embedding_bag_backward<at::BFloat16>(grad_out, weights, inputs, offsets);
  }
}
#endif

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>
AtenIpexTypeExt::embedding_bag_forward(const at::Tensor& weight, const at::Tensor& indices,
  const at::Tensor& offsets, bool scale_grad_by_freq, int64_t mode, bool sparse,
  const c10::optional<at::Tensor>& per_sample_weights, bool include_last_offset) {
  at::Tensor _per_sample_weights;
  if(per_sample_weights.has_value()) {
    _per_sample_weights =  per_sample_weights.value();
  }
  return cpu::aten::embedding_bag::embedding_bag_impl(weight, indices, offsets, scale_grad_by_freq, mode, sparse, _per_sample_weights, include_last_offset);
}

at::Tensor
AtenIpexTypeExt::embedding_bag_backward(const at::Tensor& grad, const at::Tensor& indices,
  const at::Tensor& offsets, const at::Tensor& offset2bag, const at::Tensor& bag_size, const at::Tensor& maximum_indices,
  int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
  const c10::optional<at::Tensor>& per_sample_weights) {
  at::Tensor _per_sample_weights;
  if(per_sample_weights.has_value()) {
    _per_sample_weights =  per_sample_weights.value();
   }
  return cpu::aten::embedding_bag::embedding_bag_backward_impl(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, _per_sample_weights);
}


at::Tensor AtenIpexTypeExt::linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias) {
  if (bias.has_value()) {
    return NewLinearOp::apply(input, weight, bias.value());
  } else {
    return NewLinearOp::apply(input, weight);
  }
}

at::Tensor AtenIpexTypeExt::adaptive_avg_pool2d(at::Tensor const& input, at::IntArrayRef output_size) {
    return NewApaptiveAvgPoolingOp::apply(input, output_size);
}

at::Tensor AtenIpexTypeExt::max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    return NewMaxPool2dOp::apply(input, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenIpexTypeExt::max_pool3d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    return NewMaxPool3dOp::apply(input, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenIpexTypeExt::reshape(const at::Tensor& input, at::IntArrayRef size) {
    return cpu::AtenIpexCPUDev::dil_reshape(input.is_contiguous() ? input : input.contiguous(), size);
}

}  // namespace torch_ipex

namespace {
  static auto dispatch = torch::RegisterOperators()
    .op("torch_ipex::linear", &torch_ipex::AtenIpexTypeExt::linear)
    .op("torch_ipex::max_pool2d", [](const at::Tensor& self, c10::List<int64_t> kernel_size,
      c10::List<int64_t> stride, c10::List<int64_t> padding, c10::List<int64_t> dilation, bool ceil_mode=false){
      return torch_ipex::AtenIpexTypeExt::max_pool2d(self, kernel_size.vec(), stride.vec(), padding.vec(), dilation.vec(), ceil_mode);
    })
    .op("torch_ipex::max_pool3d", [](const at::Tensor& self, c10::List<int64_t> kernel_size,
      c10::List<int64_t> stride, c10::List<int64_t> padding, c10::List<int64_t> dilation, bool ceil_mode=false){
      return torch_ipex::AtenIpexTypeExt::max_pool3d(self, kernel_size.vec(), stride.vec(), padding.vec(), dilation.vec(), ceil_mode);
    })
    .op("torch_ipex::adaptive_avg_pool2d", [](const at::Tensor&self, c10::List<int64_t> output_size) {
      return torch_ipex::AtenIpexTypeExt::adaptive_avg_pool2d(self, output_size.vec());
    });
}