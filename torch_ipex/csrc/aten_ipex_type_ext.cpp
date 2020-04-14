#include <algorithm>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/record_function.h>
#include "aten_ipex_type_ext.h"
#include "cpu/vec/bf16_vec_kernel.h"
#include "cpu/dil/dil.hpp"
#include "utils.h"

namespace torch_ipex {

void AtenIpexTypeExt::packed_add_(at::Tensor & top_half, at::Tensor & bot_half, const at::Tensor & grad, float alpha) {
  TORCH_INTERNAL_ASSERT(grad.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(top_half.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(bot_half.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT(top_half.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT(bot_half.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT(top_half.sizes() == bot_half.sizes());
  TORCH_INTERNAL_ASSERT(top_half.dim() == 2);
  TORCH_INTERNAL_ASSERT(top_half.is_contiguous());
  TORCH_INTERNAL_ASSERT(bot_half.is_contiguous());

  RECORD_FUNCTION("packed_add_", std::vector<c10::IValue>({top_half, bot_half, grad, alpha}), torch::autograd::Node::peek_at_next_sequence_nr());
  if (grad.is_sparse()) {
    auto sparse_nnz = grad._nnz();
    auto sparse_dim = grad.sparse_dim();
    auto values = grad._values();
    auto indices = grad._indices();
    auto entry_range = top_half.size(0);
    auto feature_size = values.stride(0);
    auto indices_accessor = indices.accessor<int64_t, 2>();

    TORCH_INTERNAL_ASSERT(values.is_contiguous());
    auto value_ptr = values.data_ptr<at::BFloat16>();
    auto top_half_ptr = top_half.data_ptr<at::BFloat16>();
    auto bot_half_ptr = bot_half.data_ptr<at::BFloat16>();

    TORCH_INTERNAL_ASSERT(value_ptr != nullptr);
    TORCH_INTERNAL_ASSERT(top_half_ptr != nullptr);
    TORCH_INTERNAL_ASSERT(bot_half_ptr != nullptr);

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
    TORCH_INTERNAL_ASSERT(grad.is_contiguous());
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
static inline void flat_triangle(const T *in, T *out, size_t size) {
  size_t offset = 0;
  for (int i = 1; i < size; i++) {
    std::memcpy(&out[offset], &in[i * size], i * sizeof(T));
    offset += i;
  }
}

template<typename T>
inline at::Tensor _interaction_forward(const std::vector<at::Tensor> & input) {
  uint32_t total_feature_size = 0;
  int64_t batch_size = input[0].sizes()[0];
  uint32_t vector_size = input[0].sizes()[1];
  std::vector<uint32_t> feature_sizes(input.size());
  std::vector<T *> input_data;
  for (int i = 0; i < input.size(); i++) {
    TORCH_INTERNAL_ASSERT(input[i].device().is_dpcpp());
    TORCH_INTERNAL_ASSERT(input[i].dim() == 2);
    feature_sizes[i] = input[i].sizes()[1];
    total_feature_size += input[i].sizes()[1];
    input_data.push_back(input[i].data_ptr<T>());
  }
  auto vector_nums = total_feature_size / vector_size;
  TORCH_INTERNAL_ASSERT(total_feature_size % vector_size == 0);
  auto interact_feature_size = vector_nums * (vector_nums - 1) / 2;
  auto out = at::empty({batch_size, interact_feature_size + vector_size}, input[0].options());
  auto out_data = out.data_ptr<T>();

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      T cat_buf[vector_nums * vector_size];
      T mm_buf[vector_nums * vector_nums];
      T flat_buf[interact_feature_size];
      uint32_t offset = 0;
      for (int j = 0; j < input.size(); j++) {
        std::memcpy(&cat_buf[offset], &input_data[j][i * feature_sizes[j]], feature_sizes[j] * sizeof(T));
        offset += feature_sizes[j];
      }

      dil::tensor trs_in{{vector_nums, vector_size}, get_dil_data_type(input[0].scalar_type()), cat_buf};
      dil::tensor mm_out{{vector_nums, vector_nums}, get_dil_data_type(input[0].scalar_type()), mm_buf};
      auto trs_out = trs_in.transpose(0, 1);
      dil::matmul_forward::compute(trs_in, trs_out, mm_out);
      flat_triangle<T>(mm_buf, flat_buf, vector_nums);
      cat<T>(&input_data[0][i * vector_size], flat_buf,
             &out_data[i * (interact_feature_size + vector_size)],
             vector_size, interact_feature_size);
    }
  });

  return out;
}

at::Tensor AtenIpexTypeExt::interaction_forward(const std::vector<at::Tensor> & input) {
  if (input[0].scalar_type() == at::kFloat) {
    for (const auto &in : input) { TORCH_INTERNAL_ASSERT(in.scalar_type() == at::kFloat); }
    return _interaction_forward<float>(input);
  } else {
    for (const auto &in : input) { TORCH_INTERNAL_ASSERT(in.scalar_type() == at::kBFloat16); }
    return _interaction_forward<at::BFloat16>(input);
  }
}

std::vector<at::Tensor> AtenIpexTypeExt::interaction_backward(const at::Tensor & grad_out, const std::vector<at::Tensor> & input) {
  return input;
}

}  // namespace torch_ipex

