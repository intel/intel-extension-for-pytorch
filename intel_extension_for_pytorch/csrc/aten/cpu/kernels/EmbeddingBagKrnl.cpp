#include <csrc/aten/cpu/EmbeddingBag.h>
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/vec512/bf16/vec/bf16_vec_kernel.h"
#include "csrc/cpu/vec512/int8/vec/int8_vec_kernel.h"
#include "csrc/jit/cpu/kernels/Embeddingbag.h"
#include "csrc/quantization/AutoCast.hpp"
#include "csrc/utils/rw_lock.h"

#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include <algorithm>

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

static inline void make_offset2bag(
    const at::Tensor& offsets,
    const at::Tensor& indices,
    at::Tensor& offset2bag) {
  offset2bag.index_add_(
      0,
      offsets,
      at::ones_like(offsets, offsets.options())); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1; // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0); // offset2bag = [0 0 1 1 2]
}

// To save compute, if we are going to go down the fast path case for the 'sum'
// mode, we skip calculating offset2bag, since it is not going to be used.
static inline bool is_bfloat16_tensor(const at::Tensor tensor_) {
  if (tensor_.scalar_type() == at::kBFloat16)
    return true;
  return false;
}

template <typename T>
static inline at::Tensor _embedding_bag_index_add_select_fast(
    const at::Tensor indices,
    const at::Tensor src,
    const at::Tensor offsets,
    bool include_last_offset) {
  int64_t ddim = src.size(1);
  auto* src_data = src.data_ptr<T>();
  int64_t output_size = offsets.numel() - 1;
  int64_t* offsets_data = offsets.data_ptr<int64_t>();
  std::vector<int64_t> offsets_include_last;

  if (!include_last_offset) {
    output_size = offsets.numel();
    offsets_include_last.resize(output_size + 1);
    int64_t* offsets_include_last_data = offsets_include_last.data();
    int64_t iter_time = (output_size >> 5);
    int64_t align32_size = (iter_time << 5);
    int64_t left_size = output_size - align32_size;
    // std::memcpy(offsets_include_last.data(), offsets_data, sizeof(int64_t) *
    // output_size);
    at::parallel_for(0, iter_time, 16, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i += 1) {
        auto start_offset = i << 5;
        move_ker(
            &offsets_include_last_data[start_offset],
            &offsets_data[start_offset],
            32);
      }
    });
    if (left_size > 0) {
      move_ker(
          &offsets_include_last_data[align32_size],
          &offsets_data[align32_size],
          left_size);
    }
    offsets_include_last[output_size] = indices.numel();
    offsets_data = offsets_include_last.data();
  }

  at::Tensor output = at::empty({output_size, src.size(1)}, src.options());
  auto* output_data = output.data_ptr<T>();
  auto indices_accessor = indices.accessor<int64_t, 1>();
  at::parallel_for(0, output_size, 16, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      auto* out_data_ptr = &output_data[i * ddim];
      zero_ker((T*)out_data_ptr, ddim);
      auto inputs_start = offsets_data[i];
      auto inputs_end = offsets_data[i + 1];
      for (int64_t s = inputs_start; s < inputs_end; s++) {
        T* select_data_ptr = &src_data[indices_accessor[s] * ddim];
        add_ker((T*)out_data_ptr, (T*)select_data_ptr, ddim);
      }
    }
  });

  return output;
}

at::Tensor embedding_bag_kernel_impl(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool include_last_offset) {
  at::Tensor offsets_ =
      offsets.is_contiguous() ? offsets : offsets.contiguous();

  at::Tensor output;
  if (is_bfloat16_tensor(weight)) {
    output = _embedding_bag_index_add_select_fast<at::BFloat16>(
        indices, weight, offsets_, include_last_offset);
  } else {
    output = _embedding_bag_index_add_select_fast<float>(
        indices, weight, offsets_, include_last_offset);
  }
  return output;
}

static inline at::Tensor expand_values_if_needed(const at::Tensor& values) {
  // expand
  if (values.dim() == 0) {
    // Mimic Numpy behavior here and treat it as a 1D tensor
    return values.expand({1});
  }

  return values;
}

static inline at::Tensor _sparse_coo_tensor_unsafe(
    const at::Tensor& indices,
    const at::Tensor& values_,
    c10::ArrayRef<int64_t> size,
    const at::TensorOptions& options) {
  at::Tensor values = expand_values_if_needed(values_);
  assert(options.has_layout() && options.layout() == c10::kSparse);
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;
  return at::native::new_with_dims_and_tensor_sparse(
      sparse_dim,
      dense_dim,
      size,
      indices,
      values,
      values.scalar_type(),
      c10::kSparse,
      values.device());
}

template <typename T>
static inline at::Tensor embedding_bag_sparse_backward_sum_fast(
    const at::Tensor grad,
    const at::Tensor indices,
    const at::Tensor offsets,
    int num_weights) {
  assert(grad.stride(1) == 1);

  int64_t indices_size0 = indices.size(0);
  int64_t ddim = grad.size(1);
  at::Tensor index_grad = at::empty({indices_size0, ddim}, grad.options());
  int grad_stride0 = grad.stride(0);

  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  auto offset_numel = offsets.numel();

  T* gradout_data = index_grad.data_ptr<T>();
  T* grad_data = grad.data_ptr<T>();
  at::parallel_for(0, offset_numel, 16, [&](int64_t start, int64_t end) {
    for (auto mb = start; mb < end; mb++) {
      int64_t select_off_start = offsets_accessor[mb];
      int64_t select_off_end =
          (mb < (offset_numel - 1) ? offsets_accessor[mb + 1] : indices_size0);
      auto grad_block = grad_data + grad_stride0 * mb;
      for (int64_t s = select_off_start; s < select_off_end; s++) {
        move_ker((T*)(gradout_data + ddim * s), (T*)grad_block, ddim);
      }
    }
  });

  int64_t num_features = index_grad.size(-1);
  auto weight_size = std::array<int64_t, 2>{{num_weights, num_features}};
  auto dense_options = index_grad.options();

  if (index_grad.numel() == 0) {
    return _sparse_coo_tensor_unsafe(
        at::empty({1, 0}, indices.options()),
        at::empty({0, num_features}, dense_options),
        weight_size);
  }

  auto index = indices.reshape({1, -1});
  auto values = index_grad.reshape({-1, num_features});

  return _sparse_coo_tensor_unsafe(index, values, weight_size);
}

static inline int64_t count_and_map_uniq(
    const at::TensorAccessor<int64_t, 1>& indices_accessor,
    int64_t indices_length,
    std::vector<int64_t>& indices_to_index,
    std::vector<int64_t>& index_to_indices) {
  int64_t u = 0;
  for (int64_t i = 0; i < indices_length; i++) {
    int64_t indices = indices_accessor[i];
    if (indices_to_index[indices] == -1ull) {
      indices_to_index[indices] = u;
      index_to_indices[u] = indices;
      u++;
    }
  }
  return u;
}

template <typename T>
static inline at::Tensor embedding_bag_dense_backward_sum_fast(
    const at::Tensor grad,
    const at::Tensor indices,
    const at::Tensor offsets,
    int num_weights) {
  int64_t indices_numel = indices.numel();
  assert(grad.stride(1) == 1 && indices_numel > 0);
  auto offset_numel = offsets.numel();
  at::Tensor offset2bag_;
  if (offset_numel != indices_numel) {
    offset2bag_ =
        at::native::full({indices.sizes()[0] + 1}, 0, indices.scalar_type());
    make_offset2bag(offsets, indices, offset2bag_);
    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    offset2bag_ = offsets;
  }
  auto indices_accessor = indices.accessor<int64_t, 1>();
  std::vector<int64_t> indices_to_index(num_weights, -1ull);
  std::vector<int64_t> index_to_indices;
  index_to_indices.reserve(num_weights);
  int64_t unique_indices = count_and_map_uniq(
      indices_accessor, indices_numel, indices_to_index, index_to_indices);

  int max_threads = at::get_num_threads();
  max_threads = (unique_indices < max_threads) ? unique_indices : max_threads;
  int64_t avg_chunk_down = unique_indices / max_threads;
  std::vector<int64_t> chuck_size(max_threads);
  std::vector<int64_t> chuck_sum_size(max_threads + 1);
  for (auto i = 0; i < max_threads; i++) {
    chuck_size[i] = avg_chunk_down;
  }
  // make chunk balance among threads as 211
  for (auto i = 0; i < unique_indices % max_threads; i++) {
    chuck_size[i] += 1;
  }
  chuck_sum_size[0] = 0;
  for (auto i = 1; i < max_threads; i++) {
    chuck_sum_size[i] = chuck_sum_size[i - 1] + chuck_size[i - 1];
  }
  chuck_sum_size[max_threads] = unique_indices;

  int64_t ddim = grad.size(1);

  at::Tensor index_grad_weight = at::empty({num_weights, ddim}, grad.options());
  T* gradout_data = index_grad_weight.data_ptr<T>();
  zero_ker((T*)gradout_data, num_weights * ddim);

  std::vector<float> temp_grad_weight(unique_indices * ddim);
  float* temp_output = temp_grad_weight.data();
  zero_ker(temp_output, unique_indices * ddim);

  auto offset2bag_accessor = offset2bag_.accessor<int64_t, 1>();
  T* grad_data = grad.data_ptr<T>();
  at::parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
    for (int k = start; k < end; k++) {
      int64_t chunk_start = chuck_sum_size[k];
      int64_t chunk_end = chuck_sum_size[k + 1];
      for (int64_t mb = 0; mb < indices_numel; mb++) {
        int64_t indices_num = indices_accessor[mb];
        int64_t index = indices_to_index[indices_num];
        if (index >= chunk_start && index < chunk_end) {
          auto s = offset2bag_accessor[mb];
          add_ker(
              (float*)(temp_output + index * ddim),
              (T*)(grad_data + s * ddim),
              ddim);
        }
      }
      for (int64_t index = chunk_start; index < chunk_end; index++) {
        auto indices = index_to_indices[index];
        move_ker(
            (T*)(gradout_data + indices * ddim),
            (float*)(temp_output + index * ddim),
            ddim);
      }
    }
  });

  return index_grad_weight;
}

bool embedding_bag_backward_fast_path_sum(
    const at::Tensor grad,
    const at::Tensor indices,
    const at::Tensor offset2bag,
    const at::Tensor per_sample_weights,
    bool scale_grad_by_freq,
    int64_t mode) {
  if ((grad.scalar_type() != at::kFloat) &&
      (grad.scalar_type() != at::kBFloat16))
    return false;
  if ((mode != MODE_SUM) || (grad.stride(1) != 1))
    return false;
  if ((indices.numel() == 0) || (offset2bag.numel() != 0))
    return false;
  if (per_sample_weights.defined() || scale_grad_by_freq)
    return false;

  return true;
}

at::Tensor embedding_bag_get_offset2bag(
    const at::Tensor indices,
    const at::Tensor& offsets,
    const at::Tensor& offset2bag) {
  int64_t indices_numel = indices.numel();
  at::Tensor offset2bag_;
  if (indices_numel != 0 && offset2bag.numel() == 0) {
    offset2bag_ =
        at::native::full({indices.sizes()[0] + 1}, 0, indices.scalar_type());
    make_offset2bag(offsets, indices, offset2bag_);
    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    offset2bag_ = offset2bag;
  }
  return offset2bag_;
}

at::Tensor embedding_bag_backward_kernel_impl(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    int64_t num_weights,
    bool sparse) {
  if (sparse) {
    if (is_bfloat16_tensor(grad)) {
      return embedding_bag_sparse_backward_sum_fast<at::BFloat16>(
          grad, indices, offsets, num_weights);
    } else {
      return embedding_bag_sparse_backward_sum_fast<float>(
          grad, indices, offsets, num_weights);
    }
  } else {
    if (is_bfloat16_tensor(grad)) {
      return embedding_bag_dense_backward_sum_fast<at::BFloat16>(
          grad, indices, offsets, num_weights);
    } else {
      return embedding_bag_dense_backward_sum_fast<float>(
          grad, indices, offsets, num_weights);
    }
  }
}

at::Tensor embedding_bag_int8_kernel_impl(
    const at::Tensor& qweight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool include_last_offset) {
  int64_t ddim = qweight.size(1);
  double scale = at::native::q_scale_quant(qweight);
  int8_t* qweight_data =
      reinterpret_cast<int8_t*>(qweight.data_ptr<at::qint8>());
  int64_t output_size = offsets.numel() - 1;
  int64_t* offsets_data = offsets.data_ptr<int64_t>();
  std::vector<int64_t> offsets_include_last;
  if (!include_last_offset) {
    output_size = offsets.numel();
    offsets_include_last.resize(output_size + 1);
    int64_t* offsets_include_last_data = offsets_include_last.data();
    int64_t iter_time = (output_size >> 5);
    int64_t align32_size = (iter_time << 5);
    int64_t left_size = output_size - align32_size;
    at::parallel_for(0, iter_time, 16, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i += 1) {
        auto start_offset = i << 5;
        move_ker(
            &offsets_include_last_data[start_offset],
            &offsets_data[start_offset],
            32);
      }
    });
    if (left_size > 0) {
      move_ker(
          &offsets_include_last_data[align32_size],
          &offsets_data[align32_size],
          left_size);
    }
    offsets_include_last[output_size] = indices.numel();
    offsets_data = offsets_include_last.data();
  }
  // init output tensor
  at::QuantizerPtr output_quantizer =
      at::make_per_tensor_affine_quantizer(scale, /*zp=*/0, at::kQInt8);
  at::Tensor output = at::new_qtensor(
      /*sizes=*/{output_size, qweight.size(1)},
      qweight.options(),
      output_quantizer);
  int8_t* output_data = reinterpret_cast<int8_t*>(output.data_ptr<at::qint8>());
  auto indices_accessor = indices.accessor<int64_t, 1>();
  at::parallel_for(0, output_size, 16, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      int8_t* out_data_ptr = &output_data[i * ddim];
      auto inputs_start = offsets_data[i];
      auto inputs_end = offsets_data[i + 1];
      if (inputs_start >= inputs_end) {
        zero_ker(out_data_ptr, ddim);
      } else {
        int8_t* select_data_ptr =
            &qweight_data[indices_accessor[inputs_start] * ddim];
        move_ker(out_data_ptr, select_data_ptr, ddim);
      }
      for (int64_t s = (inputs_start + 1); s < inputs_end; s++) {
        int8_t* select_data_ptr = &qweight_data[indices_accessor[s] * ddim];
        add_ker(out_data_ptr, select_data_ptr, ddim);
      }
    }
  });

  return output;
}

#if defined(DYN_DISP_BUILD)
} // anonymous namespace

REGISTER_DISPATCH(embedding_bag_kernel_stub, &embedding_bag_kernel_impl);
REGISTER_DISPATCH(
    embedding_bag_backward_kernel_stub,
    &embedding_bag_backward_kernel_impl);
REGISTER_DISPATCH(
    embedding_bag_int8_kernel_stub,
    &embedding_bag_int8_kernel_impl);

#endif

} // namespace cpu
} // namespace torch_ipex
