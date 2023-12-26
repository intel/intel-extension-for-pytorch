#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/quantized/Quantizer.h>
#include <aten/EmbeddingBag.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include <algorithm>

#include "autocast/autocast_mode.h"
#include "cpu/kernels/Embeddingbag.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace torch_ipex::cpu::kernel;
using namespace at;

static inline void make_offset2bag(
    const Tensor& offsets,
    const Tensor& indices,
    Tensor& offset2bag) {
  offset2bag.index_add_(
      0,
      offsets,
      ones_like(offsets, offsets.options())); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1; // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0); // offset2bag = [0 0 1 1 2]
}

// To save compute, if we are going to go down the fast path case for the 'sum'
// mode, we skip calculating offset2bag, since it is not going to be used.
static inline bool is_bfloat16_tensor(const Tensor tensor_) {
  if (tensor_.scalar_type() == kBFloat16)
    return true;
  return false;
}

template <typename T>
static inline Tensor _embedding_bag_index_add_select_fast(
    const Tensor indices,
    const Tensor src,
    const Tensor offsets,
    bool include_last_offset) {
  int64_t ddim = src.size(1);
  T* src_data = src.data_ptr<T>();
  int64_t output_size = offsets.numel();
  if (include_last_offset) {
    output_size -= 1;
  }
  int64_t* offsets_data = offsets.data_ptr<int64_t>();
  auto indices_accessor = indices.accessor<int64_t, 1>();
  int64_t last_index = indices.numel();
  int64_t last_offset = output_size - 1;

  Tensor output = empty({output_size, src.size(1)}, src.options());
  auto* output_data = output.data_ptr<T>();
  parallel_for(0, output_size, 16, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      auto* out_data_ptr = &output_data[i * ddim];
      auto inputs_start = offsets_data[i];
      auto inputs_end = i == last_offset ? last_index : offsets_data[i + 1];
      if (inputs_end - inputs_start == 1) {
        T* select_data_ptr = &src_data[indices_accessor[inputs_start] * ddim];
        move_ker(out_data_ptr, select_data_ptr, ddim);
      } else {
        using acc_t = acc_type<T, true>;
        acc_t temp_out[ddim];
        zero_ker(temp_out, ddim);
        for (int64_t s = inputs_start; s < inputs_end; s++) {
          T* select_data_ptr = &src_data[indices_accessor[s] * ddim];
          add_ker(temp_out, select_data_ptr, ddim);
        }
        move_ker(out_data_ptr, temp_out, ddim);
      }
    }
  });

  return output;
}

Tensor embedding_bag_kernel_impl(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool include_last_offset) {
  Tensor offsets_ = offsets.is_contiguous() ? offsets : offsets.contiguous();

  Tensor output;
  if (is_bfloat16_tensor(weight)) {
    output = _embedding_bag_index_add_select_fast<BFloat16>(
        indices, weight, offsets_, include_last_offset);
  } else {
    output = _embedding_bag_index_add_select_fast<float>(
        indices, weight, offsets_, include_last_offset);
  }
  return output;
}

static inline Tensor expand_values_if_needed(const Tensor& values) {
  // expand
  if (values.dim() == 0) {
    // Mimic Numpy behavior here and treat it as a 1D tensor
    return values.expand({1});
  }

  return values;
}

template <typename T>
static inline Tensor embedding_bag_sparse_backward_sum_fast(
    const Tensor grad,
    const Tensor indices,
    const Tensor offsets,
    int num_weights) {
  assert(grad.stride(1) == 1);

  int64_t indices_size0 = indices.size(0);
  int64_t ddim = grad.size(1);
  Tensor index_grad = empty({indices_size0, ddim}, grad.options());
  int grad_stride0 = grad.stride(0);

  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  auto offset_numel = offsets.numel();

  T* gradout_data = index_grad.data_ptr<T>();
  T* grad_data = grad.data_ptr<T>();
  parallel_for(0, offset_numel, 16, [&](int64_t start, int64_t end) {
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
  auto weight_size = std::array<SymInt, 2>{{num_weights, num_features}};
  auto dense_options = index_grad.options();

  if (index_grad.numel() == 0) {
    return _sparse_coo_tensor_unsafe_symint(
        empty({1, 0}, indices.options()),
        empty_symint({c10::SymInt(0), std::move(num_features)}, dense_options),
        weight_size);
  }

  auto index = indices.reshape({1, -1});
  auto values =
      index_grad.reshape_symint({c10::SymInt(-1), std::move(num_features)});

  return _sparse_coo_tensor_unsafe_symint(
      index, values, weight_size, values.scalar_type());
}

static inline int64_t count_and_map_uniq(
    const TensorAccessor<int64_t, 1>& indices_accessor,
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
static inline Tensor embedding_bag_dense_backward_sum_fast(
    const Tensor grad_,
    const Tensor indices,
    const Tensor offsets,
    int num_weights) {
  int64_t indices_numel = indices.numel();
  auto grad = grad_.contiguous();
  assert(indices_numel > 0);
  auto offset_numel = offsets.numel();
  Tensor offset2bag_;

  offset2bag_ =
      native::full({indices.sizes()[0] + 1}, 0, indices.scalar_type());
  make_offset2bag(offsets, indices, offset2bag_);
  offset2bag_.resize_({indices.sizes()[0]});

  auto indices_accessor = indices.accessor<int64_t, 1>();
  std::vector<int64_t> indices_to_index(num_weights, -1ull);
  std::vector<int64_t> index_to_indices;
  index_to_indices.reserve(num_weights);
  int64_t unique_indices = count_and_map_uniq(
      indices_accessor, indices_numel, indices_to_index, index_to_indices);

  int max_threads = get_num_threads();
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

  Tensor index_grad_weight = empty({num_weights, ddim}, grad.options());
  T* gradout_data = index_grad_weight.data_ptr<T>();
  zero_ker((T*)gradout_data, num_weights * ddim);

  std::vector<float> temp_grad_weight(unique_indices * ddim);
  float* temp_output = temp_grad_weight.data();
  zero_ker(temp_output, unique_indices * ddim);

  auto offset2bag_accessor = offset2bag_.accessor<int64_t, 1>();
  T* grad_data = grad.data_ptr<T>();
  parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
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

Tensor embedding_bag_backward_kernel_impl(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t num_weights,
    bool sparse) {
  if (sparse) {
    if (is_bfloat16_tensor(grad)) {
      return embedding_bag_sparse_backward_sum_fast<BFloat16>(
          grad, indices, offsets, num_weights);
    } else {
      return embedding_bag_sparse_backward_sum_fast<float>(
          grad, indices, offsets, num_weights);
    }
  } else {
    if (is_bfloat16_tensor(grad)) {
      return embedding_bag_dense_backward_sum_fast<BFloat16>(
          grad, indices, offsets, num_weights);
    } else {
      return embedding_bag_dense_backward_sum_fast<float>(
          grad, indices, offsets, num_weights);
    }
  }
}

Tensor embedding_bag_int8_kernel_impl(
    const Tensor& qweight,
    const Tensor& indices,
    const Tensor& offsets,
    double output_scale,
    bool include_last_offset) {
  int64_t ddim = qweight.size(1);
  double weight_scale = native::q_scale_quant(qweight);
  double inv_o_scale = 1.0 / output_scale;
  int8_t* qweight_data = reinterpret_cast<int8_t*>(qweight.data_ptr<qint8>());
  int64_t output_size = offsets.numel();
  if (include_last_offset) {
    output_size -= 1;
  }
  int64_t* offsets_data = offsets.data_ptr<int64_t>();
  auto indices_accessor = indices.accessor<int64_t, 1>();
  int64_t last_index = indices.numel();
  int64_t last_offset = output_size - 1;

  // init output tensor
  QuantizerPtr output_quantizer =
      make_per_tensor_affine_quantizer(output_scale, /*zp=*/0, kQInt8);
  Tensor output = new_qtensor(
      /*sizes=*/{output_size, qweight.size(1)},
      qweight.options(),
      output_quantizer);
  int8_t* output_data = reinterpret_cast<int8_t*>(output.data_ptr<qint8>());
  bool need_requantize = (output_scale - weight_scale) > 0.0001;
  parallel_for(0, output_size, 16, [&](int64_t start, int64_t end) {
    // float fp32_buffer[ddim] __attribute__((aligned(64)));
    std::unique_ptr<float, decltype(ipex_free_aligned)*> fp32_buffer_uq_ptr(
        (float*)ipex_alloc_aligned(sizeof(float) * ddim, 64),
        ipex_free_aligned);
    float* fp32_buffer = fp32_buffer_uq_ptr.get();

    for (int64_t i = start; i < end; i++) {
      int8_t* out_data_ptr = &output_data[i * ddim];
      auto inputs_start = offsets_data[i];
      auto inputs_end = i == last_offset ? last_index : offsets_data[i + 1];
      if (inputs_end - inputs_start <= 1 && !need_requantize) {
        // Do not re-quantize when bag-size == 1 for performance consideraion
        // It is proved to be have enough accuracy on DLRM-V1
        // We can revise this if other models with embeddingbag are not accurate
        // enough
        int8_t* select_data_ptr =
            &qweight_data[indices_accessor[inputs_start] * ddim];
        move_ker(out_data_ptr, select_data_ptr, ddim);
      } else {
        zero_ker(&fp32_buffer[0], ddim);
        for (int64_t s = inputs_start; s < inputs_end; s++) {
          int8_t* select_data_ptr = &qweight_data[indices_accessor[s] * ddim];
          scale_fp32_and_fma(
              &fp32_buffer[0], select_data_ptr, weight_scale, ddim);
        }
#ifdef CPU_CAPABILITY_AVX2
        vec::QuantizeAvx2<c10::qint8::underlying>(
            &fp32_buffer[0],
            out_data_ptr,
            ddim,
            inv_o_scale,
            /*zp=*/0);
#else
        vec::QuantizeAvx512<c10::qint8::underlying>(
            &fp32_buffer[0],
            out_data_ptr,
            ddim,
            inv_o_scale,
            /*zp=*/0);
#endif
      }
    }
  });

  return output;
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(embedding_bag_kernel_stub, &embedding_bag_kernel_impl);
IPEX_REGISTER_DISPATCH(
    embedding_bag_backward_kernel_stub,
    &embedding_bag_backward_kernel_impl);
IPEX_REGISTER_DISPATCH(
    embedding_bag_int8_kernel_stub,
    &embedding_bag_int8_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
