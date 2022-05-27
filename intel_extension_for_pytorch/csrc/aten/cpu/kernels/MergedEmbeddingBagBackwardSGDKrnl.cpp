#include <c10/core/CPUAllocator.h>
#include <csrc/aten/cpu/MergedEmbeddingBag.h>
#include <omp.h>
#include "csrc/cpu/vec/vec.h"
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at;
using namespace torch_ipex::cpu::kernel;

template <typename param_t, typename acc_t>
inline void sgd_update(
    param_t* param_ptr,
    at::BFloat16* trail_ptr,
    acc_t* grad_ptr,
    float weight_decay,
    float lr,
    int size) {
  using Vec = at::vec::Vectorized<param_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec param_vec = Vec::loadu(param_ptr + d);
    Vec grad_vec =
        Vec::loadu(grad_ptr + d) + param_vec * Vec(param_t(weight_decay));

    param_vec -= grad_vec * Vec(param_t(lr));
    param_vec.store(param_ptr + d);
  }
  for (; d < size; d++) {
    param_t grad_val = grad_ptr[d] + param_ptr[d] * weight_decay;
    param_ptr[d] -= grad_val * lr;
  }
}

template <>
inline void sgd_update<at::BFloat16, float>(
    at::BFloat16* param_ptr,
    at::BFloat16* trail_ptr,
    float* grad_ptr,
    float weight_decay,
    float lr,
    int size) {
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec param_bvec = bVec::loadu(param_ptr + d);
    bVec trail_bvec = bVec::loadu(trail_ptr + d);
    fVec param_fvec, param_fvec2;
    std::tie(param_fvec, param_fvec2) =
        at::vec::pack_bfloat16_float(param_bvec, trail_bvec);

    fVec grad_fvec = fVec::loadu(grad_ptr + d);
    fVec grad_fvec2 = fVec::loadu(grad_ptr + d + fVec::size());

    grad_fvec = grad_fvec + param_fvec * fVec(weight_decay);
    grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(weight_decay);

    param_fvec -= grad_fvec * fVec(lr);
    param_fvec2 -= grad_fvec2 * fVec(lr);

    std::tie(param_bvec, trail_bvec) =
        at::vec::unpack_float_bfloat16(param_fvec, param_fvec2);
    param_bvec.store(param_ptr + d);
    trail_bvec.store(trail_ptr + d);
  }
  for (; d < size; d++) {
    float param_val = at::vec::pack_bfloat16_float(param_ptr[d], trail_ptr[d]);
    float grad_val = grad_ptr[d] + param_val * weight_decay;
    param_val -= grad_val * lr;
    std::tie(param_ptr[d], trail_ptr[d]) =
        at::vec::unpack_float_bfloat16(param_val);
  }
}

template <typename T>
inline void AccGradUpdate<T, SGDArgs>::update(
    T* weight,
    T* grad,
    const BatchedHyperCompressedSparseColumn& batched_csc,
    int64_t uniq_index_id,
    int64_t weight_offsets,
    int vector_size,
    int table_id,
    const SGDArgs& args) {
  // grad accumulate
  using acc_t = acc_type<T, true>;
  acc_t grad_acc_buffer[vector_size];
  zero_ker(grad_acc_buffer, vector_size);
  for (int r = batched_csc.segment_ptr[uniq_index_id];
       r < batched_csc.segment_ptr[uniq_index_id + 1];
       ++r) {
    T* grad_ptr = &grad[batched_csc.output_row_indices[r] * vector_size];
    if (batched_csc.weights && batched_csc.weights[r] != 1) {
      madd_ker(grad_acc_buffer, grad_ptr, vector_size, batched_csc.weights[r]);
    } else {
      add_ker(grad_acc_buffer, grad_ptr, vector_size);
    }
  }
  // sgd update
  T* weight_ptr = &weight[weight_offsets];
  BFloat16* bf16_trail_ptr = nullptr;
  if (std::is_same<T, BFloat16>::value) {
    bf16_trail_ptr =
        args.bf16_trail[table_id].data_ptr<BFloat16>() + weight_offsets;
  }
  sgd_update<T, acc_t>(
      weight_ptr,
      bf16_trail_ptr,
      grad_acc_buffer,
      args.weight_decay,
      args.lr,
      vector_size);
}

template <typename optimizer_arg_t>
void merged_embeddingbag_backward_cpu_kernel(
    const std::vector<Tensor>& grads_y,
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const Tensor& indices_with_row_offset,
    const Tensor& row_offsets,
    std::vector<int64_t> pooling_modes,
    const optimizer_arg_t& args) {
  int64_t n_tables = weights.size();
  int64_t bs = (offsets.numel() - 1) / n_tables;
  int64_t* row_offset_data = row_offsets.data_ptr<int64_t>();
  int64_t max_embeddings = row_offset_data[n_tables];
  BatchedHyperCompressedSparseColumn batched_csc;
  sort_based_batched_csr2csc_opt(
      batched_csc,
      bs,
      offsets,
      indices_with_row_offset,
      pooling_modes,
      max_embeddings);
  IPEX_RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  auto get_table_id = [&](int index) {
    int table_id = 0;
    while (index >= row_offset_data[table_id + 1]) {
      table_id++;
    }
    return table_id;
  };

  int uniq_indice = batched_csc.uniq_indices;

  std::vector<void*> weights_ptr;
  std::vector<int64_t> weights_max_offsets;
  std::vector<void*> grads_ptr;
  std::vector<ScalarType> dtypes;

  for (int i = 0; i < n_tables; i++) {
    weights_ptr.emplace_back(weights[i].data_ptr());
    grads_ptr.emplace_back(grads_y[i].data_ptr());
    dtypes.emplace_back(weights[i].scalar_type());
    weights_max_offsets.emplace_back(weights[i].size(0) * weights[i].size(1));
  }

#pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < uniq_indice; ++c) {
    int row_index = batched_csc.segment_indices[c];
    int table_id = get_table_id(row_index);
    int vector_size = weights[table_id].size(1);
    int64_t weight_offsets =
        (row_index - row_offset_data[table_id]) * vector_size;
    TORCH_CHECK(
        weight_offsets >= 0 && weight_offsets < weights_max_offsets[table_id]);
    if (dtypes[table_id] == ScalarType::BFloat16) {
      AccGradUpdate<BFloat16, optimizer_arg_t>::update(
          (BFloat16*)weights_ptr[table_id],
          (BFloat16*)grads_ptr[table_id],
          batched_csc,
          c,
          weight_offsets,
          vector_size,
          table_id,
          args);
    } else if (dtypes[table_id] == ScalarType::Float) {
      AccGradUpdate<float, optimizer_arg_t>::update(
          (float*)weights_ptr[table_id],
          (float*)grads_ptr[table_id],
          batched_csc,
          c,
          weight_offsets,
          vector_size,
          table_id,
          args);
    } else {
      AccGradUpdate<double, optimizer_arg_t>::update(
          (double*)weights_ptr[table_id],
          (double*)grads_ptr[table_id],
          batched_csc,
          c,
          weight_offsets,
          vector_size,
          table_id,
          args);
    }
  }

  return;
}

void merged_embeddingbag_backward_sgd_cpu_kernel_impl(
    const std::vector<Tensor>& grads_y_,
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const Tensor& indices_with_row_offset,
    const Tensor& row_offsets,
    std::vector<int64_t> pooling_modes,
    const std::vector<Tensor>& bf16_trail,
    double weight_decay,
    double lr) {
  int64_t n_tables = weights.size();
  TORCH_CHECK(n_tables == grads_y_.size());
  auto grads_y = grads_y_;
  for (auto i = 0; i < n_tables; i++) {
    TORCH_CHECK(grads_y_[i].scalar_type() == weights[i].scalar_type());
    grads_y[i] = grads_y_[i].contiguous();
  }
  SGDArgs args = SGDArgs(bf16_trail, weight_decay, lr);
  merged_embeddingbag_backward_cpu_kernel<SGDArgs>(
      grads_y,
      indices,
      offsets,
      weights,
      indices_with_row_offset,
      row_offsets,
      pooling_modes,
      args);

  return;
}

} // anonymous namespace

REGISTER_DISPATCH(
    merged_embeddingbag_backward_sgd_cpu_kernel_stub,
    &merged_embeddingbag_backward_sgd_cpu_kernel_impl);

} // namespace cpu
} // namespace torch_ipex