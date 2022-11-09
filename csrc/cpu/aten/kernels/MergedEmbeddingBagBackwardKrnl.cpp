#include <aten/MergedEmbeddingBag.h>
#include <c10/core/CPUAllocator.h>
#include <omp.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at;
using namespace torch_ipex::cpu::kernel;

template <typename T>
inline void grad_accumulate(
    T* grad_w_ptr,
    T* grad_out,
    const BatchedHyperCompressedSparseColumn& batched_csc,
    int64_t uniq_index_id,
    int64_t grad_w_ptr_offsets,
    int vector_size) {
  // grad_out accumulate
  using acc_t = acc_type<T, true>;
  acc_t grad_out_acc_buffer[vector_size] __attribute__((aligned(64)));
  zero_ker(grad_out_acc_buffer, vector_size);
  for (int r = batched_csc.segment_ptr[uniq_index_id];
       r < batched_csc.segment_ptr[uniq_index_id + 1];
       ++r) {
    T* grad_out_ptr =
        &grad_out[batched_csc.output_row_indices[r] * vector_size];
    if (batched_csc.weights && batched_csc.weights[r] != 1) {
      madd_ker(
          grad_out_acc_buffer,
          grad_out_ptr,
          vector_size,
          batched_csc.weights[r]);
    } else {
      add_ker(grad_out_acc_buffer, grad_out_ptr, vector_size);
    }
  }
  // write the accumulated grad_out into grad_w_row_ptr
  T* grad_w_row_ptr = &grad_w_ptr[grad_w_ptr_offsets];
  move_ker(grad_w_row_ptr, grad_out_acc_buffer, vector_size);
}

std::vector<Tensor> merged_embeddingbag_backward_cpu_kernel_impl(
    const std::vector<Tensor>& grad_outs_,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const Tensor& indices_with_row_offset,
    const Tensor& row_offsets,
    const std::vector<int64_t> pooling_modes) {
  int64_t n_tables = weights.size();
  int64_t bs = (offsets.numel() - 1) / n_tables;
  int64_t* row_offset_data = row_offsets.data_ptr<int64_t>();
  BatchedHyperCompressedSparseColumn batched_csc;
  sort_based_batched_csr2csc_opt(
      batched_csc,
      bs,
      offsets,
      indices_with_row_offset,
      pooling_modes,
      row_offset_data[n_tables]);
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(__FUNCTION__, std::vector<c10::IValue>({}));
#endif
  std::vector<int> vector_sizes;
  std::vector<void*> grad_weights_ptr;
  std::vector<Tensor> grad_weights;
  std::vector<void*> grad_outs_ptr;
  std::vector<ScalarType> dtypes;
  auto grad_outs = grad_outs_;
  for (int i = 0; i < n_tables; i++) {
    grad_outs[i] = grad_outs_[i].contiguous();
    grad_outs_ptr.emplace_back(grad_outs[i].data_ptr());
    grad_weights.emplace_back(at::zeros(
        {weights[i].size(0), grad_outs[i].size(-1)}, grad_outs[i].options()));
    grad_weights_ptr.emplace_back(grad_weights[i].data_ptr());
    vector_sizes.emplace_back(weights[i].size(1));
    dtypes.emplace_back(weights[i].scalar_type());
  }

  auto get_table_id = [&](int index) {
    int table_id = 0;
    while (index >= row_offset_data[table_id + 1]) {
      table_id++;
    }
    return table_id;
  };

  int uniq_indice = batched_csc.uniq_indices;
#pragma omp parallel for schedule(static, 1)
  for (int uniq_index_id = 0; uniq_index_id < uniq_indice; ++uniq_index_id) {
    int row_index = batched_csc.segment_indices[uniq_index_id];
    int table_id = get_table_id(row_index);
    int vector_size = vector_sizes[table_id];
    int64_t weight_offsets =
        (row_index - row_offset_data[table_id]) * vector_size;
    if (dtypes[table_id] == ScalarType::BFloat16) {
      grad_accumulate<BFloat16>(
          (BFloat16*)grad_weights_ptr[table_id],
          (BFloat16*)grad_outs_ptr[table_id],
          batched_csc,
          uniq_index_id,
          weight_offsets,
          vector_size);
    } else if (dtypes[table_id] == ScalarType::Float) {
      grad_accumulate<float>(
          (float*)grad_weights_ptr[table_id],
          (float*)grad_outs_ptr[table_id],
          batched_csc,
          uniq_index_id,
          weight_offsets,
          vector_size);
    } else {
      grad_accumulate<double>(
          (double*)grad_weights_ptr[table_id],
          (double*)grad_outs_ptr[table_id],
          batched_csc,
          uniq_index_id,
          weight_offsets,
          vector_size);
    }
  }
  return grad_weights;
}

} // anonymous namespace

REGISTER_DISPATCH(
    merged_embeddingbag_backward_cpu_kernel_stub,
    &merged_embeddingbag_backward_cpu_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
