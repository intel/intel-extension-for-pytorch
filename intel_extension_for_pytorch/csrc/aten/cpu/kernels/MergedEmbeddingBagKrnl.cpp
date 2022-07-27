#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <csrc/aten/cpu/MergedEmbeddingBag.h>
#include <torch/extension.h>
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at;
using namespace torch_ipex::cpu::kernel;

template <typename T>
inline void emb_pooling_ker(
    T* out,
    T* in,
    size_t pool_begin,
    size_t pool_end,
    size_t vector_size,
    int64_t* indices_data,
    int64_t* offsets_data,
    int64_t pooling_mode) {
  auto idx = indices_data[pool_begin];
  auto weight_ptr = &in[idx * vector_size];
  if (pool_end - pool_begin == 1) {
    move_ker(out, weight_ptr, vector_size);
  } else {
    using acc_t = acc_type<T, true>;
    // add if there is more than 1 indice in this bag, need accumulate to float
    // buffer
    acc_t temp_out[vector_size];
    zero_ker(temp_out, vector_size);
    for (auto p = pool_begin; p < pool_end; ++p) {
      idx = indices_data[p];
      weight_ptr = &in[idx * vector_size];
      add_ker(temp_out, weight_ptr, vector_size);
    }
    if (pooling_mode == MEAN) {
      auto L = pool_end - pool_begin;
      const double scale_factor = 1.0 / L;
#pragma omp simd
      for (int d = 0; d < vector_size; ++d) {
        temp_out[d] = scale_factor * temp_out[d];
      }
    }
    move_ker(out, temp_out, vector_size);
  }
}

void merged_embeddingbag_forward_cpu_kernel(
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const std::vector<int64_t> pooling_modes,
    std::vector<Tensor>& outputs) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  int64_t n_tables = weights.size();
  TORCH_CHECK(n_tables > 0);
  // offsets.numel = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / n_tables;
  TORCH_CHECK(B >= 0);
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(offsets.is_contiguous());

  std::vector<void*> weights_ptr;
  std::vector<ScalarType> dtypes;

  for (auto& w : weights) {
    TORCH_CHECK(w.is_contiguous());
    weights_ptr.emplace_back(w.data_ptr());
    dtypes.emplace_back(w.scalar_type());
  }

  std::vector<void*> outs_ptr;
  for (auto& o : outputs) {
    outs_ptr.emplace_back(o.data_ptr());
  }

  const auto indices_data = indices.data_ptr<int64_t>();
  const auto offsets_data = offsets.data_ptr<int64_t>();

  int64_t n_offsets = offsets.numel() - 1;
  parallel_for(0, n_offsets, 0, [&](int64_t offset_begin, int64_t offset_end) {
    for (int n = offset_begin; n < offset_end; ++n) {
      int table_id = 0;
      int64_t temp_n = n;
      while (temp_n >= B) {
        temp_n -= B;
        table_id += 1;
      }
      const auto pool_begin = offsets_data[n];
      const auto pool_end = offsets_data[n + 1];
      auto feature_size = weights[table_id].size(1);
      if (dtypes[table_id] == ScalarType::BFloat16) {
        BFloat16* out_ptr =
            &(((BFloat16*)outs_ptr[table_id])[temp_n * feature_size]);
        emb_pooling_ker<BFloat16>(
            out_ptr,
            (BFloat16*)weights_ptr[table_id],
            pool_begin,
            pool_end,
            feature_size,
            indices_data,
            offsets_data,
            pooling_modes[table_id]);
      } else if (dtypes[table_id] == ScalarType::Float) {
        float* out_ptr = &(((float*)outs_ptr[table_id])[temp_n * feature_size]);
        emb_pooling_ker<float>(
            out_ptr,
            (float*)weights_ptr[table_id],
            pool_begin,
            pool_end,
            feature_size,
            indices_data,
            offsets_data,
            pooling_modes[table_id]);
      } else {
        double* out_ptr =
            &(((double*)outs_ptr[table_id])[temp_n * feature_size]);
        emb_pooling_ker<double>(
            out_ptr,
            (double*)weights_ptr[table_id],
            pool_begin,
            pool_end,
            feature_size,
            indices_data,
            offsets_data,
            pooling_modes[table_id]);
      }
    }
  });
  return;
}

std::vector<Tensor> merged_embeddingbag_forward_cpu_kernel_impl(
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const std::vector<int64_t> pooling_modes) {
  int64_t n_tables = weights.size();
  int64_t bs = (offsets.numel() - 1) / n_tables;

  std::vector<Tensor> outputs;
  for (auto& w : weights) {
    auto dtype = w.scalar_type();
    TORCH_CHECK(
        kBFloat16 == dtype || kFloat == dtype || kDouble == dtype,
        "merged_embeddingbag_forward_cpu only support weight dtype in bfloat16, float, double");
    int64_t feature_size = w.size(1);
    outputs.emplace_back(empty({bs, feature_size}, w.options()));
  }
  merged_embeddingbag_forward_cpu_kernel(
      indices, offsets, weights, pooling_modes, outputs);

  return outputs;
}

} // anonymous namespace

REGISTER_DISPATCH(
    merged_embeddingbag_forward_cpu_kernel_stub,
    &merged_embeddingbag_forward_cpu_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
