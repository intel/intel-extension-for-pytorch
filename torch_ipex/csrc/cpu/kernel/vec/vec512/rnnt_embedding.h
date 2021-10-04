#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>

#include "torch_ipex/csrc/cpu/bf16/vec/bf16_vec_kernel.h"

namespace torch_ipex {
namespace kernel {
namespace vec {
namespace vec512 {

template <typename T>
inline void rnnt_embedding_kernel_impl(const at::Tensor &embedding_table,
                                       const at::Tensor &idx,
                                       at::Tensor embedding_out, int64_t _SOS,
                                       int64_t batch_size,
                                       int64_t embedding_dim) {
  auto *embedding_table_ptr = embedding_table.data_ptr<T>();
  auto *embedding_out_ptr = embedding_out.data_ptr<T>();

  int64_t *idx_ptr = static_cast<int64_t *>(idx.data_ptr());

  at::parallel_for(0, batch_size, 16, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; i++) {
      int64_t embed_idx = idx_ptr[i];
      if (embed_idx == _SOS) {
        continue;
      }
      int64_t in_pos = embed_idx * embedding_dim;
      int64_t out_pos = i * embedding_dim;
      move_ker(&embedding_out_ptr[out_pos], &embedding_table_ptr[in_pos],
               embedding_dim);
    }
  });
}

inline void rnnt_embedding_kernel(const at::Tensor &embedding_table,
                                  const at::Tensor &idx,
                                  at::Tensor embedding_out, int64_t _SOS,
                                  int64_t batch_size, int64_t embedding_dim) {
  AT_ASSERTM((embedding_table.scalar_type() == at::kBFloat16 ||
              embedding_table.scalar_type() == at::kFloat),
             "only support embedding_table to be float or bf16 tensor");
  if (embedding_table.scalar_type() == at::kBFloat16) {
    rnnt_embedding_kernel_impl<at::BFloat16>(
        embedding_table, idx, embedding_out, _SOS, batch_size, embedding_dim);
  } else {
    rnnt_embedding_kernel_impl<float>(embedding_table, idx, embedding_out, _SOS,
                                      batch_size, embedding_dim);
  }
}

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace torch_ipex
