#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include "torch_ipex/csrc/cpu/bf16/vec/bf16_vec_kernel.h"

namespace torch_ipex {
namespace kernel {

template <typename T>
inline void rnnt_embedding_kernel_impl(
    const at::Tensor& embedding_table,
    const at::Tensor& idx,
    at::Tensor embedding_out,
    int64_t _SOS,
    int64_t batch_size,
    int64_t embedding_dim) {
  auto* embedding_table_ptr = embedding_table.data_ptr<T>();
  auto* embedding_out_ptr = embedding_out.data_ptr<T>();

  int64_t* idx_ptr = static_cast<int64_t*>(idx.data_ptr());

  at::parallel_for(0, batch_size, 16, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; i++) {
      int64_t embed_idx = idx_ptr[i];
      if (embed_idx == _SOS) {
        continue;
      }
      int64_t in_pos = embed_idx * embedding_dim;
      int64_t out_pos = i * embedding_dim;
      move_ker(
          &embedding_out_ptr[out_pos],
          &embedding_table_ptr[in_pos],
          embedding_dim);
    }
  });
}

inline void rnnt_embedding_kernel(
    const at::Tensor& embedding_table,
    const at::Tensor& idx,
    at::Tensor embedding_out,
    int64_t _SOS,
    int64_t batch_size,
    int64_t embedding_dim) {
  AT_ASSERTM(
      (embedding_table.scalar_type() == at::kBFloat16 ||
       embedding_table.scalar_type() == at::kFloat),
      "only support embedding_table to be float or bf16 tensor");
  if (embedding_table.scalar_type() == at::kBFloat16) {
    rnnt_embedding_kernel_impl<at::BFloat16>(
        embedding_table, idx, embedding_out, _SOS, batch_size, embedding_dim);
  } else {
    rnnt_embedding_kernel_impl<float>(
        embedding_table, idx, embedding_out, _SOS, batch_size, embedding_dim);
  }
}

/*
  rnnt_embedding: used in the predict_batch of the batched_decoder of RNN-T.
  Get embeddings for given idx.
  When the index is equal to -1, set the lookup result to be 0.0.

  embedding_table: the lookup table that stores embeddings
  idx: indices to extract from the embedding_table
    shape: [batch_size, 1], dtype: torch.int64
    The index could be -1, which means filling the lookup result with 0.0
  embedding_out: output of the embedding look-up
  _SOS: -1 to mark the Start Of Sequence
  batch_size: equals to idx.shape[0]
  embedding_dim: equals to embedding_table.weight.shape[1]
*/
static void rnnt_embedding(const at::Tensor &embedding_table,
                           const at::Tensor &idx, at::Tensor embedding_out,
                           int64_t _SOS, int64_t batch_size,
                           int64_t embedding_dim) {
#if defined(IPEX_DISP_OP)
  printf("IPEX::rnnt_embedding\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEX::rnnt_embedding", std::vector<c10::IValue>({}));
#endif
  rnnt_embedding_kernel(
      embedding_table, idx, embedding_out, _SOS, batch_size, embedding_dim);
}

} // namespace kernel
} // namespace torch_ipex

namespace {

static auto dispatch = torch::RegisterOperators().op(
    "torch_ipex::rnnt_embedding", &torch_ipex::kernel::rnnt_embedding);

}