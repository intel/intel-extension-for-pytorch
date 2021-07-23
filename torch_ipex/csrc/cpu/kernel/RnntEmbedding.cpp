#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>

#if defined(AVX512)
#include "vec/vec512/rnnt_embedding.h"
#else
#warning "IPEX RNN-T custom kernel requires AVX512"
#endif

namespace torch_ipex {
namespace kernel {

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
  vec::vec512::rnnt_embedding_kernel(embedding_table, idx, embedding_out, _SOS,
                                     batch_size, embedding_dim);
}

} // namespace kernel
} // namespace torch_ipex

namespace {

static auto dispatch = torch::RegisterOperators().op(
    "torch_ipex::rnnt_embedding", &torch_ipex::kernel::rnnt_embedding);

}