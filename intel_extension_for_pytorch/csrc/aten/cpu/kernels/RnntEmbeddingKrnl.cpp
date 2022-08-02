#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <csrc/aten/cpu/RnntEmbedding.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>

#include "csrc/cpu/vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace torch_ipex::cpu::kernel;

template <typename T>
inline void rnnt_embedding_kernel_body(
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
      int64_t out_pos = i * embedding_dim;
      if (embed_idx == _SOS) {
        zero_ker(&embedding_out_ptr[out_pos], embedding_dim);
      } else {
        int64_t in_pos = embed_idx * embedding_dim;
        move_ker(
            &embedding_out_ptr[out_pos],
            &embedding_table_ptr[in_pos],
            embedding_dim);
      }
    }
  });
}

void rnnt_embedding_kernel_impl(
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
    rnnt_embedding_kernel_body<at::BFloat16>(
        embedding_table, idx, embedding_out, _SOS, batch_size, embedding_dim);
  } else {
    rnnt_embedding_kernel_body<float>(
        embedding_table, idx, embedding_out, _SOS, batch_size, embedding_dim);
  }
}

} // anonymous namespace

REGISTER_DISPATCH(rnnt_embedding_kernel_stub, &rnnt_embedding_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
