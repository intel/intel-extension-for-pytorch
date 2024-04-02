
// The orginal python code can be found in
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
// apply_rotary_pos_emb
#include "RotaryPositionEmbedding.h"
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(rotary_position_embedding_kernel_stub);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rotary_position_embedding_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims) {
  RECORD_FUNCTION(
      "ipex::rotary_position_embedding", c10::ArrayRef<c10::IValue>({}));
  return rotary_position_embedding_kernel_stub(
      kCPU, t_in, t_emb_pos, t_pos, N, H, offset, rotary_ndims);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "rotary_position_embedding(Tensor t_in, Tensor t_emb_pos, Tensor t_pos, int N, int H, int offset, int rotary_ndims)-> (Tensor, Tensor, Tensor)");
  m.impl(
      "rotary_position_embedding",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::rotary_position_embedding_forward_cpu);
}
} // namespace
