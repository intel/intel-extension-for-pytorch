
// The orginal python code can be found in
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
// apply_rotary_pos_emb
#include "RotaryPositionEmbedding.h"
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(rotary_position_embedding_kernel_stub);

at::Tensor& rotary_position_embedding_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims) {
  RECORD_FUNCTION(
      "ipex::rotary_position_embedding", c10::ArrayRef<c10::IValue>({}));
  rotary_position_embedding_kernel_stub(
      kCPU, t_in, t_emb_pos, t_pos, N, H, offset, rotary_ndims);
  return t_in;
}

at::Tensor rotary_position_embedding_forward_out_cpu(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims) {
  RECORD_FUNCTION(
      "ipex::rotary_position_embedding_out", c10::ArrayRef<c10::IValue>({}));
  rotary_position_embedding_kernel_stub(
      kCPU, t_in, t_emb_pos, t_pos, N, H, offset, rotary_ndims);
  return t_in;
}

at::Tensor& rotary_position_embedding_forward_functionalization(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t_in));
  at::functionalization::impl::sync(t_in);
  auto t_in_ = at::functionalization::impl::from_functional_tensor(t_in);
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(t_emb_pos));
  at::functionalization::impl::sync(t_emb_pos);
  auto t_emb_pos_ =
      at::functionalization::impl::from_functional_tensor(t_emb_pos);
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t_pos));
  at::functionalization::impl::sync(t_pos);
  auto t_pos_ = at::functionalization::impl::from_functional_tensor(t_pos);
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::rotary_position_embedding_out", "")
          .typed<decltype(rotary_position_embedding_forward_out_cpu)>();
  at::AutoDispatchSkipFunctionalize guard;
  auto tmp_output =
      op.call(t_in_, t_emb_pos_, t_pos_, N, H, offset, rotary_ndims);
  at::functionalization::impl::replace_(t_in, tmp_output);
  at::functionalization::impl::commit_update(t_in);
  at::functionalization::impl::sync(t_in);
  return t_in;
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "rotary_position_embedding(Tensor(a!) t_in, Tensor t_emb_pos, Tensor t_pos, int N, int H, int offset, int rotary_ndims)-> Tensor(a!)");
  m.impl(
      "rotary_position_embedding",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::rotary_position_embedding_forward_cpu);
  m.impl(
      "rotary_position_embedding",
      c10::DispatchKey::Functionalize,
      torch_ipex::cpu::rotary_position_embedding_forward_functionalization);
  m.def(
      "rotary_position_embedding_out(Tensor t_in, Tensor t_emb_pos, Tensor t_pos, int N, int H, int offset, int rotary_ndims)-> Tensor");
  m.impl(
      "rotary_position_embedding_out",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::rotary_position_embedding_forward_out_cpu);
}
} // namespace
