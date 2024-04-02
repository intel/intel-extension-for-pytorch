#include "MergedEmbeddingBag.h"
#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <torch/all.h>
#include "autocast/autocast_mode.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(merged_embeddingbag_forward_cpu_kernel_stub);

std::vector<Tensor> merged_embeddingbag_forward_cpu(
    const std::vector<Tensor>& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets) {
  /*
  pointer to merged_embeddingbag_forward_cpu_kernel_impl(
      weights, indices, offsets, pooling_mode, include_last_offsets);
  */
  return merged_embeddingbag_forward_cpu_kernel_stub(
      kCPU, weights, indices, offsets, pooling_mode, include_last_offsets);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

std::vector<Tensor> merged_embeddingbag_forward(
    const std::vector<Tensor>& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::merged_embeddingbag_forward", "")
          .typed<decltype(merged_embeddingbag_forward)>();
  bool cast_to_bfloat16 =
      !at::GradMode::is_enabled() && at::kBFloat16 == get_autocast_dtype();
  auto casted_weights =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weights) : weights;
  return op.call(
      casted_weights, indices, offsets, pooling_mode, include_last_offsets);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "merged_embeddingbag_forward(Tensor[] weights, Tensor[] indices, Tensor[] offsets, int pooling_mode, bool include_last_offsets) -> Tensor[]");
  m.impl(
      "merged_embeddingbag_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::merged_embeddingbag_forward_cpu);
  m.impl(
      "merged_embeddingbag_forward",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::merged_embeddingbag_forward);
}

} // namespace
