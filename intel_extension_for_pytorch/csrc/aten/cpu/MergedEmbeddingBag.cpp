#include "MergedEmbeddingBag.h"
#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <torch/all.h>
#include "csrc/autocast/autocast_mode.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(merged_embeddingbag_forward_cpu_kernel_stub);

std::vector<Tensor> merged_embeddingbag_forward_cpu(
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const std::vector<int64_t> pooling_modes) {
  /*
  pointer to merged_embeddingbag_forward_cpu_kernel_impl(
      indices, offsets, weights, pooling_modes);
  */
  return merged_embeddingbag_forward_cpu_kernel_stub(
      kCPU, indices, offsets, weights, pooling_modes);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

std::vector<Tensor> merged_embeddingbag_forward(
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const std::vector<int64_t> pooling_modes) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::merged_embeddingbag_forward", "")
          .typed<decltype(merged_embeddingbag_forward)>();
  bool cast_to_bfloat16 =
      !at::GradMode::is_enabled() && at::kBFloat16 == get_autocast_dtype();
  auto casted_weights =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weights) : weights;
  return op.call(indices, offsets, casted_weights, pooling_modes);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "merged_embeddingbag_forward(Tensor indices, Tensor offsets, Tensor[] weight, int[] pooling_modes) -> Tensor[]");
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
