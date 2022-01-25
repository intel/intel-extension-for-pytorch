#include "embeddingbag.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/jit/cpu/kernels/Embeddingbag.h"
#include "csrc/quantization/AutoCast.hpp"
#include "csrc/utils/rw_lock.h"

#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include <algorithm>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(embedding_bag_kernel_stub);
DEFINE_DISPATCH(embedding_bag_backward_kernel_stub);
DEFINE_DISPATCH(embedding_bag_int8_kernel_stub);

class NewEmbeddingBagOp : public torch::autograd::Function<NewEmbeddingBagOp> {
 public:
  static at::Tensor _forward(
      const at::Tensor& weight,
      const at::Tensor& indices,
      const at::Tensor& offsets,
      bool sparse,
      bool include_last_offset) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION(
        "IPEXEmbeddingBagOp::_forward", std::vector<c10::IValue>({}));
#endif

#if defined(DYN_DISP_BUILD)
    auto ret = embedding_bag_kernel_stub(
        kCPU, weight, indices, offsets, include_last_offset);
#else
    auto ret = embedding_bag_kernel_impl(
        weight, indices, offsets, include_last_offset);
#endif

    return ret;
  }

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& weight,
      const at::Tensor& indices,
      const at::Tensor& offsets,
      bool sparse,
      bool include_last_offset) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION(
        "IPEXEmbeddingBagOp::forward", std::vector<c10::IValue>({}));
#endif
    at::AutoNonVariableTypeMode g;
    ctx->saved_data["sparse"] = sparse;
    auto ret = _forward(weight, indices, offsets, sparse, include_last_offset);
    ctx->save_for_backward({weight, indices, offsets});
    return ret;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION(
        "IPEXEmbeddingBagOp::backward", std::vector<c10::IValue>({}));
#endif
    at::AutoNonVariableTypeMode g;
    auto saved = ctx->get_saved_variables();
    at::Tensor weight = saved[0];
    at::Tensor indices = saved[1];
    at::Tensor offsets = saved[2];

    int64_t num_weights = weight.size(0);
    bool sparse = ctx->saved_data["sparse"].toBool();

    at::Tensor grad = sparse ? grad_outputs[0] : grad_outputs[0].contiguous();

#if defined(DYN_DISP_BUILD)
    return {
        embedding_bag_backward_kernel_stub(
            kCPU, grad, indices, offsets, num_weights, sparse),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
#else
    return {
        embedding_bag_backward_kernel_impl(
            grad, indices, offsets, num_weights, sparse),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
#endif
  }
};

at::Tensor _embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset) {
  if (at::GradMode::is_enabled() && weight.requires_grad())
    return NewEmbeddingBagOp::apply(
        weight, indices, offsets, sparse, include_last_offset);
  return NewEmbeddingBagOp::_forward(
      weight, indices, offsets, sparse, include_last_offset);
}

at::Tensor dil_qembeddingbag(
    const at::Tensor weight,
    const at::Tensor indices,
    const at::Tensor offsets,
    bool sparse,
    bool include_last_offset,
    double o_scale,
    int64_t o_zp,
    at::ScalarType o_dtype) {
#if defined(DYN_DISP_BUILD)
  return torch_ipex::cpu::embedding_bag_int8_kernel_stub(
      kCPU, weight, indices, offsets, include_last_offset);
#else
  return torch_ipex::cpu::embedding_bag_int8_kernel_impl(
      weight, indices, offsets, include_last_offset);
#endif
}

} // namespace cpu
} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      torch::schema(
          "torch_ipex::embedding_bag(Tensor weight, Tensor indices, Tensor "
          "offsets, bool sparse, bool include_last_offset) -> Tensor",
          c10::AliasAnalysisKind::PURE_FUNCTION),
      torch_ipex::embedding_bag);
}
} // namespace

namespace torch_ipex {
namespace autocast {

at::Tensor embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::embedding_bag", "")
                       .typed<decltype(embedding_bag)>();
  auto target_type = get_autocast_dtype();
  if (is_quantization_enabled()) {
    return int8::embedding_bag(
        weight, indices, offsets, sparse, include_last_offset);
  }
  // only have bf16 support now, keep fp32 for other target_type
  bool cast_to_bfloat16 =
      !at::GradMode::is_enabled() && at::kBFloat16 == target_type;
  auto casted_weight =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weight) : weight;
  return op.call(casted_weight, indices, offsets, sparse, include_last_offset);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("embedding_bag", torch_ipex::autocast::embedding_bag);
}

} // namespace autocast
} // namespace torch_ipex

/*
A namespace wrapper to keep API compatiable to callers.
And also compatiable to new dyndisp.
*/
namespace torch_ipex {

bool embedding_bag_fast_path_sum(
    const at::Tensor weight,
    const c10::optional<at::Tensor> per_sample_weights,
    int64_t mode,
    const c10::optional<int64_t> padding_idx) {
  if ((mode != MODE_SUM) || (weight.stride(1) != 1))
    return false;
  if ((weight.scalar_type() != at::kFloat) &&
      (weight.scalar_type() != at::kBFloat16))
    return false;
  if (padding_idx.has_value() ||
      (per_sample_weights.has_value() && per_sample_weights.value().defined()))
    return false;
  return true;
}

at::Tensor embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset) {
  return cpu::_embedding_bag(
      weight, indices, offsets, sparse, include_last_offset);
}

} // namespace torch_ipex