#include "LinearWoqPacked.h"
#include <ideep.hpp>
#include "aten/Linear.h"
#include "aten/WeightPack.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace woq_linear {

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexWoqLinearOpContext::create_context(
      std::move(weight), std::move(bias), batch_size);
}

at::Tensor woq_linear_run(
    const at::Tensor& input,
    c10::intrusive_ptr<WoqLinearOpContext> op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::woq_linear_run", c10::ArrayRef<c10::IValue>({}));

  return op_context->run(input);
}

ContextLinearWoq create(
    at::Tensor& weight,
    at::Tensor& zero_points,
    at::Tensor& scales,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size) {
  auto packed_weight = woq_linear_pack_weight(weight, zero_points, scales);
  return ContextLinearWoq{
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
  };
}

at::Tensor run(
    ContextLinearWoq& context,
    const at::Tensor& zero_points_float,
    const at::Tensor& scales_float,
    const at::Tensor& input) {
  TORCH_CHECK(
      input.size(input.dim() - 1) == context.at_weight_.size(1),
      "Check the shapes of mat1 and mat2, they cannot be multiplied!");
  auto input_ = input.contiguous();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(context.at_bias_);
  const at::Tensor& bias = *bias_maybe_owned;
  return woq_linear_kernel(
      input_, context.at_weight_, zero_points_float, scales_float, bias);
}

at::Tensor pack(ContextLinearWoq& context, const at::Tensor& tensor) {
  return tensor;
}

at::Tensor unpack(ContextLinearWoq& context, const at::Tensor& tensor) {
  return woq_linear_unpack_weight(tensor);
}

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
