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
    c10::optional<int64_t> batch_size,
    int64_t lowp_mode,
    int64_t num_concats) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexWoqLinearOpContext::create_context(
      std::move(weight), std::move(bias), batch_size, lowp_mode, num_concats);
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
    at::Tensor& scales,
    at::Tensor& zero_points,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size,
    int64_t lowp_mode) {
  // // TODO Will support optimized impl
  // if (weight.scalar_type() == c10::ScalarType::QUInt4x2) {
  //   return ContextLinearWoq{
  //       std::move(weight),
  //       bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
  //   };
  // }
  auto packed_weight = woq_linear_pack_weight(weight, scales, zero_points, lowp_mode);
  return ContextLinearWoq(
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt
  );
}

at::Tensor run(
    ContextLinearWoq& context,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const at::Tensor& input,
    int64_t lowp_mode,
    int64_t num_concats) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2 ? context.at_weight_.size(1) :
      context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ", input.size(input.dim() - 1), " and ", w_k, " respectively.");
  auto input_ = input.contiguous();
//   c10::MaybeOwned<at::Tensor> bias_maybe_owned =
//       at::borrow_from_optional_tensor(context.at_bias_);
//   const at::Tensor& bias = *bias_maybe_owned;
  return woq_linear_kernel(
      input_, context.at_weight_, scales_list, zps_list, context.bias_list_, lowp_mode, num_concats);
}

// Called by IpexWoqLinearOpContext::run_eltwise
at::Tensor run_eltwise(
    ContextLinearWoq& context,
    const at::Tensor& scales_float,
    const at::Tensor& zero_points_float,
    const at::Tensor& input,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm,
    int64_t lowp_mode) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2 ? context.at_weight_.size(1) : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ", input.size(input.dim() - 1), " and ", w_k, " respectively.");
  auto input_ = input.contiguous();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(context.at_bias_);
  const at::Tensor& bias = *bias_maybe_owned;
  return woq_linear_eltwise_kernel(
      input_, context.at_weight_, scales_float, zero_points_float, bias,
      post_op, scalars, algorithm, lowp_mode);
}

// Registered as JIT op
at::Tensor woq_linear_eltwise_run(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm) {
  static std::map<c10::string_view, std::string> postop_to_record_name_map = {
    {"relu", "torch_ipex::woq_linear_relu_run"},
    {"gelu", "torch_ipex::woq_linear_gelu_run"},
  };
  RECORD_FUNCTION(
      postop_to_record_name_map[post_op], c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_eltwise(input, post_op, scalars, algorithm);
}

// Called by IpexWoqLinearOpContext::run_add
at::Tensor run_add(
    ContextLinearWoq& context,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    int64_t lowp_mode,
    int64_t num_concats) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2 ? context.at_weight_.size(1) : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ", input.size(input.dim() - 1), " and ", w_k, " respectively.");
  auto input_ = input.contiguous();
//   c10::MaybeOwned<at::Tensor> bias_maybe_owned =
//       at::borrow_from_optional_tensor(context.at_bias_);
//   const at::Tensor& bias = *bias_maybe_owned;
  auto output = woq_linear_kernel(
      input_, context.at_weight_, scales_list, zps_list, context.bias_list_, lowp_mode, num_concats);
  at::add_out(accumu, output, accumu, alpha.value());
  return accumu;
}

// Called by IpexWoqLinearOpContext::run_add_relu
at::Tensor run_add_relu(
    ContextLinearWoq& context,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    int64_t lowp_mode,
    int64_t num_concats) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2 ? context.at_weight_.size(1) : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ", input.size(input.dim() - 1), " and ", w_k, " respectively.");
  auto input_ = input.contiguous();
//   c10::MaybeOwned<at::Tensor> bias_maybe_owned =
//       at::borrow_from_optional_tensor(context.at_bias_);
//   const at::Tensor& bias = *bias_maybe_owned;
  auto output = woq_linear_kernel(
      input_, context.at_weight_, scales_list, zps_list, context.bias_list_, lowp_mode, num_concats);
  at::add_out(accumu, output, accumu, alpha.value());
  at::relu_(accumu);
  return accumu;
}

// Registered as JIT op
at::Tensor woq_linear_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_add_run", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_add(input, accumu, alpha);
}

// Registered as JIT op
at::Tensor woq_linear_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_add_relu_run", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
            op_context.data_ptr<int64_t>()[0])
      ->run_add_relu(input, accumu, alpha);
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
