#include "LinearMKLPacked.h"
#include "aten/LinearMKL.h"
#include "aten/WeightPack.h"
#include "ideep/IDeepConversions.h"
#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace mkl_sgemm {

c10::intrusive_ptr<MKLOpContext> createLinearMKLPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size) {
  RECORD_FUNCTION(
      "ipex_prepack::createLinearMKLPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexLinearMKLOpContext::create_context(
      std::move(weight), std::move(bias), batch_size);
}

at::Tensor mkl_sgemm_run(
    const at::Tensor& input,
    c10::intrusive_ptr<MKLOpContext> op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::mkl_sgemm_run", c10::ArrayRef<c10::IValue>({}));

  return op_context->run(input);
}

ContextLinearMKL create(
    at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size) {
  auto out_features = weight.size(0);
  auto in_features = weight.size(1);
  auto batch = batch_size.has_value() ? batch_size.value() : 128;
  std::vector<int64_t> sgemm_sizes;
  sgemm_sizes.push_back(batch);
  sgemm_sizes.push_back(in_features);
  sgemm_sizes.push_back(out_features);

  auto mkl_weight =
      mkl_sgemm_pack_weight(batch, out_features, in_features, weight);

  return ContextLinearMKL{
      std::move(sgemm_sizes),
      std::move(mkl_weight),
      std::move(weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
  };
}

at::Tensor run(ContextLinearMKL& context, const at::Tensor& input) {
  int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(
      K == context.sgemm_sizes_[1],
      "Check the shapes of mat1 and mat2, they cannot be multiplied!");
  auto input_ = input.contiguous();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(context.bias_);
  const at::Tensor& bias = *bias_maybe_owned;
  int64_t input_batch = (int64_t)(input_.numel() / K);

  // Since MKL prepack API only accepts fixed M/N/K, a repack is required
  // when M changes. To avoid frequently repacking the weights,
  // it will fall back to the MKL cblas_sgemm kernel when M-dim is
  // dynamically changed.
  if (input_batch != context.sgemm_sizes_[0])
    return mkl_sgemm_kernel(input_, context.ori_weight_, bias);
  return mkl_prepack_sgemm_kernel(
      input_, context.mkl_weight_, bias, context.sgemm_sizes_[2]);
}

at::Tensor& run(
    ContextLinearMKL& context,
    const at::Tensor& input,
    at::Tensor& accumu) {
  int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(
      K == context.sgemm_sizes_[1],
      "Check the shapes of mat1 and mat2, they cannot be multiplied!");
  auto input_ = input.contiguous();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(context.bias_);
  const at::Tensor& bias = *bias_maybe_owned;
  int64_t input_batch = (int64_t)(input_.numel() / K);
  if (input_batch != context.sgemm_sizes_[0])
    mkl_sgemm_kernel_output(input_, context.ori_weight_, bias, accumu);
  mkl_prepack_sgemm_kernel_output(
      input_, context.mkl_weight_, bias, context.sgemm_sizes_[2], accumu);
  return accumu;
}

at::Tensor pack(ContextLinearMKL& context, const at::Tensor& tensor) {
  auto batch_size = context.sgemm_sizes_[0];
  auto in_features = context.sgemm_sizes_[1];
  auto out_features = context.sgemm_sizes_[2];
  return mkl_sgemm_pack_weight(batch_size, out_features, in_features, tensor);
}

} // namespace mkl_sgemm
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
