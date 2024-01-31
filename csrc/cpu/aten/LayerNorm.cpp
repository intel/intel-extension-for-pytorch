#include "LayerNorm.h"
#include <torch/all.h>
#include "../cpu/utils/isa_utils.h"
#include "ideep/IDeepConversions.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

/**layer_norm kernel for inference mode with oneDNN implementation
 *
 * @param X: input tensor for layernorm
 * @param gamma: scale for layernorm
 * @param beta: shift for layernorm
 * @param M
 * @param N
 * @param eps
 **/
std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_impl(
    const at::Tensor& X,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t M,
    int64_t N,
    double eps) {
  TORCH_CHECK(
      gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
      "gamma adn beta's data type should be float");
  ideep::tensor x = itensor_view_from_dense(X);
  const ideep::tensor scale = itensor_view_from_dense(gamma);
  const ideep::tensor shift = itensor_view_from_dense(beta);
  int64_t i = 0;
  auto dim = at::maybe_wrap_dim(0, X.dim(), false);
  auto j = X.sizes()[dim];
  std::vector<int64_t> input_size;
  while (j <= M) {
    dim = at::maybe_wrap_dim(i++, X.dim(), false);
    input_size.push_back(X.sizes()[dim]);
    dim = at::maybe_wrap_dim(i, X.dim(), false);
    j *= X.sizes()[dim];
  }
  input_size.push_back(N);
  auto src = x.reshape(input_size);
  at::Tensor Y = at::native::empty_like(X);
  at::Tensor mean = at::empty({M}, X.options());
  at::Tensor variance = at::empty({M}, X.options());
  auto onednn_Y = itensor_view_from_dense(Y);
  auto onednn_mean = itensor_view_from_dense(mean);
  auto onednn_variance = itensor_view_from_dense(variance);
  ideep::layer_normalization_forward::compute(
      src, scale, shift, onednn_Y, onednn_mean, onednn_variance, eps);
  return std::make_tuple(Y, mean, variance);
}

/**
 *prepare inputs for dil_layernorm
 *
 *@param input: the source tensor to layernorm
 *@param normalized_shape: input shape from an expected input of size
 *@param weight: scale tensor for layernorm
 *@param bias: shift tensor for layernorm
 *
 *@return inputs for dil_layernorm.
 **/
std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t>
_prepare_layer_norm_inputs(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());
  ;

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();
  return std::make_tuple(X, gamma, beta, M, N);
}

at::Tensor layer_norm_forward(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  auto inputs =
      _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto X = std::get<0>(inputs);
  auto gamma = std::get<1>(inputs);
  auto beta = std::get<2>(inputs);
  auto M = std::get<3>(inputs);
  auto N = std::get<4>(inputs);
  return std::get<0>(layer_norm_impl(X, gamma, beta, M, N, eps));
}

/**
 * at::layer_norm performance drop due to
 * #PR https://github.com/pytorch/pytorch/pull/59987
 * This is a workaround for layernorm regression.
 * Replace at::layer_norm with ipex::layernorm in jit pass for inference.
 * Now, we only use oneDNN kernel when both weight and bias are provided.
 * ToDo: more scenarios to use oneDNN or remvoe this pass
 * when at::layer_norm performance is back compared to w/o
 * mergeing https://github.com/pytorch/pytorch/pull/59987
 *
 * @param input: the source tensor to layernorm
 * @param normalized_shape: input shape from an expected input of size
 * @param weight_opt: scale tensor for layernorm
 * @param bias_opt: shift tensor for layernorm
 * @param bias: a value added to the denominator for numerical stability.
 * Default: 1e-5
 *
 * return: output for layernorm
 */
at::Tensor layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps,
    bool cudnn_enable) {
  RECORD_FUNCTION("torch_ipex::layer_norm", c10::ArrayRef<c10::IValue>({}));

  // onednn path for inference.
  // TODO: enable training path ??
  if (weight_opt.has_value() && weight_opt.value().defined() &&
      weight_opt.value().scalar_type() == at::kFloat && bias_opt.has_value() &&
      bias_opt.value().defined() &&
      bias_opt.value().scalar_type() == at::kFloat &&
      !at::GradMode::is_enabled() && input.dim() >= 2 && input.dim() <= 5 &&
      (input.scalar_type() != at::kHalf ||
       (input.scalar_type() == at::kHalf &&
        utils::isa_has_avx512_fp16_support()))) {
    return layer_norm_forward(
        input, normalized_shape, weight_opt.value(), bias_opt.value(), eps);
  }

  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  at::Tensor output = std::get<0>(
      at::native_layer_norm(input, normalized_shape, weight, bias, eps));
  return output;
}

} // namespace cpu
} // namespace torch_ipex

namespace {

// replace aten::layer_norm with ipex layer_norm.
IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::layer_norm"),
      TORCH_FN((&torch_ipex::cpu::layer_norm)));
}

} // namespace