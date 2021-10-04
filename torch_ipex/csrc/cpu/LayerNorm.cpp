#include "LayerNorm.h"
#include "mkldnn/MKLDNNCommon.h"

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
std::tuple<at::Tensor, at::Tensor, at::Tensor> dil_native_layer_norm_impl(
    const at::Tensor &X, const at::Tensor &gamma /* optional */,
    const at::Tensor &beta /* optional */, int64_t M, int64_t N, double eps) {
  ideep::tensor x = itensor_view_from_dense(X);
  auto gamma_fp32 = gamma.to(at::kFloat);
  auto beta_fp32 = beta.to(at::kFloat);
  const ideep::tensor scale = itensor_view_from_dense(gamma_fp32);
  const ideep::tensor shift = itensor_view_from_dense(beta_fp32);
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

} // namespace cpu
} // namespace torch_ipex
