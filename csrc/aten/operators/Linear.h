#include "BlasImpl.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor& dpcpp_linear_out(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output);

// IPEX customer linear for weight prepack
Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt);

Tensor& linear_out(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    Tensor& output);

Tensor linear_pow(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar exponent);

Tensor linear_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar negative_slope);

Tensor linear_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar minval,
    Scalar maxval);

Tensor linear_elu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale);

} // namespace AtenIpexTypeXPU
} // namespace at
