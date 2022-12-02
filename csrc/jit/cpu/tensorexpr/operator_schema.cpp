#include "operator_schema.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

const char kMmDivSchema[] =
    "ipex::matmul_div(Tensor left, Tensor right,  Tensor div_input) -> Tensor";
const char kConvNoneSchema[] =
    "ipex_prepack::convolution_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvReluSchema[] =
    "ipex_prepack::convolution_relu_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvAddReluSchema[] =
    "ipex_prepack::convolution_add_relu_run(Tensor input, Tensor(a!) accumu, *, Scalar? alpha, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvAbsSchema[] =
    "ipex_prepack::convolution_abs_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvClampSchema[] =
    "ipex_prepack::convolution_hardtanh_run(Tensor input, Scalar lower_bound, Scalar upper_bound, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvEluSchema[] =
    "ipex_prepack::convolution_elu_run(Tensor input, Scalar alpha, Scalar scale, Scalar input_scale, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvExpSchema[] =
    "ipex_prepack::convolution_exp_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvGeluSchema[] =
    "ipex_prepack::convolution_gelu_run(Tensor input, str approximate, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvHardswishSchema[] =
    "ipex_prepack::convolution_hardswish_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvLogSchema[] =
    "ipex_prepack::convolution_log_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvMishSchema[] =
    "ipex_prepack::convolution_mish_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvSigmoidSchema[] =
    "ipex_prepack::convolution_sigmoid_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvPowSchema[] =
    "ipex_prepack::convolution_pow_run(Tensor input, Scalar exponent, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvRoundSchema[] =
    "ipex_prepack::convolution_round_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvSqrtSchema[] =
    "ipex_prepack::convolution_sqrt_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvSquareSchema[] =
    "ipex_prepack::convolution_square_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvTanhSchema[] =
    "ipex_prepack::convolution_tanh_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvLeakyReluSchema[] =
    "ipex_prepack::convolution_leaky_relu_run(Tensor input, Scalar alpha, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
const char kConvSiluSchema[] =
    "ipex_prepack::convolution_swish_run(Tensor input, __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) -> Tensor";
} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
