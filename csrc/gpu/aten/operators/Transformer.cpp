#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <core/detail/TensorInfo.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor _transformer_encoder_layer_fwd(
    const Tensor& src,
    int64_t embed_dim,
    int64_t num_heads,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    bool use_gelu,
    bool norm_first,
    double eps,
    const Tensor& norm_weight_1,
    const Tensor& norm_bias_1,
    const Tensor& norm_weight_2,
    const Tensor& norm_bias_2,
    const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1,
    const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2,
    const c10::optional<Tensor>& mask,
    c10::optional<int64_t> mask_type) {
  return at::native::transformer_encoder_layer_forward(
      src,
      embed_dim,
      num_heads,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      use_gelu,
      norm_first,
      eps,
      norm_weight_1,
      norm_bias_1,
      norm_weight_2,
      norm_bias_2,
      ffn_weight_1,
      ffn_bias_1,
      ffn_weight_2,
      ffn_bias_2,
      mask,
      mask_type);
}

std::tuple<Tensor, Tensor, Tensor> _transformer_decoder_only_layer_fwd(
    const Tensor& src,
    int64_t embed_dim,
    int64_t num_heads,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    bool use_gelu,
    bool norm_first,
    double eps,
    const Tensor& norm_weight_1,
    const Tensor& norm_bias_1,
    const Tensor& norm_weight_2,
    const Tensor& norm_bias_2,
    const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1,
    const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2,
    const c10::optional<Tensor>& mask,
    const c10::optional<Tensor>& incr_key,
    const c10::optional<Tensor>& incr_value) {
  return at::native::transformer_decoder_only_layer_forward(
      src,
      embed_dim,
      num_heads,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      use_gelu,
      norm_first,
      eps,
      norm_weight_1,
      norm_bias_1,
      norm_weight_2,
      norm_bias_2,
      ffn_weight_1,
      ffn_bias_1,
      ffn_weight_2,
      ffn_bias_2,
      mask,
      incr_key,
      incr_value);
}

} // namespace AtenIpexTypeXPU
} // namespace at
