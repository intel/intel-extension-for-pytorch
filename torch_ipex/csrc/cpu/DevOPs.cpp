#include "torch_ipex/csrc/cpu/DevOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <ATen/InferSize.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <limits>

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "dbl/Common.h"
#include "dbl/Conv.h"
#include "dbl/Deconv.h"
#include "dbl/Pool.h"
#include "dbl/DNNLChecker.h"
#include "dbl/Linear.h"
#include "ShadeDataContext.h"

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

//#define DBG
#if defined(DBG)
#define DEBUG(fmt) printf(fmt);
#else
#define DEBUG(fmt)
#endif

#define CHECK_DNNL_OP_PRE_COND(tensor)                                               \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.defined());                                \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.layout() == c10::kStrided)

at::Tensor AtenIpexCPUDev::dil_convolution(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  DEBUG("AtenIpexCPUDev::dil_convolution\n");
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);

  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(input, /* uint8_used for output*/false);
    if (quantized) {
      output_scale.push_back(scales[1]);
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
      dbl::comm::reorder_to_int8_for_mix_prec(weight, {});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight);
  }

  dil_input = dbl::comm::try_gen_dil_tensor(input);
  dbl::conv::prepack_conv_weights(input, dil_input,
      weight, stride, padding, dilation, groups);
 
  if (bias.defined()) {
    CHECK_DNNL_OP_PRE_COND(bias);
    if (!check_auto_mix_int8_fp32()) {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias);
    }
    dil_bias = dbl::comm::try_gen_dil_tensor(bias);
  }

  dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  dil::tensor dil_output = dbl::conv::convolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups,
    dil::attr_t(),
    output_scale);

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_output));

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(input, aten_output, "Convolution");
  }

  return aten_output;
}

at::Tensor dil_convolution_backward_input(
    at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // for training case, grad_output can be cpu tensor or MKLDNN tensor,
  // but weight and bias always cpu tensor
  auto dil_grad_output = dbl::comm::try_gen_dil_tensor(grad_output);
  auto dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  dil::tensor dil_grad_input;
  dil::convolution_backward_data::compute(
      dil_grad_output,
      dil_weight,
      input_size.vec(),
      dil_grad_input,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);
  return dbl::comm::gen_aten_tensor_by(std::move(dil_grad_input));
}

std::tuple<at::Tensor, at::Tensor> dil_convolution_backward_weights(
    const at::Tensor& weight, const at::Tensor& grad_output, const at::Tensor& input,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // for training case, grad_output and input can be cpu tensor or MKLDNN tensor,
  // but weight and bias always cpu tensor
  const dil::tensor dil_grad_output = dbl::comm::try_gen_dil_tensor(grad_output);
  const dil::tensor dil_input = dbl::comm::try_gen_dil_tensor(input);

  dil::tensor dil_grad_weight, dil_grad_bias;
  dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  auto diff_weight_type = w.get_data_type();
  auto weight_size = weight.sizes();

  if (bias_defined) {
    dil::convolution_backward_weights::compute(
        dil_input,
        dil_grad_output,
        weight_size.vec(),
        dil_grad_weight,
        dil_grad_bias,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        diff_weight_type);
    return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(dil_grad_weight)),
        dbl::comm::gen_aten_tensor_by(std::move(dil_grad_bias)));
  } else {
    dil::convolution_backward_weights::compute(
        dil_input,
        dil_grad_output,
        weight_size.vec(),
        dil_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        diff_weight_type);
    return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(dil_grad_weight)),
        at::Tensor());
  }
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::dil_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  DEBUG("AtenIpexCPUDev::dil_convolution_backward\n");
  at::Tensor grad_output = grad_output_t.is_contiguous() ? grad_output_t : grad_output_t.contiguous();
  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = dil_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = dil_convolution_backward_weights(
      weight, grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor AtenIpexCPUDev::dil_deconvolution(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  DEBUG("AtenIpexCPUDev::dil_deconvolution\n");
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);

  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dil_input = dbl::comm::try_gen_dil_tensor(input);

  if (bias.defined()) {
    CHECK_DNNL_OP_PRE_COND(bias);
    dbl::comm::reorder_to_bf16_for_mix_prec(bias);
    dil_bias = dbl::comm::try_gen_dil_tensor(bias);
  }

  dbl::comm::reorder_to_bf16_for_mix_prec(weight);;
  dbl::deconv::prepack_deconv_weights(
      input, weight, stride, padding, output_padding, dilation, groups, bias.defined());
  dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  dil::tensor dil_output = dbl::deconv::deconvolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    output_padding,
    stride,
    dilation,
    groups);

  return dbl::comm::gen_aten_tensor_by(std::move(dil_output));
}

at::Tensor AtenIpexCPUDev::dil_convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  DEBUG("AtenIpexCPUDev::convolution_overrideable\n");
  try {
    if (check_auto_dnnl()) {
      std::vector<at::Tensor> dnnl_input_tensors;
      dnnl_input_tensors.push_back(input);
      dnnl_input_tensors.push_back(weight);
      if (bias.defined()) {
        dnnl_input_tensors.push_back(bias);
      }
      if (dbl::chk::dnnl_support_the_tensors(dnnl_input_tensors)) {
        if (transposed) {
          return AtenIpexCPUDev::dil_deconvolution(input.is_contiguous() ? input : input.contiguous(), weight.is_contiguous() ? weight : weight.contiguous(), bias.defined() && !bias.is_contiguous() ? bias.contiguous() : bias, padding, output_padding, stride, dilation, groups);
        } else {
          // for int8 path, input always acbd format which is non-contiguous, .contiguous() will reorder to fp32
          return AtenIpexCPUDev::dil_convolution(input, weight, bias, stride, padding, dilation, groups);
        }
      }
    }
  } catch (std::exception& e) {
#if defined(_DEBUG)
    TORCH_WARN(e.what());
#endif
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bias.layout() == c10::kStrided);
  auto&& _ipex_input = bridge::shallowFallbackToCPUTensor(input);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_bias = bridge::shallowFallbackToCPUTensor(bias);
  auto&& _ipex_result = at::convolution(_ipex_input, _ipex_weight, _ipex_bias, stride, padding, dilation, transposed, output_padding, groups);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::dil_convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::convolution_backward_overrideable\n");
  // NOTE: DO NOT always call contiguous. It may break lazy-reorder. Because contiguous will call reorder instantly.
  if (check_auto_dnnl()) {
    if (transposed) {
      IPEX_CHECK(false, "deconvolution backward not support for dnnl path now");
    } else {
      return AtenIpexCPUDev::dil_convolution_backward(
        input.is_contiguous() ? input : input.contiguous(),
        grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
        weight.is_contiguous() ? weight : weight.contiguous(),
        padding,
        stride,
        dilation,
        groups,
        output_mask);
    }
  } else {
    if (transposed) {
      IPEX_CHECK(false, "deconvolution backward not support for native path now");
    } else {
      return AtenIpexCPUDev::mkldnn_convolution_backward(input, grad_output, weight, padding, stride, dilation, groups, output_mask);
    }
  }
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::mkldnn_convolution_backward\n");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_grad_output = bridge::shallowFallbackToCPUTensor(grad_output);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_result = at::mkldnn_convolution_backward(_ipex_self.contiguous(), _ipex_grad_output.contiguous(), _ipex_weight.contiguous(), padding, stride, dilation, groups, output_mask);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return std::tuple<at::Tensor,at::Tensor,at::Tensor>(bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<2>(_ipex_result)));
}

template<bool inplace>
at::Tensor& dil_add_common(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);

  IPEX_CHECK(self.sizes().equals(other.sizes()),
      "dil add not support broadcast yet");

  if (check_auto_mix_int8_fp32()) {
    // for accuracy, reorder int8 to fp32 
    dbl::comm::reorder_to_dtype(self, at::kFloat);
    dbl::comm::reorder_to_dtype(other, at::kFloat);
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(self);
    dbl::comm::reorder_to_bf16_for_mix_prec(other);
  }

  auto x = dbl::comm::try_gen_dil_tensor(self);
  auto y = dbl::comm::try_gen_dil_tensor(other);
  auto z = inplace ? x : dil::tensor();

  dil::sum::compute({1.0, alpha.to<float>()}, {x, y}, z);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, z);
  }
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_add_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add_out\n");

  return dil_add_common</*inplace=*/false>(result, self, other, alpha);
}

at::Tensor AtenIpexCPUDev::dil_add(const at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_add_common</*inplace=*/false>(result, self, other, alpha);
}

at::Tensor & AtenIpexCPUDev::dil_add_(at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add_\n");

  return dil_add_common</*inplace=*/true>(self, self, other, alpha);
}

template<bool inplace>
at::Tensor& dil_mul_common(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);

  // IPEX_CHECK(self.sizes().equals(other.sizes()),
  //     "dil mul not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(other);

  auto x = dbl::comm::try_gen_dil_tensor(self);
  auto y = dbl::comm::try_gen_dil_tensor(other);
  auto z = inplace ? x : dil::tensor();

  dil::binary::compute(x, y, z, dil::algorithm::binary_mul);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, z);
  }
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_mul_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul_out\n");

  return dil_mul_common</*inplace=*/false>(result, self, other);
}

at::Tensor AtenIpexCPUDev::dil_mul(const at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_mul_common</*inplace=*/false>(result, self, other);
}

at::Tensor& AtenIpexCPUDev::dil_mul_(at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul_\n");

  return dil_mul_common</*inplace=*/true>(self, self, other);
}

void matmul_common(
    const dil::tensor &x,
    const dil::tensor &w,
    const dil::tensor &bias,
    dil::tensor &y,
    at::Scalar beta=1,
    at::Scalar alpha=1,
    const dil::attr_t& attr = dil::attr_t()) {
  DEBUG("AtenIpexCPUDev::matmul_common\n");
  float dst_coeff = alpha.to<float>();
  float sum_coeff = beta.to<float>();
  if (!bias.is_empty()) {
    // DNNL only supports bias in 1xN dims
    // use bias for sum can save tensor memory copy
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.get_dim(0) == 1) {
      dil::matmul_forward::compute(x, w, bias, y);
      return;
    }
    dil::direct_copy::compute(bias, y);
  }

  dil::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
      dil::scale_t(), dil::scale_t(), dil::scale_t(), attr);
}

at::Tensor AtenIpexCPUDev::dil_bmm(const at::Tensor& self, const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_bmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_bmm_out(result, self, mat2);
}

at::Tensor& AtenIpexCPUDev::dil_bmm_out(at::Tensor &result, const at::Tensor& batch1, const at::Tensor& batch2) {
  DEBUG("AtenIpexCPUDev::dil_bmm_out\n");
  CHECK_DNNL_OP_PRE_COND(batch1);
  CHECK_DNNL_OP_PRE_COND(batch2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch1.dim() == 3 && batch2.dim() == 3);
  dil::dims inferred_size{batch1.size(0), batch1.size(1), batch2.size(2)};

  dbl::comm::reorder_to_bf16_for_mix_prec(batch1);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch2);

  auto x = dbl::comm::try_gen_dil_tensor(batch1);
  auto w = dbl::comm::try_gen_dil_tensor(batch2);
  dil::tensor y;
  matmul_common(x, w, dil::tensor(), y);

  dbl::comm::equip_dil_buffer(result, y);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor AtenIpexCPUDev::dil_mm(const at::Tensor& self, const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_mm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_mm_out(result, self, mat2);
}

at::Tensor& AtenIpexCPUDev::dil_mm_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_mm_out\n");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.dim() == 2 && mat2.dim() == 2);
  dil::dims inferred_size{self.size(0), mat2.size(1)};

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(mat2);

  auto x = dbl::comm::try_gen_dil_tensor(self);
  auto w = dbl::comm::try_gen_dil_tensor(mat2);
  dil::tensor y;
  matmul_common(x, w, dil::tensor(), y);

  dbl::comm::equip_dil_buffer(result, y);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

template <bool inplace>
at::Tensor& dil_baddbmm_common(
    at::Tensor &result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(batch1);
  CHECK_DNNL_OP_PRE_COND(batch2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch1.dim() == 3 && batch2.dim() == 3);
  dil::dims inferred_size{batch1.size(0), batch1.size(1), batch2.size(2)};
  IPEX_CHECK(self.sizes().equals(inferred_size),
      "dil baddbmm not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch1);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch2);

  auto x = dbl::comm::try_gen_dil_tensor(batch1);
  auto w = dbl::comm::try_gen_dil_tensor(batch2);
  dil::tensor bias;
  if (self.numel() != 0) {
    bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.ndims() < x.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  auto y = inplace ? dbl::comm::try_gen_dil_tensor(self) : dil::tensor();
  auto attr_ = dil::attr_t::fuse_sum();
  matmul_common(x, w, bias, y, beta, alpha, attr_);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, y);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_baddbmm_out(
    at::Tensor &result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_baddbmm_out\n");

  return dil_baddbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_baddbmm(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_baddbmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_baddbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_baddbmm_(at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_baddbmm_\n");

  return dil_baddbmm_out(self, self, batch1, batch2, beta, alpha);
}

template<bool inplace>
at::Tensor& dil_addmm_common(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::Scalar beta,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(mat1);
  CHECK_DNNL_OP_PRE_COND(mat2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2 && mat2.dim() == 2);
  dil::dims inferred_size{mat1.size(0), mat2.size(1)};
  IPEX_CHECK(self.sizes().equals(inferred_size),
      "dil addmm not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(mat1);
  dbl::comm::reorder_to_bf16_for_mix_prec(mat2);

  auto x = dbl::comm::try_gen_dil_tensor(mat1);
  auto w = dbl::comm::try_gen_dil_tensor(mat2);
  dil::tensor bias;
  if (self.numel() != 0) {
    bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.ndims() < x.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  auto y = inplace ? dbl::comm::try_gen_dil_tensor(self) : dil::tensor();
  auto attr_ = dil::attr_t::fuse_sum();
  matmul_common(x, w, bias, y, beta, alpha, attr_);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, y);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_addmm_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addmm_out\n");

  return dil_addmm_common</*inplace=*/false>(result, self, mat1, mat2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_addmm_common</*inplace=*/false>(result, self, mat1, mat2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_addmm_(at::Tensor& self, const at::Tensor& mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addmm_\n");

  return dil_addmm_common</*inplace=*/false>(self, self, mat1, mat2, beta, alpha);
}

template<bool inplace>
at::Tensor& dil_addbmm_common(
    at::Tensor& result,
    const at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(batch1);
  CHECK_DNNL_OP_PRE_COND(batch2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch1.dim() == 3 && batch2.dim() == 3);
  dil::dims inferred_size{batch1.size(1), batch2.size(2)};
  IPEX_CHECK(self.sizes().equals(inferred_size),
      "dil addbmm not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch1);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch2);

  // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
  // [n, b*m] * [b*m, p] = [n, p]
  // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
  // For batch2: reshape from [b, m, p] to [b*m, p]
  auto x = dbl::comm::try_gen_dil_tensor(batch1);
  auto w = dbl::comm::try_gen_dil_tensor(batch2);

  auto x_ = x;
  if (x.get_dim(0) > 1) {
    x_ = x.transpose(0, 1);
  }
  x_ = x_.reshape({x.get_dim(1), x.get_dim(0) * x.get_dim(2)});
  auto w_ = w.reshape({w.get_dim(0) * w.get_dim(1), w.get_dim(2)});
  auto y = inplace ? dbl::comm::try_gen_dil_tensor(self) : dil::tensor();
  auto attr_ = dil::attr_t::fuse_sum();

  dil::tensor bias;
  if (self.numel() != 0) {
    bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.ndims() < x_.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  matmul_common(x_, w_, bias, y, beta, alpha, attr_);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, y);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_addbmm_out(at::Tensor& result, const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm_out\n");

  return dil_addbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_addbmm(const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_addbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_addbmm_(at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm_\n");

  return dil_addbmm_common</*inplace=*/true>(self, self, batch1, batch2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_linear(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  DEBUG("AtenIpexCPUDev::dil_linear\n");
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(weight);
  IPEX_CHECK(self.dim() >= 2,
      "dil_linear: input needs to has dim at least 2, input dim ", self.dim());

  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(self, /*  uint8_used for output*/false);
    ///quantized = false;
    if (quantized) {
      output_scale.push_back(scales[1]);
      dbl::comm::reorder_to_int8_for_mix_prec(self, {scales[0]});
      dbl::comm::reorder_to_int8_for_mix_prec(weight, {});
    } else {
      dbl::comm::reorder_to_dtype(self, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(self);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight);
  }

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? dil_reshape(self, {-1, self.size(self.dim() - 1)}) : self;
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(self_reshaped);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);

  c10::optional<dil::tensor> b{c10::nullopt};
  
  if (bias.defined()) {
    if (!check_auto_mix_int8_fp32()) {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias);
    }
    b = dbl::comm::try_gen_dil_tensor(bias);
  }

  dil::tensor y = dbl::linear::linear_impl(x, w, b, output_scale);

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(y));

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(self, aten_output, "Linear");
  }
 
  if (self.dim() > 2) {
    return aten_output.reshape(output_size);
  }
  return aten_output;
}

at::Tensor AtenIpexCPUDev::dil_linear_backward_input(
    at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight){
  DEBUG("AtenIpexCPUDev::dil_linear_backward_input\n");

  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(weight);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight);

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    dil_reshape(grad_output, {-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_reshaped);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  dil::tensor gradx;
  dil::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  if (input_size.size() > 2) {
    return dbl::comm::gen_aten_tensor_by(std::move(gradx)).reshape(input_size);
  }
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

std::tuple<at::Tensor, at::Tensor> AtenIpexCPUDev::dil_linear_backward_weights(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, bool bias_defined) {
  DEBUG("AtenIpexCPUDev::dil_linear_backward_weights\n");

  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(weight);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight);

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    dil_reshape(grad_output, {-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? dil_reshape(input, {-1, input.size(input.dim() - 1)}) : input;

  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_reshaped);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input_reshaped);
  dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  auto diff_weight_type = w.get_data_type();

  dil::tensor gradw, gradb;
  if (bias_defined) {
    dil::inner_product_backward_weights::compute(x, grady, gradw, gradb, diff_weight_type);
    return std::tuple<at::Tensor, at::Tensor>{
    dbl::comm::gen_aten_tensor_by(std::move(gradw)),
    dbl::comm::gen_aten_tensor_by(std::move(gradb))};
  } else {
    dil::inner_product_backward_weights::compute(x, grady, gradw, diff_weight_type);
    return std::tuple<at::Tensor, at::Tensor>{
    dbl::comm::gen_aten_tensor_by(std::move(gradw)),
    at::Tensor()};
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_linear_backward(
    const at::Tensor& input, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::dil_linear_backward\n");

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = dil_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = dil_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight, grad_bias};
}

std::tuple<at::Tensor, at::Tensor> _dil_dropout(
    const at::Tensor& self,
    double ratio) {
  IPEX_CHECK(
      ratio >= 0 && ratio < 1 && self.numel() != 0,
      "dropout probability has to be between 0 and 1, but got ",
      ratio);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor mask;
  dil::tensor y;
  dil::dropout_forward::compute(x, ratio, y, mask);
  return std::tuple<at::Tensor, at::Tensor>{
      dbl::comm::gen_aten_tensor_by(std::move(y)),
      dbl::comm::gen_aten_tensor_by(std::move(mask))};
}

at::Tensor AtenIpexCPUDev::dil_dropout(const at::Tensor& self, double ratio, bool train) {
  DEBUG("AtenIpexCPUDev::dil_dropout\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  return std::get<0>(_dil_dropout(self, ratio));
}

at::Tensor AtenIpexCPUDev::dil_dropout_backward(
    const at::Tensor& grady,
    const at::Tensor& mask,
    double ratio) {
  DEBUG("AtenIpexCPUDev::dil_dropout_backward\n");
  CHECK_DNNL_OP_PRE_COND(grady);
  if (ratio == 0 || grady.numel() == 0) {
    return grady;
  }

  dbl::comm::reorder_to_bf16_for_mix_prec(grady);
  dbl::comm::reorder_to_bf16_for_mix_prec(mask);

  dil::tensor dY = dbl::comm::try_gen_dil_tensor(grady);
  dil::tensor mask_dil = dbl::comm::try_gen_dil_tensor(mask);
  dil::tensor dX;
  dil::dropout_backward::compute(mask_dil, dY, dX);
  return dbl::comm::gen_aten_tensor_by(std::move(dX));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  DEBUG("AtenIpexCPUDev::dil_native_batch_norm\n");
  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);
  IPEX_CHECK(input.dim() == 4 || input.dim() == 5,
             "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");
  IPEX_CHECK(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model");

  if (check_auto_mix_int8_fp32()) {
    IPEX_CHECK(!train, "mkldnn_bacth_norm: mkldnn only support inference model for int8");
  }
  std::vector<float> output_scale = {};
  bool quantized = false;
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(input, /*  uint8_used for output*/false);
    quantized = false;
    if (quantized) {
      output_scale.push_back(scales[1]);
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
  }

  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  const dil::tensor b = dbl::comm::try_gen_dil_tensor(bias);
  bool use_running_stat = (running_mean.defined() && running_var.defined());
  dil::tensor y;
  if (train) {
    dil::tensor saved_mean;
    dil::tensor saved_var;
    dil::batch_normalization_forward_training::compute(
        x, w, b, y, saved_mean, saved_var, momentum, eps);
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      dil::tensor m = dbl::comm::try_gen_dil_tensor(running_mean);
      dil::tensor v = dbl::comm::try_gen_dil_tensor(running_var);
      const std::vector<float> scales_mean{1 - float(momentum), float(momentum)};
      const std::vector<float> scales_var{1 - float(momentum), float(momentum) * len / (len - 1)};
      dil::sum::compute(scales_mean, {m, saved_mean}, m);
      dil::sum::compute(scales_var, {v, saved_var}, v);
    }
    return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(y)),
        dbl::comm::gen_aten_tensor_by(std::move(saved_mean)),
        dbl::comm::gen_aten_tensor_by(std::move(saved_var)));
  } else {
    if (use_running_stat) {
      dil::tensor m = dbl::comm::try_gen_dil_tensor(running_mean);
      dil::tensor v = dbl::comm::try_gen_dil_tensor(running_var);
      dil::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps);
    } else {
      dil::batch_normalization_forward_inference::compute(
          x, w, b, y, eps);
    }

    
    if (check_auto_mix_int8_fp32() && !check_int8_calibration() && quantized) {
      y.set_scale(output_scale);
    }
    auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(y));

    if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
      insert_or_updata_observer(input, aten_output, "BatchNorm");
    }

    return std::make_tuple(aten_output, at::Tensor(), at::Tensor());
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_batch_norm_backward(const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  DEBUG("AtenIpexCPUDev::dil_native_batch_norm_backward\n");
  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);

  IPEX_CHECK(train, "mkldnn_batch_norm_backward: currently mkldnn only support train model");
  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  dil::tensor m = dbl::comm::try_gen_dil_tensor(save_mean);
  dil::tensor v = dbl::comm::try_gen_dil_tensor(save_invstd);

  dil::tensor gradx, gradw, gradb;
  dil::batch_normalization_backward::compute(
      x, m, v, grady, w, gradx, gradw, gradb, eps);

  return std::make_tuple(
      dbl::comm::gen_aten_tensor_by(std::move(gradx)),
      dbl::comm::gen_aten_tensor_by(std::move(gradw)),
      dbl::comm::gen_aten_tensor_by(std::move(gradb)));
}

at::Tensor AtenIpexCPUDev::dil_max_pooling(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  DEBUG("AtenIpexCPUDev::dil_max_pooling\n");
  CHECK_DNNL_OP_PRE_COND(input);

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(input, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(input, at::Tensor(), "MaxPooling");
  }
  return dbl::pool::_dil_pooling(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      dil::algorithm::pooling_max);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool2d\n");
  CHECK_DNNL_OP_PRE_COND(input);
  IPEX_CHECK(!divisor_override.has_value(),
           "dil_avg_pooling operator does not support divisor");

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(input, /*  uint8_used for output*/false);
    quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(input, at::Tensor(), "AvgPool2d");
  }

  return dbl::pool::_dil_pooling(
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation*/ std::vector<int64_t> {1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool3d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool3d\n");
  CHECK_DNNL_OP_PRE_COND(input);
  IPEX_CHECK(!divisor_override.has_value(),
           "dil_avg_pooling operator does not support divisor");

  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  return dbl::pool::_dil_pooling(
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation*/ std::vector<int64_t> {1, 1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_adaptive_avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  DEBUG("AtenIpexCPUDev::dil_adaptive_avg_pool2d\n");
  CHECK_DNNL_OP_PRE_COND(input);

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(input, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
  }

  auto output_size_vec =
      dbl::comm::expand_param_if_needed(output_size, "output_size", input.dim() - 2);
  std::vector<int64_t> kernel_size(input.dim() - 2);
  for (int64_t i = 2; i < input.dim(); ++i) {
    auto s1 = input.size(i);
    auto s2 = output_size_vec[i - 2];
    IPEX_CHECK(s2 != 0, "output size can not be zero");
    IPEX_CHECK(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    kernel_size[i - 2] = s1 / s2;
  }
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};

  if (input.dim() == 5) {
    padding.push_back(0);
    dilation.push_back(1);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(input, at::Tensor(), "AdaptiveAvgPool2d");
  }
  return dbl::pool::_dil_pooling(
      input,
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ padding,
      /*dilation*/ dilation,
      /*ceil_mode*/ false,
      /*algo*/ dil::algorithm::pooling_avg);
}

at::Tensor AtenIpexCPUDev::dil_max_pooling_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  DEBUG("AtenIpexCPUDev::dil_max_pooling_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  return dbl::pool::_dil_pooling_backward(
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
      output.is_contiguous() ? output : output.contiguous(),
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      dil::algorithm::pooling_max);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool2d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  return dbl::pool::_dil_pooling_backward(
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation */ std::vector<int64_t>{1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool3d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  std::vector<int64_t> dilation{1, 1};
  return dbl::pool::_dil_pooling_backward(
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation */ std::vector<int64_t>{1, 1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_adaptive_avg_pool2d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  auto output_size_vec = grad_output.sizes();
  std::vector<int64_t> kernel_size(input.dim() - 2);
  for (size_t i = 2; i < input.dim(); ++i) {
    auto s1 = input.size(i);
    auto s2 = output_size_vec[i];
    IPEX_CHECK(s2 != 0, "output size can not be zero");
    IPEX_CHECK(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    kernel_size[i - 2] = s1 / s2;
  }
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};

  if (input.dim() == 5) {
    padding.push_back(0);
    dilation.push_back(1);
  }

  return dbl::pool::_dil_pooling_backward(
      grad_output,
      grad_output,
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ padding,
      /*dilation*/ dilation,
      false,
      /*algo*/ dil::algorithm::pooling_avg);
}

at::Tensor AtenIpexCPUDev::dil_relu(const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_relu\n");
  CHECK_DNNL_OP_PRE_COND(input);

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized)= dbl::comm::get_int8_scales(input, /*  uint8_used for output*/true);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
  }

  const dil::tensor& x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_relu, dil::prop_kind::forward_training, /*alpha*/ 0.0);

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(input, at::Tensor(), "Relu");
  }

  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor& AtenIpexCPUDev::dil_relu_(at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_relu_\n");
  CHECK_DNNL_OP_PRE_COND(input);

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<float> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales(input, /*   uint8_used for output*/true);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, {scales[0]});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer(input, at::Tensor(), "Relu_");
  }

  auto dil_self = dbl::comm::try_gen_dil_tensor(input);
  dil::eltwise_forward::compute(
    dil_self,
    dil_self,
    dil::algorithm::eltwise_relu,
    dil::prop_kind::forward_training,
    /*alpha*/ 0.0);
 
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dil_self.is_public_format() || check_tensor_own_whole_storage(input));
  dbl::comm::sync_shape_from_dil_to_aten(input, dil_self);
  return input;
}

at::Tensor AtenIpexCPUDev::dil_threshold_backward(const at::Tensor& grad_output, const at::Tensor& input, at::Scalar threshold) {
  DEBUG("AtenIpexCPUDev::dil_threshold_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);

  // TODO: support bounded relu. `threshold` is ignored for now
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output);
  dil::tensor gradx;
  dil::eltwise_backward::compute(x, grady, gradx,
      dil::algorithm::eltwise_relu, /*alpha*/ 0.0);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

at::Tensor AtenIpexCPUDev::dil__softmax(
    const at::Tensor& self,
    const int64_t dim,
    bool half_to_float) {
  DEBUG("AtenIpexCPUDev::dil__softmax\n");
  CHECK_DNNL_OP_PRE_COND(self);
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  dil::softmax_forward::compute(x, y, wrapped_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor AtenIpexCPUDev::dil__softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil__softmax_backward_data\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(output);
  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  dil::tensor y = dbl::comm::try_gen_dil_tensor(output);
  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gradx;
  dil::softmax_backward::compute(y, grady, gradx, wrapped_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

at::Tensor AtenIpexCPUDev::dil_sigmoid(const at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil_sigmoid\n");
  CHECK_DNNL_OP_PRE_COND(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_logistic_use_dst_for_bwd, dil::prop_kind::forward);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor& AtenIpexCPUDev::dil_sigmoid_(at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil_sigmoid_\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::eltwise_forward::compute(
      x, x, dil::algorithm::eltwise_logistic_use_dst_for_bwd, dil::prop_kind::forward);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(x.is_public_format() || check_tensor_own_whole_storage(self));
  dbl::comm::sync_shape_from_dil_to_aten(self, x);
  return self;
}

at::Tensor AtenIpexCPUDev::dil_sigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  DEBUG("AtenIpexCPUDev::dil_sigmoid_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(output);

  dil::tensor y = dbl::comm::try_gen_dil_tensor(output);
  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dil::tensor gy = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gx;
  dil::eltwise_backward::compute(y, gy, gx,
      dil::algorithm::eltwise_logistic_use_dst_for_bwd);
  return dbl::comm::gen_aten_tensor_by(std::move(gx));
}

at::Tensor AtenIpexCPUDev::dil_reshape(const at::Tensor& self, at::IntArrayRef size) {
  DEBUG("AtenIpexCPUDev::dil_reshape\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y{x};
  y.reshape(inferred_size);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

int64_t AtenIpexCPUDev::dil_size(const at::Tensor & self, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_size\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dim = at::maybe_wrap_dim(dim, self.dim(), false);
  return self.sizes()[dim];
}

at::Tensor AtenIpexCPUDev::dil_clone(const at::Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  DEBUG("AtenIpexCPUDev::dil_clone\n");
  CHECK_DNNL_OP_PRE_COND(self);

  auto memory_format =
      optional_memory_format.value_or(at::MemoryFormat::Preserve);

  if (memory_format == at::MemoryFormat::Preserve) {
    if (!self.is_non_overlapping_and_dense()) {
      memory_format = self.suggest_memory_format();
    }
  }

  IPEX_CHECK(
      memory_format == at::MemoryFormat::Preserve ||
          memory_format == at::MemoryFormat::Contiguous,
      "dil_clone only support Preserve of Contiguous");

  auto src = dbl::comm::try_gen_dil_tensor(self);
  auto dst_desc =
      memory_format == at::MemoryFormat::Preserve
          ? src.get_desc()
          : src.get_desc().to_default_format();
  dil::tensor dst{dst_desc};
  src.reorder_to(dst);

  if (src.has_scale()) {
    dst.set_scale(src.get_scale());
  }
  if (src.has_zero_point()) {
    dst.set_zero_point(src.get_zero_point());
  }
  if (src.has_workspace()) {
    dst.copy_workspace(src);
  }
  return dbl::comm::gen_aten_tensor_by(std::move(dst));
}

inline void check_cat_no_zero_dim(at::TensorList tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    IPEX_CHECK(t.dim() > 0,
      "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
  }
}

at::Tensor& AtenIpexCPUDev::dil_cat_out(at::Tensor& result, at::TensorList tensors, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_cat_out\n");
  CHECK_DNNL_OP_PRE_COND(result);

  dbl::comm::reorder_to_bf16_for_mix_prec(result);

  check_cat_no_zero_dim(tensors);
  dim = at::legacy_cat_wrap_dim(dim, tensors);
  std::vector<dil::tensor> x;
  for (auto i =0; i< tensors.size(); i++) {
    IPEX_CHECK(!(tensors[i].dim() == 1 && tensors[i].sizes()[0] == 0),
      "Currently Mkldnn cat operators do not support empty tensor.");

    dbl::comm::reorder_to_bf16_for_mix_prec(tensors[i]);

    x.push_back(dbl::comm::try_gen_dil_tensor(tensors[i]));
  }
  dil::tensor y = dbl::comm::try_gen_dil_tensor(result);
  dil::concat::compute(x, dim, y);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(y.is_public_format() || check_tensor_own_whole_storage(result));
  dbl::comm::sync_shape_from_dil_to_aten(result, y);
  return result;
}

at::Tensor AtenIpexCPUDev::dil_cat(at::TensorList tensors, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_cat\n");
  check_cat_no_zero_dim(tensors);
  dim = at::legacy_cat_wrap_dim(dim, tensors);
  std::vector<dil::tensor> x;
  at::Tensor tensors_contiguous[tensors.size()];
  for (auto i = 0; i < tensors.size(); i++) {
    IPEX_CHECK(!(tensors[i].dim() == 1 && tensors[i].sizes()[0] == 0),
      "Currently Mkldnn cat operators do not support empty tensor.");
    tensors_contiguous[i] = tensors[i].is_contiguous() ? tensors[i] : tensors[i].contiguous();
    dbl::comm::reorder_to_bf16_for_mix_prec(tensors_contiguous[i]);
    x.push_back(dbl::comm::try_gen_dil_tensor(tensors_contiguous[i]));
  }
  dil::tensor y;
  dil::concat::compute(x, dim, y);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_split_with_sizes(const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_split_with_sizes\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  int64_t num_splits = split_sizes.size();
  std::vector<at::Tensor> splits(num_splits);
  std::vector<int32_t> sizes;
  for (auto i = 0; i < num_splits; i++) {
    auto length = split_sizes[i];
    IPEX_CHECK(length >= 0,
             "split_with_sizes expects split_sizes have only non-negative ",
             "entries, but got split_sizes=", split_sizes);
    sizes.push_back((int32_t)length);
  }

  dim = at::maybe_wrap_dim(dim, self.dim());
  auto y = dil::spliter::compute(x, sizes, dim, false);
  for (auto j = 0; j < num_splits; j++) {
    splits[j] = dbl::comm::gen_aten_tensor_by(std::move(y[j]));
  }
  return splits;
}


at::Tensor dil_as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_) {

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dbl::comm::try_gen_dil_tensor(self).is_public_format(),
      "Cannot set sizes and strides for DIL tensor with non-public format");

  // share storage
  auto* self_storage = self.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
  self_storage->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));
  auto result = at::detail::make_tensor<IPEXTensorImpl>(self.storage(), at::DispatchKey::DPCPPTensorId);

  auto* _tensor_impl = (IPEXTensorImpl *)result.unsafeGetTensorImpl();
  _tensor_impl->copy_meta_info(self.unsafeGetTensorImpl());
  _tensor_impl->copy_auto_grad(self.unsafeGetTensorImpl());

  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  _tensor_impl->set_strided(size, stride, storage_offset);
  return result;
}

at::Tensor& propagate_transposed_names(
    at::Tensor& result,
    const at::Tensor& other,
    int64_t dim0,
    int64_t dim1) {
  // Port from aten/src/ATen/native/TensorShape.cp
  if (other.has_names()) {
    auto names = other.names().vec();
    std::swap(names[dim0], names[dim1]);
    at::namedinference::propagate_names_if_nonempty(result, names);
  }
  return result;
}

at::Tensor AtenIpexCPUDev::dil_transpose(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  DEBUG("AtenIpexCPUDev::dil_transpose\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  auto ndims = self.dim();
  dim0 = at::maybe_wrap_dim(dim0, ndims);
  dim1 = at::maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  // TODO: support transposing a blocked tensor
  // Even if DIL support transposing a blocked tensor, we have no place to 
  // store a different desc for transposed view when storage are sharing.
  dbl::comm::reorder_to_public(self, /*remain_dtype=*/true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  IPEX_CHECK(x.ndims() > 0, "DNNL transpose cannot generate DNNL tensor for the input aten Tensor. input tensor dim: ", self.dim());

  auto trans_desc = x.get_desc().transpose(dim0, dim1);
  auto sizes = trans_desc.get_dims();
  auto strides = trans_desc.get_strides();
  auto result = dil_as_strided(self, sizes, strides, self.storage_offset());
  propagate_transposed_names(result, self, dim0, dim1);
  return result;
}

at::Tensor AtenIpexCPUDev::dil_slice(const at::Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  DEBUG("AtenIpexCPUDev::dil_slice\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  // Port from aten/src/ATen/native/TensorShape.cpp
  int64_t ndim = self.dim();
  if (ndim == 0) {
    AT_INDEX_ERROR("dil_slice() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;

  auto result = dil_as_strided(self, sizes, strides, storage_offset);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor AtenIpexCPUDev::dil_select(const at::Tensor & self, int64_t dim, int64_t index) {
  DEBUG("AtenIpexCPUDev::dil_select\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self);

  // We do not support `select` a DIL tensor with blocked format
  dbl::comm::reorder_to_public(self, /*remain_dtype=*/true);

  // Port from aten/src/ATen/native/TensorShape.cpp
  int64_t ndim = self.dim();
  if (ndim == 0) {
    AT_INDEX_ERROR("select() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    if (self.has_names() && self.names()[dim] != at::Dimname::wildcard()) {
      AT_INDEX_ERROR("select(): index ", index, " out of range for tensor of size ",
                     self.sizes(), " at dimension ", self.names()[dim]);
    }
    AT_INDEX_ERROR("select(): index ", index, " out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    IPEX_CHECK(false, "Got a sparse tensor in select. Should not reach here.");
    // return at::select(self, dim, index);
  }
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  auto result = dil_as_strided(self, sizes, strides, storage_offset);
  at::namedinference::propagate_names_except(result, self, {dim});
  return result;
}

at::Tensor alias_with_sizes_and_strides(
    const at::Tensor& self,
    const c10::IntArrayRef sizes,
    const c10::IntArrayRef strides) {

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dbl::comm::try_gen_dil_tensor(self).is_public_format(),
      "Cannot set sizes and strides for DIL tensor with non-public format");

  // share storage
  auto* self_storage = self.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
  self_storage->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));
  auto self_ = at::detail::make_tensor<IPEXTensorImpl>(self.storage(), at::DispatchKey::DPCPPTensorId);

  auto* _tensor_impl = (IPEXTensorImpl *)self_.unsafeGetTensorImpl();
  _tensor_impl->copy_meta_info(self.unsafeGetTensorImpl());
  _tensor_impl->copy_auto_grad(self.unsafeGetTensorImpl());

  auto storage_offset = self.storage_offset();
  _tensor_impl->set_strided(sizes, strides, storage_offset);

  at::namedinference::propagate_names(self_, self);
  return self_;
}

at::Tensor AtenIpexCPUDev::dil_view(const at::Tensor & self, at::IntArrayRef size) {
  DEBUG("AtenIpexCPUDev::dil_view\n");
  CHECK_DNNL_OP_PRE_COND(self);

  // We do not support reshaping (viewing) a DIL tensor with blocked format
  dbl::comm::reorder_to_public(self, /*remain_dtype=*/true);

  // Port from aten/src/ATen/native/TensorShape.cpp
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(),
                                          self.strides(),
                                          inferred_size);
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  return alias_with_sizes_and_strides(self, inferred_size, stride_value);
}

at::Tensor AtenIpexCPUDev::dil__unsafe_view(const at::Tensor & self, at::IntArrayRef size) {
  DEBUG("AtenIpexCPUDev::dil__unsafe_view\n");
  return dil_view(self, size);
}

at::Tensor AtenIpexCPUDev::dil_select(const at::Tensor & self, at::Dimname dim, int64_t index) {
  return dil_select(self, at::dimname_to_position(self, dim), index);
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_split(const at::Tensor& self, int64_t split_size, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_split\n");
  CHECK_DNNL_OP_PRE_COND(self);
  dim = at::maybe_wrap_dim(dim, self.dim());
  int64_t dim_size = self.size(dim);
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where split_size > dim_size
    // (returns a single split).  We might want to error here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  std::vector<int64_t> split_sizes(num_splits, split_size);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);
  split_sizes[num_splits-1] = last_split_size;
  return dil_split_with_sizes(self, split_sizes, dim);
}

at::Tensor AtenIpexCPUDev::dil_gelu(const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_gelu\n");
  CHECK_DNNL_OP_PRE_COND(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_gelu_tanh, dil::prop_kind::forward_training, /*alpha*/ 0.0);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor AtenIpexCPUDev::dil_gelu_backward(const at::Tensor& grad_output, const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_gelu_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output);
  dil::tensor gradx;
  dil::eltwise_backward::compute(x, grady, gradx,
      dil::algorithm::eltwise_gelu_tanh, /*alpha*/ 0.0);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_layer_norm(
    const at::Tensor& X,
    const at::Tensor& gamma /* optional */,
    const at::Tensor& beta /* optional */,
    int64_t M,
    int64_t N,
    double eps) {
  DEBUG("AtenIpexCPUDev::dil_native_layer_norm\n");
  CHECK_DNNL_OP_PRE_COND(X);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(X);
  const dil::tensor scale = dbl::comm::try_gen_dil_tensor(gamma);
  const dil::tensor shift = dbl::comm::try_gen_dil_tensor(beta);
  int64_t i = 0;
  auto j = X.size(0);
  std::vector<int64_t> input_size;
  while(j <= M) {
    input_size.push_back(X.size(i++));
    j *= X.size(i);
  }
  input_size.push_back(N);
  auto src = x.reshape(input_size);
  dil::tensor y, mean, variance;
  dil::layer_normalization_forward::compute(
        src,
        scale,
        shift,
        y,
        mean,
        variance,
        eps);
  return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(y)).reshape(X.sizes()),
        dbl::comm::gen_aten_tensor_by(std::move(mean)),
        dbl::comm::gen_aten_tensor_by(std::move(variance)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_layer_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  DEBUG("AtenIpexCPUDev::dil_native_layer_norm_backward\n");
  CHECK_DNNL_OP_PRE_COND(dY);
  CHECK_DNNL_OP_PRE_COND(X);
  dil::tensor dy = dbl::comm::try_gen_dil_tensor(dY);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(X);
  dil::tensor m = dbl::comm::try_gen_dil_tensor(mean);
  dil::tensor r = dbl::comm::try_gen_dil_tensor(rstd);
  dil::tensor g = dbl::comm::try_gen_dil_tensor(gamma);
  int64_t i = 0;
  auto j = X.size(0);
  std::vector<int64_t> input_size;
  while(j <= M) {
    input_size.push_back(X.size(i++));
    j *= X.size(i);
  }
  input_size.push_back(N);
  auto src = x.reshape(input_size);
  auto grady = dy.reshape(input_size);
  dil::tensor gradx, gradg, gradb;
  float eps = 1e-5;
  dil::layer_normalization_backward::compute(
        src, m, r, grady, g, gradx, gradg, gradb, eps);

  return std::make_tuple(
      dbl::comm::gen_aten_tensor_by(std::move(gradx)).reshape(X.sizes()),
      dbl::comm::gen_aten_tensor_by(std::move(gradg)),
      dbl::comm::gen_aten_tensor_by(std::move(gradb)));
}

}  // namespace cpu
}  // namespace torch_ipex
