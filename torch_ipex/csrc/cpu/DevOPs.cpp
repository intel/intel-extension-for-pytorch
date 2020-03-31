#include "torch_ipex/csrc/cpu/DevOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <limits>

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "dbl/Common.h"
#include "dbl/Conv.h"
#include "ShadeDataContext.h"

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

#define DBG
#if defined(DBG)
#define DEBUG(fmt) printf(fmt);
#else
#define DEBUG(fmt)
#endif

#define CHECK_DNNL_OP_PRE_COND(tensor)                                    \
  TORCH_INTERNAL_ASSERT(tensor.defined());                                \
  TORCH_INTERNAL_ASSERT(tensor.device().type() == at::DeviceType::DPCPP); \
  TORCH_INTERNAL_ASSERT(tensor.is_contiguous());                          \
  TORCH_INTERNAL_ASSERT(tensor.layout() == c10::kStrided)

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
  dil_input = dbl::comm::try_gen_dil_tensor(input);
  dil_weight = dbl::comm::try_gen_dil_tensor(weight);
  if (bias.defined()) {
    CHECK_DNNL_OP_PRE_COND(bias);
    dil_bias = dbl::comm::try_gen_dil_tensor(bias);
  }

  dil::tensor dil_output = dbl::conv::conv2d_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups);

  return dbl::comm::gen_aten_tensor_by(dil_output);
}

at::Tensor& AtenIpexCPUDev::dil_relu_(at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_relu_\n");
  auto dil_self = dbl::comm::try_gen_dil_tensor(input);
  dil::eltwise_forward::compute(
    dil_self,
    dil_self,
    dil::algorithm::eltwise_relu,
    dil::prop_kind::forward_training,
    /*alpha*/ 0.0);
  return input;
}

at::Tensor AtenIpexCPUDev::convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  DEBUG("AtenIpexCPUDev::convolution_overrideable\n");
  // NOTE: DO NOT always call contiguous. It may break lazy-reorder. Because contiguous will call reorder instantly.
  if (check_auto_dnnl()) {
    return dil_convolution(
      input.is_contiguous() ? input : input.contiguous(),
      weight.is_contiguous() ? weight : weight.contiguous(),
      bias.defined() ? (bias.is_contiguous() ? bias :bias.contiguous()) : bias,
      stride,
      padding,
      dilation,
      groups);
  } else {
    return mkldnn_convolution(input, weight, bias, padding, stride, dilation, groups);
  }
}

at::Tensor AtenIpexCPUDev::mkldnn_convolution(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  DEBUG("AtenIpexCPUDev::mkldnn_convolution\n");
  TORCH_INTERNAL_ASSERT(self.defined());
  TORCH_INTERNAL_ASSERT(weight.defined());
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(weight.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(!(bias.defined()) || (bias.defined() && bias.layout() == c10::kStrided));
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_bias = bridge::shallowFallbackToCPUTensor(bias);
  auto&& _ipex_result = at::mkldnn_convolution(_ipex_self.contiguous(), _ipex_weight.contiguous(), _ipex_bias.contiguous(), padding, stride, dilation, groups);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  TORCH_INTERNAL_ASSERT(_ipex_result.is_contiguous());
  TORCH_INTERNAL_ASSERT(_ipex_result.layout() == c10::kStrided);
  return bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::convolution_backward_overrideable\n");
  return mkldnn_convolution_backward(input, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::mkldnn_convolution_backward\n");
  TORCH_INTERNAL_ASSERT(self.defined());
  TORCH_INTERNAL_ASSERT(grad_output.defined());
  TORCH_INTERNAL_ASSERT(weight.defined());
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(grad_output.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(weight.layout() == c10::kStrided);
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_grad_output = bridge::shallowFallbackToCPUTensor(grad_output);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_result = at::mkldnn_convolution_backward(_ipex_self.contiguous(), _ipex_grad_output.contiguous(), _ipex_weight.contiguous(), padding, stride, dilation, groups, output_mask);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return std::tuple<at::Tensor,at::Tensor,at::Tensor>(bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<2>(_ipex_result)));
}

at::Tensor& AtenIpexCPUDev::dil_add_out(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add_out\n");
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y = dbl::comm::try_gen_dil_tensor(other);

  dil::tensor z = dbl::comm::try_gen_dil_tensor(result);
  const std::vector<float> scales{1.0, alpha.to<float>()};
  dil::sum::compute(scales, {x, y}, z);

  return result;
}

at::Tensor AtenIpexCPUDev::dil_add(const at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add\n");
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y = dbl::comm::try_gen_dil_tensor(other);

  dil::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  dil::sum::compute(scales, {x, y}, z);

  return dbl::comm::gen_aten_tensor_by(z);
}

at::Tensor & AtenIpexCPUDev::dil_add_(at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add_\n");
  CHECK_DNNL_OP_PRE_COND(self);

  auto dil_self = dbl::comm::try_gen_dil_tensor(self);
  auto dil_other = dbl::comm::try_gen_dil_tensor(other);

  const std::vector<float> scales{1.0, alpha.to<float>()};
  dil::sum::compute(scales, {dil_self, dil_other}, dil_self);

  return self;
}

at::Tensor& AtenIpexCPUDev::dil_mul_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  CHECK_DNNL_OP_PRE_COND(result);
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);

  auto dil_result = dbl::comm::try_gen_dil_tensor(result);
  auto dil_self = dbl::comm::try_gen_dil_tensor(self);
  auto dil_other = dbl::comm::try_gen_dil_tensor(other);

  dil::binary::compute(dil_self, dil_other, dil_result, dil::algorithm::binary_mul);

  return result;
}

at::Tensor AtenIpexCPUDev::dil_mul(const at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul\n");
  at::Tensor result = dbl::comm::empty_dil_tensor(self.sizes(), self.options());
  return dil_mul_out(result, self, other);
}

at::Tensor& AtenIpexCPUDev::dil_mul_(at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul_\n");
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);
  return dil_mul_out(self, self, other);
}

at::Tensor AtenIpexCPUDev::dil_bmm(
    const at::Tensor& self, 
    const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_bmm\n");
  auto self_size = self.sizes();
  std::vector<int64_t> result_size(self_size.begin(), self_size.end()-1);
  result_size.push_back(mat2.size(-1));
  at::Tensor result = dbl::comm::empty_dil_tensor(result_size, self.options());
  return dil_bmm_out(result, self, mat2);
}

at::Tensor& AtenIpexCPUDev::dil_bmm_out(
    at::Tensor &result, 
    const at::Tensor& batch1, 
    const at::Tensor& batch2) {
  DEBUG("AtenIpexCPUDev::dil_bmm_out\n");
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(batch1);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(batch2);
  dil::tensor y = dbl::comm::try_gen_dil_tensor(result);
  dil::matmul_forward::compute(x, w, y);
  return result;
}

at::Tensor AtenIpexCPUDev::dil_mm(
    const at::Tensor& self,
    const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_mm\n");
  return dil_bmm(self, mat2);
}

at::Tensor& AtenIpexCPUDev::dil_mm_out(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_mm_out\n");
  return dil_bmm_out(result, self, mat2);
}

at::Tensor AtenIpexCPUDev::dil_baddbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor & batch2,
    at::Scalar beta,
    at::Scalar alpha) {
    DEBUG("AtenIpexCPUDev::dil_baddbmm\n");
    at::Tensor result = dbl::comm::empty_dil_tensor(self.sizes(), self.options());
    return dil_baddbmm_out(result, self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::baddbmm_common(
    at::Tensor &result,
    const dil::tensor &bias, 
    const dil::tensor &x,
    const dil::tensor &w,
    at::Scalar beta,
    at::Scalar alpha) {
    DEBUG("AtenIpexCPUDev::baddbmm_common\n");
    dil::tensor y = dbl::comm::try_gen_dil_tensor(result);
    float dst_coeff = alpha.to<float>();
    float sum_coeff = beta.to<float>();
    // DNNL only supports bias in 1xN dims
    // use bias for sum can save tensor memory copy 
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.get_dim(0) == 1) {
      dil::matmul_forward::compute(x, w, bias, y);
      return result;
    }

    dil::direct_copy::compute(bias, y);
    auto attr_ = dil::attr_t::fuse_sum();
    dil::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff);//, dil::scale_t(), dil::scale_t(), dil::scale_t(), attr_);
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
    const dil::tensor x = dbl::comm::try_gen_dil_tensor(batch1);
    const dil::tensor w = dbl::comm::try_gen_dil_tensor(batch2);
    dil::tensor bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.get_dims().size() < x.get_dims().size()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
    return baddbmm_common(result, bias, x, w, beta, alpha);
}

//Tensor& dil_baddbmm_

// dil_addmm will go to DNNL matmul jit path
at::Tensor AtenIpexCPUDev::dil_addmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor & batch2,
    at::Scalar beta,
    at::Scalar alpha) {
    DEBUG("AtenIpexCPUDev::dil_addmm\n");
    return dil_baddbmm(self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_addmm_out(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::Scalar beta,
    at::Scalar alpha) {    
  DEBUG("AtenIpexCPUDev::dil_addmm_out\n");  
  return dil_baddbmm_out(result, self, mat1, mat2, beta, alpha);
}

//Tensor& dil_addmm_

at::Tensor AtenIpexCPUDev::dil_addbmm(
    const at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm\n");
  at::Tensor result = dbl::comm::empty_dil_tensor(self.sizes(), self.options());
  return dil_addbmm_out(result, self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_addbmm_out(
    at::Tensor& result,
    const at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm_out\n");
  // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
  // [n, b*m] * [b*m, p] = [n, p]
  // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
  // For batch2: reshape from [b, m, p] to [b*m, p]
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(batch1);
  dil::tensor w = dbl::comm::try_gen_dil_tensor(batch2);

  auto x_ = x;
  if (x.get_dim(0) > 1) {
    auto x_desc = dil::tensor::desc(x.get_dims(), x.get_data_type(), dil::tag::bac);
    x_ = x.reorder_if_differ_in(x_desc);
  }
  dil::dims x_dims = {x.get_dim(1), x.get_dim(0) * x.get_dim(2)};
  x_ = x_.reshape(x_dims);

  dil::dims w_dims = {w.get_dim(0) * w.get_dim(1), w.get_dim(2)};
  auto w_ = w.reshape(w_dims);
   
  dil::tensor bias = dbl::comm::try_gen_dil_tensor(self);
  if (bias.get_dims().size() < x_.get_dims().size()) {
    auto bias_dims = bias.get_dims();
    bias_dims.insert(bias_dims.begin(), 1);
    bias.reshape(bias_dims);
  }
  return baddbmm_common(result, bias, x_, w_, beta, alpha);
}

//Tensor& dil_addbmm_

at::Tensor AtenIpexCPUDev::dil_linear(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  DEBUG("AtenIpexCPUDev::dil_linear\n");
  TORCH_CHECK(self.dim() >= 2,
      "dil_linear: input needs to has dim at least 2, input dim ", self.dim());
  TORCH_CHECK(self.is_mkldnn(),
      "dil_linear: input needs to be dil layout");

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? self.reshape({-1, self.size(self.dim() - 1)}) : self;
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(self_reshaped);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);

  dil::tensor y;
  if (bias.defined()) {
    const dil::tensor b = dbl::comm::try_gen_dil_tensor(bias);
    dil::inner_product_forward::compute(x, w, b, y);
  } else {
    dil::inner_product_forward::compute(x, w, y);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() > 2) {
    return dbl::comm::gen_aten_tensor_by(y).reshape(output_size);
  }
  return dbl::comm::gen_aten_tensor_by(y);
}

at::Tensor AtenIpexCPUDev::dil_linear_backward_input(
    at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight){
  DEBUG("AtenIpexCPUDev::dil_linear_backward_input\n");
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_reshaped);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  dil::tensor gradx;
  dil::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  if (input_size.size() > 2) {
    return dbl::comm::gen_aten_tensor_by(gradx).reshape(input_size);
  }
  return dbl::comm::gen_aten_tensor_by(gradx);
}

std::tuple<at::Tensor, at::Tensor> AtenIpexCPUDev::dil_linear_backward_weights(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, bool bias_defined) {
  DEBUG("AtenIpexCPUDev::dil_linear_backward_weights\n");
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_reshaped);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input_reshaped);
  dil::tensor gradw, gradb;
  if (bias_defined) {
    dil::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    dil::inner_product_backward_weights::compute(x, grady, gradw);
  }

  // Extract device info from weight and data type info from input.
  // since for current BF16 design, input is BF16 tensor while weight is FP32 tensor.  
  auto options = weight.options().dtype(input.scalar_type());
  if (weight.is_mkldnn()) {
    return std::tuple<at::Tensor, at::Tensor>{
      dbl::comm::gen_aten_tensor_by(gradw),
      dbl::comm::gen_aten_tensor_by(gradb)};
  } else {
    return std::tuple<at::Tensor, at::Tensor>{
      dbl::comm::dil_tensor_to_dense(dbl::comm::gen_aten_tensor_by(gradw)),
      dbl::comm::dil_tensor_to_dense(dbl::comm::gen_aten_tensor_by(gradb))};
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

}  // namespace cpu
}  // namespace torch_ipex
