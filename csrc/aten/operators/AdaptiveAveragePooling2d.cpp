#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"

#include <vector>

using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void adaptive_avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  TORCH_CHECK(
      (input.ndimension() == 4), "only support 4 dims on DPCPP device now!");

  // bool ceil_mode = false;
  auto nOutputCols = output_size[1];
  auto nOutputRows = output_size[0];

  // Input is NCHW format
  auto nInputCols = input.size(3);
  auto nInputRows = input.size(2);
  auto nInputPlane = input.size(1);
  auto batchSize = input.size(0);

  int dW = DPCPP::floor((float)2 * nInputCols / nOutputCols) -
      DPCPP::floor((float)nInputCols / nOutputCols);
  int dH = DPCPP::floor((float)2 * nInputRows / nOutputRows) -
      DPCPP::floor((float)nInputRows / nOutputRows);

  int kW = DPCPP::ceil((float)2 * nInputCols / nOutputCols) -
      DPCPP::floor((float)nInputCols / nOutputCols);
  int kH = DPCPP::ceil((float)2 * nInputRows / nOutputRows) -
      DPCPP::floor((float)nInputRows / nOutputRows);

  int padW = (dW * (nOutputCols - 1) + kW - nInputCols) / 2;
  int padH = (dH * (nOutputRows - 1) + kH - nInputRows) / 2;

  Tensor input_ = input;
  auto smf = input.suggest_memory_format();
  if (is_smf_channels_last(input)) {
    output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols}, smf);
  } else {
    input_ = input.contiguous();
    output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});
  }

  ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      output,
      input_,
      batchSize,
      nInputPlane,
      0,
      nInputRows,
      nInputCols,
      0,
      nOutputRows,
      nOutputCols,
      0,
      kH,
      kW,
      0,
      dH,
      dW,
      0,
      padH,
      padW);
}

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  TORCH_CHECK(
      (input.ndimension() == 4), "only support 4 dims on DPCPP device now!");
  Tensor gradOutput;
  /* resize */
  auto smf = input.suggest_memory_format();
  if (is_smf_channels_last(input)) {
    gradInput.resize_as_(input, smf);
    gradOutput = gradOutput_.contiguous(smf);
  } else {
    gradInput.resize_as_(input);
    gradOutput = gradOutput_.contiguous();
  }

  auto output_size_vec = gradOutput.sizes();
  auto nOutputCols = output_size_vec[3];
  auto nOutputRows = output_size_vec[2];

  // Input is NCHW format
  auto nInputCols = input.size(3);
  auto nInputRows = input.size(2);
  auto nInputPlane = input.size(1);
  auto batchSize = input.size(0);

  int dW = DPCPP::floor((float)2 * nInputCols / nOutputCols) -
      DPCPP::floor((float)nInputCols / nOutputCols);
  int dH = DPCPP::floor((float)2 * nInputRows / nOutputRows) -
      DPCPP::floor((float)nInputRows / nOutputRows);

  int kW = DPCPP::ceil((float)2 * nInputCols / nOutputCols) -
      DPCPP::floor((float)nInputCols / nOutputCols);
  int kH = DPCPP::ceil((float)2 * nInputRows / nOutputRows) -
      DPCPP::floor((float)nInputRows / nOutputRows);

  int padW = (dW * (nOutputCols - 1) + kW - nInputCols) / 2;
  int padH = (dH * (nOutputRows - 1) + kH - nInputRows) / 2;

  auto alg_kind = algorithm::pooling_avg_exclude_padding;

  ::xpu::oneDNN::pooling_backward<
      ::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
      gradInput,
      gradOutput,
      input,
      batchSize,
      nInputPlane,
      0,
      nInputRows,
      nInputCols,
      0,
      nOutputRows,
      nOutputCols,
      0,
      kH,
      kW,
      0,
      dH,
      dW,
      0,
      padH,
      padW);
}

} // namespace impl

Tensor& adaptive_avg_pool2d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef output_size) {
  impl::adaptive_avg_pool2d_out_template(out, self, output_size);
  return out;
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if (self.is_quantized()) {
    output = _empty_affine_quantized(
        {0},
        self.options(),
        self.q_scale(),
        self.q_zero_point(),
        MemoryFormat::Contiguous);
  } else {
    output = at::empty({0}, self.options());
  }

  return at::AtenIpexTypeXPU::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if (self.is_quantized()) {
    output = _empty_affine_quantized(
        {0},
        self.options(),
        self.q_scale(),
        self.q_zero_point(),
        MemoryFormat::Contiguous);
  } else {
    output = at::empty({0}, self.options());
  }

  return at::AtenIpexTypeXPU::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor& adaptive_avg_pool2d_backward_out_dpcpp(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input) {
  impl::adaptive_avg_pool2d_backward_out_template(gradInput, gradOutput, input);
  return gradInput;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  auto smf = self.suggest_memory_format();
  Tensor grad_input = is_smf_channels_last(self)
      ? at::empty_like(self, smf)
      : at::empty_like(self, MemoryFormat::Contiguous);
  impl::adaptive_avg_pool2d_backward_out_template(
      grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor& adaptive_avg_pool2d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef output_size) {
  at::AtenIpexTypeXPU::impl::adaptive_avg_pool2d_out_template(
      out, self, output_size);
  return out;
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  output = _empty_affine_quantized(
      {0},
      self.options(),
      self.q_scale(),
      self.q_zero_point(),
      MemoryFormat::Contiguous);
  return at::AtenIpexTypeXPU::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  output = _empty_affine_quantized(
      {0},
      self.options(),
      self.q_scale(),
      self.q_zero_point(),
      MemoryFormat::Contiguous);
  return at::AtenIpexTypeXPU::adaptive_avg_pool2d_out(
      output, self, output_size);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
