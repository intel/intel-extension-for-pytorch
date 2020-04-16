#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>

#include "AveragePooling.hpp"

using namespace mkldnn;
using namespace at::dpcpp;
namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

void adaptive_avg_pool3d_out_template(
    Tensor& output,
    Tensor const& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool3d(): expected input to have non-empty spatial "
        "dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  int kD, kH, kW, dD, dH, dW;
  int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  int64_t nblock = input.size(-4);
  int64_t inputDepth = input.size(-3);
  int64_t inputHeight = input.size(-2);
  int64_t inputWidth = input.size(-1);

  int padD = 0;
  int padH = 0;
  int padW = 0;

  int64_t outputDepth = output_size[0];
  int64_t outputHeight = output_size[1];
  int64_t outputWidth = output_size[2];

  TORCH_CHECK(
      (inputDepth % outputDepth == 0),
      "depth input size is not divisible by the output size is not supported "
      "yet");
  TORCH_CHECK(
      (inputHeight % outputHeight == 0),
      "Height input size is not divisible by the output size is not supported "
      "yet");
  TORCH_CHECK(
      (inputWidth % outputWidth == 0),
      "Width input size is not divisible by the output size is not supported "
      "yet");

  kD = inputDepth / outputDepth;
  kH = inputHeight / outputHeight;
  kW = inputWidth / outputWidth;
  dD = kD;
  dH = kH;
  dW = kW;

  auto alg_kind = algorithm::pooling_avg;
  auto prop_kind = prop_kind::forward_training;

  if (input.ndimension() == 4) {
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool3d", [&] {
        auto input_data = input.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();
        avg_pool_out_frame<scalar_t>(
            input_data,
            output_data,
            nbatch,
            nblock,
            inputDepth,
            inputHeight,
            inputWidth,
            outputDepth,
            outputHeight,
            outputWidth,
            kD,
            kH,
            kW,
            dD,
            dH,
            dW,
            padD,
            padH,
            padW,
            alg_kind,
            prop_kind);
      });
}

Tensor& adaptive_avg_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  auto gradOutput = gradOutput_.contiguous();

  int kD, kH, kW, dD, dH, dW;
  int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  int64_t nblock = input.size(-4);
  int64_t gradInputDepth = input.size(-3);
  int64_t gradInputHeight = input.size(-2);
  int64_t gradInputWidth = input.size(-1);

  int padD = 0;
  int padH = 0;
  int padW = 0;

  int64_t gradOutputDepth = gradOutput.size(-3);
  int64_t gradOutputHeight = gradOutput.size(-2);
  int64_t gradOutputWidth = gradOutput.size(-1);

  TORCH_CHECK(
      (gradInputDepth % gradOutputDepth == 0),
      "backward: depth input size is not divisible by the output size is not "
      "supported yet");
  TORCH_CHECK(
      (gradInputHeight % gradOutputHeight == 0),
      "backward: Height input size is not divisible by the output size is not "
      "supported yet");
  TORCH_CHECK(
      (gradInputWidth % gradOutputWidth == 0),
      "backward: Width input size is not divisible by the output size is not "
      "supported yet");

  kD = gradInputDepth / gradOutputDepth;
  kH = gradInputHeight / gradOutputHeight;
  kW = gradInputWidth / gradOutputWidth;
  dD = kD;
  dH = kH;
  dW = kW;

  auto alg_kind = algorithm::pooling_avg;
  auto prop_kind = prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool3d_backward", [&] {
        /* get raw pointers */
        scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();

        avg_pool_backward_out_frame<scalar_t>(
            gradInput_data,
            gradOutput_data,
            nbatch,
            nblock,
            gradInputDepth,
            gradInputHeight,
            gradInputWidth,
            gradOutputDepth,
            gradOutputHeight,
            gradOutputWidth,
            kD,
            kH,
            kW,
            dD,
            dH,
            dW,
            padD,
            padH,
            padW,
            alg_kind,
            prop_kind);
      });
  return gradInput;
}

} // namespace impl

Tensor& adaptive_avg_pool3d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef output_size) {
  impl::adaptive_avg_pool3d_out_template(out, self, output_size);
  return out;
}

Tensor adaptive_avg_pool3d(const Tensor& self, IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::adaptive_avg_pool3d_out(
      output, self, output_size);
}

Tensor& adaptive_avg_pool3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self) {
  grad_input.resize_as_(self).zero_();
  impl::adaptive_avg_pool3d_backward_out_template(
      grad_input, grad_output, self);
  return grad_input;
}

Tensor adaptive_avg_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  auto grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  impl::adaptive_avg_pool3d_backward_out_template(
      grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at