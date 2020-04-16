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

void adaptive_avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  TORCH_CHECK(
      (input.ndimension() == 4), "only support 4 dims on DPCPP device now!");
  int kW, kH, dW, dH;
  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int padW = 0;
  int padH = 0;
  // bool ceil_mode = false;
  int64_t nOutputCols = output_size[1];
  int64_t nOutputRows = output_size[0];

  // Input is NCHW format
  nInputCols = input.size(3);
  nInputRows = input.size(2);
  nInputPlane = input.size(1);
  batchSize = input.size(0);

  TORCH_CHECK(
      (nInputRows % nOutputRows == 0),
      "row input size is not "
      "divisible by the output size "
      "is not supported yet");
  TORCH_CHECK(
      (nInputCols % nOutputCols == 0),
      "column input size is not "
      "divisible by the output size "
      "is not supported yet");

  kW = nInputCols / nOutputCols;
  kH = nInputRows / nOutputRows;
  dW = kW;
  dH = kH;

  auto alg_kind = algorithm::pooling_avg;
  auto prop_kind = dnnl::prop_kind::forward_training;

  output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool2d", [&] {
        auto input_data = input.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();
        avg_pool_out_frame<scalar_t>(
            input_data,
            output_data,
            batchSize,
            nInputPlane,
            0,
            nInputCols,
            nInputRows,
            0,
            nOutputCols,
            nOutputRows,
            0,
            kW,
            kH,
            0,
            dW,
            dH,
            0,
            padW,
            padH,
            alg_kind,
            prop_kind);
      });
}

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input) {
  TORCH_CHECK(
      (input.ndimension() == 4), "only support 4 dims on DPCPP device now!");
  int kW, kH, dW, dH;
  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int padW = 0;
  int padH = 0;
  auto output_size_vec = gradOutput.sizes();
  int64_t nOutputCols = output_size_vec[3];
  int64_t nOutputRows = output_size_vec[2];

  // Input is NCHW format
  nInputCols = input.size(3);
  nInputRows = input.size(2);
  nInputPlane = input.size(1);
  batchSize = input.size(0);

  TORCH_CHECK(
      (nInputRows % nOutputRows == 0),
      "row input size is not "
      "divisible by the output size "
      "is not supported yet");
  TORCH_CHECK(
      (nInputCols % nOutputCols == 0),
      "column input size is not "
      "divisible by the output size "
      "is not supported yet");

  kW = nInputCols / nOutputCols;
  kH = nInputRows / nOutputRows;
  dW = kW;
  dH = kH;

  auto alg_kind = algorithm::pooling_avg;
  auto prop_kind = dnnl::prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool2d_backward", [&] {
        auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
        auto gradInput_data = gradInput.data_ptr<scalar_t>();
        avg_pool_backward_out_frame<scalar_t>(
            gradInput_data,
            gradOutput_data,
            batchSize,
            nInputPlane,
            0,
            nInputCols,
            nInputRows,
            0,
            nOutputCols,
            nOutputRows,
            0,
            kW,
            kH,
            0,
            dW,
            dH,
            0,
            padW,
            padH,
            alg_kind,
            prop_kind);
      });
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
  auto output = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::adaptive_avg_pool2d_out(
      output, self, output_size);
}

Tensor& adaptive_avg_pool2d_backward_out_dpcpp(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input) {
  gradInput.resize_as_(input);
  impl::adaptive_avg_pool2d_backward_out_template(gradInput, gradOutput, input);
  return gradInput;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  auto grad_input = at::zeros_like(self);
  impl::adaptive_avg_pool2d_backward_out_template(
      grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
