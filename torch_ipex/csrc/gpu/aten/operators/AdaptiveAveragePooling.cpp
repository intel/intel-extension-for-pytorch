#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>
#include <utils/ATDispatch.h>
#include "Pooling.hpp"

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

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;

  // bool ceil_mode = false;
  int64_t nOutputCols = output_size[1];
  int64_t nOutputRows = output_size[0];

  // Input is NCHW format
  nInputCols = input.size(3);
  nInputRows = input.size(2);
  nInputPlane = input.size(1);
  batchSize = input.size(0);

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
  auto prop_kind = dnnl::prop_kind::forward_training;

  Tensor input_ = input.contiguous();

  output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_.scalar_type(),
      "adaptive_avg_pool2d",
      [&] {
        avg_pool_out_frame<scalar_t>(
            input,
            output,
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
            padW,
            alg_kind,
            prop_kind);
      });
}

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  Tensor gradOutput = gradOutput_.contiguous();

  TORCH_CHECK(
      (input.ndimension() == 4), "only support 4 dims on DPCPP device now!");

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;

  auto output_size_vec = gradOutput.sizes();
  int64_t nOutputCols = output_size_vec[3];
  int64_t nOutputRows = output_size_vec[2];

  // Input is NCHW format
  nInputCols = input.size(3);
  nInputRows = input.size(2);
  nInputPlane = input.size(1);
  batchSize = input.size(0);

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
  auto prop_kind = dnnl::prop_kind::forward_training;

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "adaptive_avg_pool2d_backward",
      [&] {
        auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
        auto gradInput_data = gradInput.data_ptr<scalar_t>();
        avg_pool_backward_out_frame<scalar_t>(
            gradInput_data,
            gradOutput_data,
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
            padW,
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
  gradInput.resize_as_(input).zero_();
  impl::adaptive_avg_pool2d_backward_out_template(gradInput, gradOutput, input);
  return gradInput;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  auto grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  impl::adaptive_avg_pool2d_backward_out_template(
      grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
