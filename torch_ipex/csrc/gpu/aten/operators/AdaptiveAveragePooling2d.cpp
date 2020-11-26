#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>
#include <utils/ATDispatch.h>
#include "Pooling.hpp"

using namespace dnnl;
using namespace at::dpcpp;
namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

void adaptive_avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  TORCH_CHECK((input.ndimension() == 4), "only support 4 dims on DPCPP device now!");

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

  auto alg_kind = algorithm::pooling_avg_exclude_padding;
  auto prop_kind = dnnl::prop_kind::forward_training;

  Tensor input_ = input.contiguous();

  output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});

  if(!input_.is_quantized()){
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input_.scalar_type(),
        "adaptive_avg_pool2d",
        [&] {
          avg_pool_out_frame<scalar_t>(
              input_,
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
  } else {
    IPEX_DISPATCH_QINT_TYPES(
        input_.scalar_type(), "q_adaptive_avg_pool2d", [&] {
          avg_pool_out_frame<scalar_t>(
              input_,
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
}

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  Tensor gradOutput = gradOutput_.contiguous();

  TORCH_CHECK((input.ndimension() == 4), "only support 4 dims on DPCPP device now!");

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
  Tensor output;
  if(self.is_quantized()) {
    output = _empty_affine_quantized({0},
                self.options(),
                self.q_scale(),
                self.q_zero_point(),
                MemoryFormat::Contiguous);
  } else {
    output = at::empty({0}, self.options());
  }

  return at::AtenIpexTypeDPCPP::adaptive_avg_pool2d_out(output, self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if(self.is_quantized()) {
    output = _empty_affine_quantized({0},
                self.options(),
                self.q_scale(),
                self.q_zero_point(),
                MemoryFormat::Contiguous);
  } else {
    output = at::empty({0}, self.options());
  }

  return at::AtenIpexTypeDPCPP::adaptive_avg_pool2d_out(output, self, output_size);
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
  impl::adaptive_avg_pool2d_backward_out_template(grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
