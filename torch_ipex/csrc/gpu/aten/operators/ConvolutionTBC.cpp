#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <tuple>

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>

#include <ATen/aten_ipex_type_dpcpp.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
void ConvolutionTBC_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t pad) {
  TORCH_CHECK(input.dim() == 3, "Input must have 3 dims: time, batch, in_channel");
  TORCH_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width, in_channels, out_channels.");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  auto real_pad = (olen - ilen + kw - 1) / 2;

  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  //Bias = (out_channels)
  TORCH_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) is not == dim 1 in the weight tensor");
  TORCH_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in the weight tensor (output channels).");

  // input * weight + bias -> output_features
  output.resize_({olen, input_size[1], weight_size[2]});
  output.copy_(bias.expand(output.sizes()));
  for (auto k = 0; k < kw; k++) {
      auto iShift = std::max(0, static_cast<int>(k - real_pad));
      auto oShift = std::max(0, static_cast<int>(real_pad - k));
      auto t = std::min(ilen + real_pad - k, olen) - oShift;
      // Note: gemm assumes column-major matrices
      // input      is l*m (row-major)
      // weight   is m*r (row-major)
      // output   is l*r (row-major)
      if (t > 0) {
          auto W = weight[k];
          auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
          auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
          O.addmm_(I, W);
      }
  }
}

template <typename scalar_t>
void ConvolutionTBC_updateGradInput(
    Tensor& dInput,
    Tensor& dWeight,
    Tensor& dBias,
    const Tensor& dOutput,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t pad) {
  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  auto real_pad = (olen - ilen + kw - 1) / 2;

  dInput.resize_as_(input);
  dInput.fill_(0);
  for (int k = 0; k < kw; k++) {
      auto iShift = std::max(0, static_cast<int>(k - real_pad));
      auto oShift = std::max(0, static_cast<int>(real_pad - k));
      auto t = std::min(ilen + real_pad - k, olen) - oShift;
      // dOutput * T(weight) -> dInput
      if (t > 0) {
          auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
          auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
          dI.addmm_(dO, weight[k].t());
      }
  }

  dWeight.resize_as_(weight);
  dWeight.fill_(0);
  for (int k = 0; k < kw; k++) {
      auto iShift = std::max(0, static_cast<int>(k - real_pad));
      auto oShift = std::max(0, static_cast<int>(real_pad - k));
      auto t = std::min(ilen + real_pad - k, olen) - oShift;
      // T(input) * dOutput -> dWeight
      if (t > 0) {
          auto dW = dWeight[k];
          auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
          auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
          dW.addmm_(I, dO);
      }
  }

  dBias.resize_as_(bias);
  dBias.fill_(0);
  auto tmp = dOutput.sum(0, false);
  dBias.copy_(tmp.sum(0));
}

} // namespace impl
//namespace AtenIpexTypeDPCPP

Tensor conv_tbc(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    int64_t pad) {
  Tensor out = at::empty({}, self.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "conv_tbc", [&] {
          impl::ConvolutionTBC_updateOutput<scalar_t>(
              out, self, weight, bias, pad);
      });
  return out;
}

std::tuple<Tensor, Tensor, Tensor>conv_tbc_backward(
    const Tensor& self,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t pad) {
  Tensor dInput = at::empty({}, input.options());
  Tensor dWeight = at::empty({}, weight.options());
  Tensor dBias = at::empty({}, bias.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "conv_tbc_backward", [&] {
          impl::ConvolutionTBC_updateGradInput<scalar_t>(
              dInput, dWeight, dBias, self, input, weight, bias, pad);
      });
  return std::make_tuple(dInput, dWeight, dBias);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at