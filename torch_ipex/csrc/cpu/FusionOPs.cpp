#include "torch_ipex/csrc/cpu/FusionOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <ATen/InferSize.h>
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

at::Tensor AtenIpexJITDev::dil_convolution_relu(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  auto input_contiguous = input.contiguous();
  auto weight_contiguous = weight.contiguous();

  dil_input = dbl::comm::try_gen_dil_tensor(input_contiguous);
  dil_weight = dbl::comm::try_gen_dil_tensor(weight_contiguous);
  if (bias.defined()) {
    auto bias_contiguous = bias.contiguous();
    dil_bias = dbl::comm::try_gen_dil_tensor(bias_contiguous);
  }

  dil::tensor dil_output = dbl::conv::conv2d_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups,
    dil::attr_t::fuse_relu());

  return dbl::comm::gen_aten_tensor_by(dil_output);
}

}  // namespace cpu
}  // namespace torch_ipex
