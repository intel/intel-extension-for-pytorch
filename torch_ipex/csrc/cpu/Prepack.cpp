#include "Prepack.h"
#include "dbl/Common.h"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/utils.h"

namespace torch_ipex {

using namespace cpu::dbl::comm;

void AtenIpexPrepack::prepack_conv_weight(
    at::Tensor &weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(weight.device().type() == at::DeviceType::DPCPP,
              "Cannot prepack a non-dpcpp tensor. Call t.to('dpcpp') first.");

  auto kdims = weight.dim() - 2;
  auto stride_vec = expand_param_if_needed(stride, "stride", kdims);
  auto padding_vec = expand_param_if_needed(padding, "padding", kdims);
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", kdims);

  auto packed_desc =
      dil::convolution_forward::expected_weights_desc(
          weight.sizes().vec(),
          torch_ipex::get_dil_data_type(weight.scalar_type()),
          stride_vec,
          padding_vec,
          padding_vec,
          dilation_vec,
          groups);

  bridge::reorderDilTensorGeneric(weight, packed_desc);
}

}  // namespace torch_ipex