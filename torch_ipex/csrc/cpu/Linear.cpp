#include "Linear.h"
#include "mkldnn/MKLDNNCommon.h"

namespace torch_ipex {
namespace cpu {

at::Tensor linear_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {

  const int64_t dim = self.dim();
  // reshape first if input dim != 2 and the reshape will cost a memory copy.
  auto self_reshaped =
      dim == 2 ? self : self.reshape({-1, self.size(self.dim() - 1)});

  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(self_reshaped);
  const ideep::tensor mkldnn_weight = at::native::itensor_view_from_dense(weight);

  std::vector<int64_t> output_size_reshaped = {self_reshaped.size(0), weight.size(0)};
  auto output = at::empty(output_size_reshaped, self.options());
  ideep::tensor mkldnn_output;
  mkldnn_output = at::native::itensor_view_from_dense(output);

  if (bias.defined()) {
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias);
    ideep::inner_product_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        mkldnn_output,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::inner_product_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_output,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() != 2) {
    return output.reshape(output_size);
  }
  return output;
}

}  // namespace cpu
}  // namespace torch_ipex