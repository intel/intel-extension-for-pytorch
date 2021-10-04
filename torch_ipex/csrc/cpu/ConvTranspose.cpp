#include "ConvTranspose.h"
#include <torch/extension.h>
#include "mkldnn/MKLDNNCommon.h"
#include "mkldnn/MKLDNNConversions.h"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include "torch_ipex/csrc/utils.h"

namespace torch_ipex {
namespace cpu {

constexpr int output_batch_size_dim = 0; // also grad_output
constexpr int weight_input_channels_dim = 1;

std::vector<int64_t> conv_input_size(
    at::IntArrayRef output_size,
    at::IntArrayRef weight_size,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] -
        (2 * padding[d - 2]) + kernel + output_padding[d - 2];
  }
  return input_size;
}

at::Tensor convolution_transpose_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  auto output_sizes = conv_input_size(
      input.sizes(),
      weight.sizes(),
      padding,
      output_padding,
      stride,
      dilation,
      groups);
  auto output = at::empty({0}, input.options());

  const ideep::tensor x = itensor_from_tensor(input);
  ideep::tensor w = itensor_from_tensor(weight);
  // mkldnn transposed convolution has weight in logical order of OIHW or OIDHW,
  // while PyTorch has IOHW or IODHW, `._tranpose()` switches strides (no memory
  // copy).
  w.transpose_(0, 1);

  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, input.suggest_memory_format());
    y = itensor_from_tensor(output);
  }
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        b,
        output_sizes,
        y,
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups);
  } else {
    ideep::convolution_transpose_forward::compute(
        x,
        w,
        output_sizes,
        y,
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups);
  }

  if (!is_channels_last) {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
        std::move(y),
        optTypeMetaToScalarType(input.options().dtype_opt()),
        input.options().device_opt()));
  } else {
    TORCH_INTERNAL_ASSERT(y.get_desc().is_nhwc());
    return output;
  }
}

} // namespace cpu
} // namespace torch_ipex