#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/utils.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "PixelShuffle.h"

#include "csrc/autocast/autocast_mode.h"
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(pixel_shuffle_kernel_stub);
DEFINE_DISPATCH(pixel_shuffle_backward_kernel_stub);
DEFINE_DISPATCH(pixel_unshuffle_kernel_stub);
DEFINE_DISPATCH(pixel_unshuffle_backward_kernel_stub);

at::Tensor pixel_shuffle_cpu(const at::Tensor& self, int64_t upscale_factor) {
  // Format: (B1, ..., Bn), C, H, W
  std::vector<int64_t> output_sizes(
      self.sizes().begin(), self.sizes().end() - 3);
  output_sizes.insert(
      output_sizes.end(),
      {self.size(-3) / upscale_factor / upscale_factor,
       self.size(-2) * upscale_factor,
       self.size(-1) * upscale_factor});

  auto output = at::empty({0}, self.options());
  auto memory_format = self.suggest_memory_format();
  output.resize_(output_sizes, memory_format);
  auto input = self.contiguous(memory_format);

  // pointer to pixel_shuffle_kernel_impl(output, input, upscale_factor);
  pixel_shuffle_kernel_stub(kCPU, output, input, upscale_factor);

  return output;
}

at::Tensor pixel_shuffle_backward_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef input_sizes,
    int64_t upscale_factor) {
  auto grad_input = at::empty({0}, grad_output.options());
  auto memory_format = grad_output.suggest_memory_format();
  grad_input.resize_(input_sizes, memory_format);
  auto grad_output_ = grad_output.contiguous(memory_format);

  // pointer to pixel_shuffle_backward_kernel_impl(grad_input, grad_output_,
  // upscale_factor);
  pixel_shuffle_backward_kernel_stub(
      kCPU, grad_input, grad_output_, upscale_factor);

  return grad_input;
}

at::Tensor pixel_unshuffle_cpu(
    const at::Tensor& self,
    int64_t downscale_factor) {
  // Format: (B1, ..., Bn), C, H, W
  std::vector<int64_t> output_sizes(
      self.sizes().begin(), self.sizes().end() - 3);
  output_sizes.insert(
      output_sizes.end(),
      {self.size(-3) * downscale_factor * downscale_factor,
       self.size(-2) / downscale_factor,
       self.size(-1) / downscale_factor});

  auto output = at::empty({0}, self.options());
  auto memory_format = self.suggest_memory_format();
  output.resize_(output_sizes, memory_format);
  auto input = self.contiguous(memory_format);

  // pointer to pixel_unshuffle_kernel_impl(output, input, downscale_factor);
  pixel_unshuffle_kernel_stub(kCPU, output, input, downscale_factor);

  return output;
}

at::Tensor pixel_unshuffle_backward_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef input_sizes,
    int64_t downscale_factor) {
  auto grad_input = at::empty({0}, grad_output.options());
  auto memory_format = grad_output.suggest_memory_format();
  grad_input.resize_(input_sizes, memory_format);
  auto grad_output_ = grad_output.contiguous(memory_format);

  /*
  pointer to pixel_unshuffle_backward_kernel_impl(
      grad_input, grad_output_, downscale_factor);
  */
  pixel_unshuffle_backward_kernel_stub(
      kCPU, grad_input, grad_output_, downscale_factor);

  return grad_input;
}

at::Tensor pixel_shuffle(const at::Tensor& self, int64_t upscale_factor) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::pixel_shuffle\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::pixel_shuffle", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      self.dim() >= 3,
      "pixel_shuffle expects input to have at least 3 dimensions, but "
      "got input with ",
      self.dim(),
      " dimension(s)");
  TORCH_CHECK(
      upscale_factor > 0,
      "pixel_shuffle expects a positive upscale_factor, but got ",
      upscale_factor);
  int64_t c = self.size(-3);
  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  TORCH_CHECK(
      c % upscale_factor_squared == 0,
      "pixel_shuffle expects its input's 'channel' dimension to be "
      "divisible by the square of "
      "upscale_factor, but input.size(-3)=",
      c,
      " is not divisible by ",
      upscale_factor_squared);

  // NOTE: The original PR registers the math_pixel_shuffle as an
  // operator, and then this operator will be dispatched to
  // native_pixel_shuffle. After that, the native_pixel_shuffle will be
  // dispatched to pixel_shuffle_cpu for cpu device and to math_pixel_shuffle
  // for other devices.
  if (at::GradMode::is_enabled())
    return PixelShuffleOp::apply(self, upscale_factor);
  return PixelShuffleOp::_forward(self, upscale_factor);
}

at::Tensor pixel_unshuffle(const at::Tensor& self, int64_t downscale_factor) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::pixel_unshuffle\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::pixel_unshuffle", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      self.dim() >= 3,
      "pixel_unshuffle expects input to have at least 3 dimensions, "
      "but got input with ",
      self.dim(),
      " dimension(s)");
  TORCH_CHECK(
      downscale_factor > 0,
      "pixel_unshuffle expects a positive downscale_factor, but got ",
      downscale_factor);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  TORCH_CHECK(
      h % downscale_factor == 0,
      "pixel_unshuffle expects height to be divisible by "
      "downscale_factor, but input.size(-2)=",
      h,
      " is not divisible by ",
      downscale_factor);
  TORCH_CHECK(
      w % downscale_factor == 0,
      "pixel_unshuffle expects width to be divisible by "
      "downscale_factor, but input.size(-1)=",
      w,
      " is not divisible by ",
      downscale_factor);

  if (at::GradMode::is_enabled())
    return PixelUnshuffleOp::apply(self, downscale_factor);
  return PixelUnshuffleOp::_forward(self, downscale_factor);
}

at::Tensor PixelShuffleOp::_forward(
    const at::Tensor& self,
    int64_t upscale_factor) {
  IPEX_RECORD_FUNCTION(
      "PixelShuffleOp::_forward", c10::ArrayRef<c10::IValue>({}));

  return pixel_shuffle_cpu(self, upscale_factor);
}

at::Tensor PixelShuffleOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& self,
    int64_t upscale_factor) {
  IPEX_RECORD_FUNCTION(
      "PixelShuffleOp::forward", c10::ArrayRef<c10::IValue>({}));

  at::AutoNonVariableTypeMode g;
  ctx->saved_data["upscale_factor"] = upscale_factor;
  ctx->saved_data["input_sizes"] = self.sizes();
  return _forward(self, upscale_factor);
}

torch::autograd::tensor_list PixelShuffleOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  IPEX_RECORD_FUNCTION(
      "PixelShuffleOp::backward", c10::ArrayRef<c10::IValue>({}));

  at::Tensor grad_output = grad_outputs[0];
  int64_t upscale_factor = ctx->saved_data["upscale_factor"].toInt();
  auto input_sizes = ctx->saved_data["input_sizes"].toIntList().vec();
  return {
      pixel_shuffle_backward_cpu(grad_output, input_sizes, upscale_factor),
      at::Tensor()};
}

at::Tensor PixelUnshuffleOp::_forward(
    const at::Tensor& self,
    int64_t downscale_factor) {
  IPEX_RECORD_FUNCTION(
      "PixelUnshuffleOp::_forward", c10::ArrayRef<c10::IValue>({}));

  return pixel_unshuffle_cpu(self, downscale_factor);
}

at::Tensor PixelUnshuffleOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& self,
    int64_t downscale_factor) {
  IPEX_RECORD_FUNCTION(
      "PixelUnshuffleOp::forward", c10::ArrayRef<c10::IValue>({}));

  at::AutoNonVariableTypeMode g;
  ctx->saved_data["downscale_factor"] = downscale_factor;
  ctx->saved_data["input_sizes"] = self.sizes();
  return _forward(self, downscale_factor);
}

torch::autograd::tensor_list PixelUnshuffleOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  IPEX_RECORD_FUNCTION(
      "PixelUnshuffleOp::backward", c10::ArrayRef<c10::IValue>({}));

  at::Tensor grad_output = grad_outputs[0];
  int64_t downscale_factor = ctx->saved_data["downscale_factor"].toInt();
  auto input_sizes = ctx->saved_data["input_sizes"].toIntList().vec();
  return {
      pixel_unshuffle_backward_cpu(grad_output, input_sizes, downscale_factor),
      at::Tensor()};
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pixel_shuffle"),
      TORCH_FN((&torch_ipex::cpu::pixel_shuffle)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pixel_unshuffle"),
      TORCH_FN((&torch_ipex::cpu::pixel_unshuffle)));
}

} // namespace cpu
} // namespace torch_ipex