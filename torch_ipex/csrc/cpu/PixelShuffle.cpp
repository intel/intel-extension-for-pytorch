#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/utils.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "PixelShuffle.h"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"

namespace torch_ipex {
namespace cpu {

template <typename scalar_t>
void cpu_pixel_shuffle(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t upscale_factor) {
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // [(B1...Bn), C, H, W] => [N, C, H, W]
  int64_t channels = input.size(-3);
  int64_t height = input.size(-2);
  int64_t width = input.size(-1);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = input.numel();
  int64_t nbatch = numel / (channels * height * width);
  int64_t S = upscale_factor;

  // input strides
  int64_t stride_n = channels * height * width;
  int64_t stride_c = S * S * height * width;
  int64_t stride_s1 = S * height * width;
  int64_t stride_s2 = height * width;
  int64_t stride_h = width;
  int64_t stride_w = 1;

  // input tensor shape of [n, c, s1, s2, h, w]
  // output tensor shape of [n, c, h, s1, w, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, h{0}, s1{0}, w{0}, s2{0};
    at::native::data_index_init(
        begin, n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);

    for (int64_t i = begin; i < end; i++) {
      int64_t input_offset = n * stride_n + c * stride_c + s1 * stride_s1 +
          s2 * stride_s2 + h * stride_h + w * stride_w;
      output_data[i] = input_data[input_offset];

      at::native::data_index_step(
          n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_channels_last(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t upscale_factor) {
  TORCH_CHECK(
      input.ndimension() == 4,
      "pixel shuffle with channels last format supports tensors with 4 dims");
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = input.numel();
  int64_t S = upscale_factor;

  // input strides
  int64_t stride_n = height * width * channels;
  int64_t stride_h = width * channels;
  int64_t stride_w = channels;
  int64_t stride_c = S * S;
  int64_t stride_s1 = S;
  int64_t stride_s2 = 1;

  // input tensor shape of [n, h, w, c, s1, s2]
  // output tensor shape of [n, h, s1, w, s2, c]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, h{0}, s1{0}, w{0}, s2{0}, c{0};
    at::native::data_index_init(
        begin, n, nbatch, h, height, s1, S, w, width, s2, S, c, sub_channels);

    for (int64_t i = begin; i < end; i++) {
      int64_t input_offset = n * stride_n + h * stride_h + w * stride_w +
          c * stride_c + s1 * stride_s1 + s2 * stride_s2;
      output_data[i] = input_data[input_offset];

      at::native::data_index_step(
          n, nbatch, h, height, s1, S, w, width, s2, S, c, sub_channels);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_backward(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t upscale_factor) {
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  // [(B1...Bn), C, H, W] => [N, C, H, W]
  int64_t channels = grad_input.size(-3);
  int64_t height = grad_input.size(-2);
  int64_t width = grad_input.size(-1);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = grad_input.numel();
  int64_t nbatch = numel / (channels * height * width);
  int64_t S = upscale_factor;

  // grad_output strides
  int64_t stride_n = channels * height * width;
  int64_t stride_c = height * S * width * S;
  int64_t stride_h = S * width * S;
  int64_t stride_s1 = width * S;
  int64_t stride_w = S;
  int64_t stride_s2 = 1;

  // grad_output tensor shape of [n, c, h, s1, w, s2]
  // grad_input tensor shape of [n, c, s1, s2, h, w]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, s1{0}, s2{0}, h{0}, w{0};
    at::native::data_index_init(
        begin, n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);

    for (int64_t i = begin; i < end; i++) {
      int64_t output_offset = n * stride_n + c * stride_c + h * stride_h +
          s1 * stride_s1 + w * stride_w + s2 * stride_s2;
      grad_input_data[i] = grad_output_data[output_offset];

      at::native::data_index_step(
          n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);
    }
  });
}

template <typename scalar_t>
void cpu_pixel_shuffle_backward_channels_last(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t upscale_factor) {
  TORCH_CHECK(
      grad_output.ndimension() == 4,
      "pixel shuffle with channels last format supports tensors with 4 dims");
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t height = grad_input.size(2);
  int64_t width = grad_input.size(3);
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);
  int64_t numel = grad_input.numel();
  int64_t S = upscale_factor;

  // grad_output strides
  int64_t stride_n = height * width * channels;
  int64_t stride_h = S * width * S * sub_channels;
  int64_t stride_s1 = width * S * sub_channels;
  int64_t stride_w = S * sub_channels;
  int64_t stride_s2 = sub_channels;
  int64_t stride_c = 1;

  // grad_output tensor shape of [n, h, s1, w, s2, c]
  // grad_input tensor shape of [n, h, w, c, s1, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, h{0}, w{0}, c{0}, s1{0}, s2{0};
    at::native::data_index_init(
        begin, n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);

    for (int64_t i = begin; i < end; i++) {
      int64_t output_offset = n * stride_n + h * stride_h + s1 * stride_s1 +
          w * stride_w + s2 * stride_s2 + c * stride_c;
      grad_input_data[i] = grad_output_data[output_offset];

      at::native::data_index_step(
          n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);
    }
  });
}

void pixel_shuffle_kernel(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t upscale_factor) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pixel_shuffle", [&] {
        cpu_pixel_shuffle<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "pixel_shuffle_channels_last", [&] {
            cpu_pixel_shuffle_channels_last<scalar_t>(
                output, input, upscale_factor);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_shuffle_backward_kernel(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t upscale_factor) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(), "pixel_shuffle_backward", [&] {
            cpu_pixel_shuffle_backward<scalar_t>(
                grad_input, grad_output, upscale_factor);
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(),
          "pixel_shuffle_backward_channels_last",
          [&] {
            cpu_pixel_shuffle_backward_channels_last<scalar_t>(
                grad_input, grad_output, upscale_factor);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_unshuffle_kernel(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t downscale_factor) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // input tensor shape of [N, C, Hr, Wr]
      // output tensor shape of [N, Crr, H, W]
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pixel_unshuffle", [&] {
        cpu_pixel_shuffle_backward<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // input tensor shape of [N, Hr, Wr, C]
      // output tensor shape of [N, H, W, Crr]
      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "pixel_unshuffle_channels_last", [&] {
            cpu_pixel_shuffle_backward_channels_last<scalar_t>(
                output, input, downscale_factor);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_unshuffle_backward_kernel(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t downscale_factor) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // grad_output tensor shape of [N, Crr, H, W]
      // grad_input tensor shape of [N, C, Hr, Wr]
      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(), "pixel_unshuffle_backward", [&] {
            cpu_pixel_shuffle<scalar_t>(
                grad_input, grad_output, downscale_factor);
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // grad_output tensor shape of [N, H, W, Crr]
      // grad_input tensor shape of [N, Hr, Wr, C]
      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(),
          "pixel_unshuffle_backward_channels_last",
          [&] {
            cpu_pixel_shuffle_channels_last<scalar_t>(
                grad_input, grad_output, downscale_factor);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

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

  pixel_shuffle_kernel(output, input, upscale_factor);
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

  pixel_shuffle_backward_kernel(grad_input, grad_output_, upscale_factor);
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

  pixel_unshuffle_kernel(output, input, downscale_factor);
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

  pixel_unshuffle_backward_kernel(grad_input, grad_output_, downscale_factor);
  return grad_input;
}

at::Tensor pixel_shuffle(const at::Tensor& self, int64_t upscale_factor) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::pixel_shuffle\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::pixel_shuffle", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      self.dim() >= 3,
      "pixel_shuffle expects input to have at least 3 dimensions, but got input with ",
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
      "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
      "upscale_factor, but input.size(-3)=",
      c,
      " is not divisible by ",
      upscale_factor_squared);

  return at::native_pixel_shuffle(self, upscale_factor);
}

at::Tensor math_pixel_shuffle(const at::Tensor& self, int64_t upscale_factor) {
  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  const auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  int64_t oc = c / upscale_factor_squared;
  int64_t oh = h * upscale_factor;
  int64_t ow = w * upscale_factor;

  // First, reshape to split the channels dim from c into 3 separate dims: (oc,
  // upscale_factor, upscale_factor). This allows shuffling to be done next by
  // permuting dims.
  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {oc, upscale_factor, upscale_factor, h, w});
  const auto input_reshaped = self.reshape(added_dims_shape);

  // Next, shuffle by permuting the new upscale_factor dims alongside the height
  // and width dims.
  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  // std::iota is used to maintain the batch dims within the permutation.
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(
      permutation.end(),
      {-5 /* oc */,
       -2 /* h */,
       -4 /* 1st upscale_factor */,
       -1 /* w */,
       -3 /* 2nd upscale_factor */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // Finally, upscale by collapsing (h, upscale_factor) -> a single dim (oh)
  // and (w, upscale_factor) -> a single dim (ow).
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  return input_permuted.reshape(final_shape);
}

at::Tensor pixel_unshuffle(const at::Tensor& self, int64_t downscale_factor) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::pixel_unshuffle\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::pixel_unshuffle", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      self.dim() >= 3,
      "pixel_unshuffle expects input to have at least 3 dimensions, but got input with ",
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
      "pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=",
      h,
      " is not divisible by ",
      downscale_factor);
  TORCH_CHECK(
      w % downscale_factor == 0,
      "pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)=",
      w,
      " is not divisible by ",
      downscale_factor);

  return at::native_pixel_unshuffle(self, downscale_factor);
}

at::Tensor math_pixel_unshuffle(
    const at::Tensor& self,
    int64_t downscale_factor) {
  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  int64_t downscale_factor_squared = downscale_factor * downscale_factor;
  int64_t oc = c * downscale_factor_squared;
  int64_t oh = h / downscale_factor;
  int64_t ow = w / downscale_factor;

  // First, reshape to split height dim into (oh, downscale_factor) dims and
  // width dim into (ow, downscale_factor) dims. This allows unshuffling to be
  // done next by permuting dims.
  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {c, oh, downscale_factor, ow, downscale_factor});
  const auto input_reshaped = self.reshape(added_dims_shape);

  // Next, unshuffle by permuting the downscale_factor dims alongside the
  // channel dim.
  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  // std::iota is used to maintain the batch dims within the permutation.
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(
      permutation.end(),
      {-5 /* c */,
       -3 /* 1st downscale_factor */,
       -1 /*2nd downscale_factor */,
       -4 /* oh */,
       -2 /* ow */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // Finally, downscale by collapsing (c, downscale_factor, downscale_factor) ->
  // a single dim (oc), resulting in height=oh and width=ow.
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  return input_permuted.reshape(final_shape);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pixel_shuffle"),
      TORCH_FN((&torch_ipex::cpu::pixel_shuffle)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pixel_unshuffle"),
      TORCH_FN((&torch_ipex::cpu::pixel_unshuffle)));
}

} // namespace cpu
} // namespace torch_ipex