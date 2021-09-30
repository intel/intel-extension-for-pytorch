#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

namespace torch_ipex {
namespace cpu {

template <typename scalar_t>
void cpu_pixel_shuffle(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t upscale_factor);

template <typename scalar_t>
void cpu_pixel_shuffle_channels_last(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t upscale_factor);

template <typename scalar_t>
void cpu_pixel_shuffle_backward(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t upscale_factor);

template <typename scalar_t>
void cpu_pixel_shuffle_backward_channels_last(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t upscale_factor);

void pixel_shuffle_kernel(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t upscale_factor);

void pixel_shuffle_backward_kernel(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t upscale_factor);

void pixel_unshuffle_kernel(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t downscale_factor);

void pixel_unshuffle_backward_kernel(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int64_t downscale_factor);

at::Tensor pixel_shuffle_cpu(const at::Tensor& self, int64_t upscale_factor);

at::Tensor pixel_shuffle_backward_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef input_sizes,
    int64_t upscale_factor);

at::Tensor pixel_unshuffle_cpu(
    const at::Tensor& self,
    int64_t downscale_factor);

at::Tensor pixel_unshuffle_backward_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef input_sizes,
    int64_t downscale_factor);

at::Tensor pixel_shuffle(const at::Tensor& self, int64_t upscale_factor);

at::Tensor math_pixel_shuffle(const at::Tensor& self, int64_t upscale_factor);

at::Tensor pixel_unshuffle(const at::Tensor& self, int64_t downscale_factor);

at::Tensor math_pixel_unshuffle(
    const at::Tensor& self,
    int64_t downscale_factor);

class PixelShuffleOp : public torch::autograd::Function<PixelShuffleOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(const at::Tensor& self, int64_t upscale_factor);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      int64_t upscale_factor);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

class PixelUnshuffleOp : public torch::autograd::Function<PixelUnshuffleOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(const at::Tensor& self, int64_t downscale_factor);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      int64_t downscale_factor);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

} // namespace cpu
} // namespace torch_ipex