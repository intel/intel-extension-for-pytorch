#pragma once

#include <dyndisp/DispatchStub.h>
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

namespace {

void replication_pad2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding);
void replication_pad3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding);
void reflection_pad2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding);
void reflection_pad3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding);

} // namespace

using padding_fn =
    void (*)(const at::Tensor&, const at::Tensor&, c10::IntArrayRef);
IPEX_DECLARE_DISPATCH(padding_fn, replication_pad2d_kernel_stub);
IPEX_DECLARE_DISPATCH(padding_fn, replication_pad3d_kernel_stub);
IPEX_DECLARE_DISPATCH(padding_fn, reflection_pad2d_kernel_stub);
IPEX_DECLARE_DISPATCH(padding_fn, reflection_pad3d_kernel_stub);

namespace padding {

template <int dim>
static inline void check_valid_input(const at::Tensor& input) {
  int input_dim = input.dim();

  bool is_batch_mode = input_dim == (dim + 2);

  bool valid_batch_mode = is_batch_mode;
  bool valid_non_batch_mode = !is_batch_mode;

  if (is_batch_mode) {
    // allow batch size of 0-dim.
    for (const auto d : c10::irange(1, input_dim)) {
      valid_batch_mode = valid_batch_mode && input.size(d) != 0;
    }
  } else {
    for (const auto d : c10::irange(0, input_dim)) {
      valid_non_batch_mode = valid_non_batch_mode && input.size(d) != 0;
    }
  }

  TORCH_CHECK(
      valid_batch_mode || valid_non_batch_mode,
      "Expected ",
      dim + 1,
      "D or ",
      dim + 2,
      "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());
}

} // namespace padding

} // namespace cpu
} // namespace torch_ipex
