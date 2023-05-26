#include "QCircularPad.h"

#include <ATen/ATen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

at::Tensor qpad_circular(const at::Tensor& self, c10::IntArrayRef padding) {
  RECORD_FUNCTION("qpad_circular", c10::ArrayRef<c10::IValue>({}));
  const auto in_shape = self.sizes();
  TORCH_CHECK(
      in_shape.size() >= 3 && in_shape.size() <= 5,
      "qpad_circular(): Expected 3D, 4D or 5D tensor for input, but got tensor with ",
      in_shape.size(),
      " dimensions instead");
  const auto ndim = static_cast<int64_t>(in_shape.size()) - 2;
  TORCH_CHECK(
      padding.size() + 4 == in_shape.size() * 2,
      "Invalid padding size, expected ",
      ndim * 2,
      " but got ",
      padding.size());

  c10::DimVector out_shape(in_shape.size());
  out_shape[0] = in_shape[0];
  out_shape[1] = in_shape[1];

  // Get shape of padded tensor
  for (const auto i : c10::irange(ndim)) {
    const auto& pad_l = padding[2 * (ndim - i - 1) + 0];
    const auto& pad_r = padding[2 * (ndim - i - 1) + 1];
    const auto& size = in_shape[2 + i];
    out_shape[2 + i] = size + pad_l + pad_r;

    TORCH_CHECK(
        pad_l <= size && pad_r <= size,
        "Padding value causes wrapping around more than once.");
    TORCH_CHECK(
        out_shape[2 + i] >= 0,
        "Negative padding value is resulting in an empty dimension");
  }

  at::Tensor out;

  if (in_shape.size() == 3) {
    out = at::_empty_affine_quantized(
        out_shape, self.options(), self.q_scale(), self.q_zero_point());
  } else {
    out = at::_empty_affine_quantized(
        out_shape,
        self.options().memory_format(self.suggest_memory_format()),
        self.q_scale(),
        self.q_zero_point(),
        c10::nullopt);
  }

  // Put original array into the padded array
  at::Tensor out_slice = out;
  at::Tensor in_slice = self;
  const int64_t zero = 0;
  for (const auto i : c10::irange(ndim)) {
    const auto dim = ndim - i + 1;
    const auto& pad_l = padding[2 * i + 0];
    const auto& pad_r = padding[2 * i + 1];
    out_slice = out_slice.slice(
        dim, std::max(pad_l, zero), out_shape[dim] - std::max(pad_r, zero));
    in_slice = in_slice.slice(
        dim, std::max(-pad_l, zero), in_shape[dim] - std::max(-pad_r, zero));
  }
  out_slice.copy_(in_slice);

  // The following steps first pad the beginning of the tensor (left side),
  // and then pad the end of the tensor (right side).
  // Note: Corners will be written more than once when ndim > 1.
  //
  // Only in cases where padding values are > 0 are when additional copying
  // is required.
  for (const auto i : c10::irange(ndim)) {
    const auto dim = ndim - i + 1;
    const auto& pad_l = padding[2 * i + 0];
    const auto& pad_r = padding[2 * i + 1];

    if (pad_l > 0) {
      out_slice = out.slice(dim, 0, pad_l);
      in_slice = out.slice(
          dim,
          out_shape[dim] - pad_l - std::max(pad_r, zero),
          out_shape[dim] - std::max(pad_r, zero));
      out_slice.copy_(in_slice);
    }

    if (pad_r > 0) {
      out_slice = out.slice(dim, out_shape[dim] - pad_r, out_shape[dim]);
      in_slice =
          out.slice(dim, std::max(pad_l, zero), std::max(pad_l, zero) + pad_r);
      out_slice.copy_(in_slice);
    }
  }

  return out;
}

} // namespace cpu
} // namespace torch_ipex
