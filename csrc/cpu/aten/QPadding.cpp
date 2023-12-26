#include <ATen/ATen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "QPadding.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(replication_pad2d_kernel_stub);
IPEX_DEFINE_DISPATCH(replication_pad3d_kernel_stub);
IPEX_DEFINE_DISPATCH(reflection_pad2d_kernel_stub);
IPEX_DEFINE_DISPATCH(reflection_pad3d_kernel_stub);

at::Tensor replication_pad2d(
    const at::Tensor& input,
    c10::IntArrayRef padding) {
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  // allow empty batch size but not other dimensions.
  padding::check_valid_input<2>(input);

  int ndim = input.dim();
  if (ndim == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth = iwidth + pad_l + pad_r;

  TORCH_CHECK(
      owidth >= 1 || oheight >= 1,
      "input (H: ",
      iheight,
      ", W: ",
      iwidth,
      " ) is too small."
      " Calculated output H: ",
      oheight,
      " W: ",
      owidth);

  at::Tensor output;
  if (ndim == 3) {
    output = at::_empty_affine_quantized(
        {nslices, oheight, owidth},
        input.options(),
        input.q_scale(),
        input.q_zero_point());
  } else {
    output = at::_empty_affine_quantized(
        {nbatch, nslices, oheight, owidth},
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
  }

  replication_pad2d_kernel_stub(kCPU, output, input, padding);
  return output;
}

at::Tensor replication_pad3d(
    const at::Tensor& input,
    c10::IntArrayRef padding) {
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  int64_t pleft = padding[0];
  int64_t pright = padding[1];
  int64_t ptop = padding[2];
  int64_t pbottom = padding[3];
  int64_t pfront = padding[4];
  int64_t pback = padding[5];

  // allow empty batch size but not other dimensions.
  padding::check_valid_input<3>(input);

  int ndim = input.dim();
  if (ndim == 5) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth = iwidth + pleft + pright;

  TORCH_CHECK(
      owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ",
      idepth,
      " H: ",
      iheight,
      ", W: ",
      iwidth,
      ") is too small."
      " Calculated output D: ",
      odepth,
      " H: ",
      oheight,
      " W: ",
      owidth);

  const auto memory_format = input.suggest_memory_format();
  at::Tensor output;
  if (ndim == 4) {
    output = at::_empty_affine_quantized(
        {nbatch, nslices, oheight, owidth},
        input.options().memory_format(memory_format),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
  } else {
    output = at::_empty_affine_quantized(
        {nbatch, nslices, odepth, oheight, owidth},
        input.options().memory_format(memory_format),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
  }

  replication_pad3d_kernel_stub(kCPU, output, input, padding);
  return output;
}

at::Tensor reflection_pad2d(const at::Tensor& input, at::IntArrayRef padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;

  // allow empty batch size but not other dimensions.
  padding::check_valid_input<2>(input);

  int ndim = input.dim();
  if (ndim == 4) {
    nbatch = input.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input.size(dim_slices);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size should be less than the corresponding "
      "input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      ndim);

  TORCH_CHECK(
      pad_t < input_h && pad_b < input_h,
      "Argument #6: Padding size should be less than the corresponding "
      "input dimension, but got: padding (",
      pad_t,
      ", ",
      pad_b,
      ") at dimension ",
      dim_h,
      " of input ",
      ndim);

  TORCH_CHECK(
      output_w >= 1 || output_h >= 1,
      "input (H: ",
      input_h,
      ", W: ",
      input_w,
      ")is too small. Calculated "
      "output H: ",
      output_h,
      " W: ",
      output_w);

  at::Tensor output;
  if (ndim == 3) {
    output = at::_empty_affine_quantized(
        {nplane, output_h, output_w},
        input.options(),
        input.q_scale(),
        input.q_zero_point());
  } else {
    output = at::_empty_affine_quantized(
        {nbatch, nplane, output_h, output_w},
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
  }
  reflection_pad2d_kernel_stub(kCPU, output, input, padding);
  return output;
}

at::Tensor reflection_pad3d(const at::Tensor& input, c10::IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;
  int64_t dim_plane = 0;

  // allow empty batch size but not other dimensions.
  padding::check_valid_input<3>(input);

  bool batch_mode = (input.dim() == 5);
  if (batch_mode) {
    dim_w++;
    dim_h++;
    dim_d++;
    dim_plane++;
  }

  int64_t nplane = input.size(dim_plane);
  int64_t input_d = input.size(dim_d);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_d = input_d + pad_front + pad_back;
  int64_t output_h = input_h + pad_top + pad_bottom;
  int64_t output_w = input_w + pad_left + pad_right;

  TORCH_CHECK(
      pad_left < input_w && pad_right < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_left,
      ", ",
      pad_right,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());
  TORCH_CHECK(
      pad_top < input_h && pad_bottom < input_h,
      "Argument #6: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_top,
      ", ",
      pad_bottom,
      ") at dimension ",
      dim_h,
      " of input ",
      input.sizes());
  TORCH_CHECK(
      pad_front < input_d && pad_back < input_d,
      "Argument #8: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_front,
      ", ",
      pad_back,
      ") at dimension ",
      dim_d,
      " of input ",
      input.sizes());

  TORCH_CHECK(
      output_w >= 1 || output_h >= 1 || output_d >= 1,
      "input (D: ",
      input_d,
      " H: ",
      input_h,
      ", W: ",
      input_w,
      ") is too small."
      " Calculated output D: ",
      output_d,
      " H: ",
      output_h,
      " W: ",
      output_w);

  at::Tensor output;
  const auto memory_format = input.suggest_memory_format();
  const auto options = input.options().memory_format(memory_format);
  if (batch_mode) {
    output = at::_empty_affine_quantized(
        {input.size(0), nplane, output_d, output_h, output_w},
        options,
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
  } else {
    output = at::_empty_affine_quantized(
        {nplane, output_d, output_h, output_w},
        options,
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
  }
  reflection_pad3d_kernel_stub(kCPU, output, input, padding);
  return output;
}

IPEX_TORCH_LIBRARY_IMPL(aten, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::replication_pad2d"),
      TORCH_FN((&torch_ipex::cpu::replication_pad2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::replication_pad3d"),
      TORCH_FN((&torch_ipex::cpu::replication_pad3d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::reflection_pad2d"),
      TORCH_FN((&torch_ipex::cpu::reflection_pad2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::reflection_pad3d"),
      TORCH_FN((&torch_ipex::cpu::reflection_pad3d)));
}

} // namespace cpu
} // namespace torch_ipex
