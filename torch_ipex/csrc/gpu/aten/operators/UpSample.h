#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/Atomics.h>
#include <utils/ParamUtils.h>

namespace at {
namespace dpcpp {

template <typename scalar_t>
DPCPP_DEVICE inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
DPCPP_DEVICE inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

DPCPP_DEF_K1(bicubic2d);
DPCPP_DEF_K1(bicubic2d_bwd);

static inline void upsample_2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t nbatch,
    int64_t nchannels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  if (input.defined()) {
    TORCH_CHECK(
        input.numel() != 0 && input.dim() == 4,
        "Non-empty 4D data tensor expected but got a tensor with sizes ",
        input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 4, 0, nbatch);
    check_dim_size(grad_output, 4, 1, nchannels);
    check_dim_size(grad_output, 4, 2, output_height);
    check_dim_size(grad_output, 4, 3, output_width);
  }
}

template <typename scalar_t>
static inline scalar_t area_pixel_compute_scale(
    int input_size,
    int output_size,
    bool align_corners) {
  /* We view each pixel as an area, idx + 0.5 as its center index.
   * Here is an example formula in 1D case.
   * if align_corners: center of two corner pixel areas are preserved,
   *     (0.5, 0.5) -> (0.5, 0.5),
   *     (input_size - 0.5, 0.5) -> (output_size - 0.5)
   *     scale = (input_size - 0.5 - 0.5) / (output_size - 0.5 - 0.5)
   *     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
   * if not align_corners: the whole range is scaled accordingly
   *     scale = input_size / output_size
   *     src_idx + 0.5 = scale * (dst_index + 0.5)
   */
  if (output_size > 1) {
    return align_corners
        ? static_cast<scalar_t>(input_size - 1) / (output_size - 1)
        : static_cast<scalar_t>(input_size) / output_size;
  } else {
    return scalar_t(0);
  }
}

template <typename scalar_t>
static inline scalar_t area_pixel_compute_source_index(
    scalar_t scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
    // [Note] Follow Opencv resize logic:
    // We allow negative src_idx here and later will use
    //   dx = src_idx - floorf(src_idx)
    // to compute the "distance"(which affects weights).
    // For linear modes, weight distribution doesn't matter
    // for negative indices as they use 2 pixels to interpolate.
    // For example, [-1, 0], they both use pixel 0 value so it
    // doesn't affect if we bound the src_idx to 0 or not.
    // TODO: Our current linear mode impls use unbound indices
    // where we should and then remove this cubic flag.
    // This matters in cubic mode, as we might need [-1, 0, 1, 2]
    // to interpolate and the weights can be affected.
    return (!cubic && src_idx < 0) ? scalar_t(0) : src_idx;
  }
}

template <typename scalar_t>
static scalar_t upsample_get_value_bounded(
    const dpcpp_global_ptr_pt<scalar_t>& data,
    int batch,
    int channel,
    int width,
    int height,
    int x,
    int y) {
  int access_x = max(min(x, width - 1), static_cast<int>(0));
  int access_y = max(min(y, height - 1), static_cast<int>(0));
  return data[batch * height * width * channel + channel * height * width +
              access_y * width + access_x];
}

template <typename scalar_t>
static void upsample_increment_value_bounded(
    const dpcpp_global_ptr_pt<scalar_t>& data,
    int batch,
    int channel,
    int width,
    int height,
    int x,
    int y,
    scalar_t value) {
  int64_t access_x = max(min(x, width - 1), static_cast<int>(0));
  int64_t access_y = max(min(y, height - 1), static_cast<int>(0));
  atomicAdd(
      &data[batch * height * width * channel + channel * height * width +
            access_y * width + access_x],
      value);
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename scalar_t>
static inline scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
static inline scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename scalar_t>
static inline void get_cubic_upsample_coefficients(
    scalar_t coeffs[4],
    scalar_t t) {
  scalar_t A = -0.75;

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

  // opposite coefficients
  scalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}

template <typename scalar_t>
static inline scalar_t cubic_interp1d(
    scalar_t x0,
    scalar_t x1,
    scalar_t x2,
    scalar_t x3,
    scalar_t t) {
  scalar_t coeffs[4];
  get_cubic_upsample_coefficients<scalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

static inline void set_params(
    IntArrayRef input_size,
    IntArrayRef output_size,
    dnnl::memory::dims& src_dims,
    dnnl::memory::dims& dst_dims,
    std::vector<float>& factors,
    int64_t ndims,
    const double& scales_w = 0.0,
    const double& scales_h = 0.0,
    const double& scales_d = 0.0) {
  int64_t n, c, id, ih, iw, od, oh, ow;

  n = input_size[0];
  c = input_size[1];
  id = ih = iw = od = oh = ow = 1;
  if (ndims == 5) {
    od = output_size[0];
    oh = output_size[1];
    ow = output_size[2];

    id = input_size[2];
    ih = input_size[3];
    iw = input_size[4];
  }
  if (ndims == 4) {
    oh = output_size[0];
    ow = output_size[1];

    ih = input_size[2];
    iw = input_size[3];
  }
  if (ndims == 3) {
    ow = output_size[0];
    iw = input_size[2];
  }

  const float depth_scale = scales_d != 0.0
      ? scales_d
      : (std::round((float)od / (float)id * 100) / 100);
  const float height_scale = scales_h != 0.0
      ? scales_h
      : (std::round((float)oh / (float)ih * 100) / 100);
  const float width_scale = scales_w != 0.0
      ? scales_w
      : (std::round((float)ow / (float)iw * 100) / 100);

  src_dims = {n, c};
  dst_dims = {n, c};
  if (ndims == 5) {
    factors.push_back(depth_scale);
    src_dims.push_back(id);
    dst_dims.push_back(od);
  }
  if (ndims >= 4) {
    factors.push_back(height_scale);
    src_dims.push_back(ih);
    dst_dims.push_back(oh);
  }
  if (ndims >= 3) {
    factors.push_back(width_scale);
    src_dims.push_back(iw);
    dst_dims.push_back(ow);
  }
}

} // namespace dpcpp
} // namespace at
