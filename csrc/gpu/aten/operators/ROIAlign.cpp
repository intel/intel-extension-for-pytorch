#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <torch/library.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include <ATen/autocast_mode.h>
#include "RandomEngine.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/TensorOptions.h"

#include <aten/operators/MemoryAccess.h>
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
inline T bilinear_interpolate(
    const T* input,
    int height,
    int width,
    T y,
    T x,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0f || y > height || x < -1.0f || x > width) {
    // empty
    return 0;
  }

  if (y <= 0.f)
    y = 0.f;
  if (x <= 0.f)
    x = 0.f;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1.f - ly, hx = 1.f - lx;

  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T, typename item_t>
inline void roi_align_forward_kernel_impl(
    item_t& item,
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const T* rois,
    T* output) {
  int lid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  for (int index = lid; index < nthreads;
       index += item.get_group_range(0) * item.get_local_range(0)) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5f : (T)0.0f;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (T)1.f);
      roi_height = std::max(roi_height, (T)1.f);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0.f)
        ? sampling_ratio
        : std::ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0.f)
        ? sampling_ratio
        : std::ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    T output_val = 0.f;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_input, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    output[index] = output_val;
  }
}

template <typename T>
inline void bilinear_interpolate_gradient(
    int height,
    int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0f || y > height || x < -1.0f || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.f;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0.f)
    y = 0.f;
  if (x <= 0.f)
    x = 0.f;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1.f - ly, hx = 1.f - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename T, typename item_t>
inline void roi_align_backward_kernel_impl(
    item_t& item,
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride) {
  int lid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  for (int index = lid; index < nthreads;
       index += item.get_group_range(0) * item.get_local_range(0)) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5f : (T)0.0f;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (T)1.f);
      roi_height = std::max(roi_height, (T)1.f);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    // We need to index the gradient using the tensor strides to access the
    // correct values.
    int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0.f)
        ? sampling_ratio
        : std::ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0.f)
        ? sampling_ratio
        : std::ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              (dpcpp_global_ptr_pt<
                  T>)(offset_grad_input + y_low * width + x_low),
              static_cast<T>(g1));
          atomicAdd(
              (dpcpp_global_ptr_pt<
                  T>)(offset_grad_input + y_low * width + x_high),
              static_cast<T>(g2));
          atomicAdd(
              (dpcpp_global_ptr_pt<
                  T>)(offset_grad_input + y_high * width + x_low),
              static_cast<T>(g3));
          atomicAdd(
              (dpcpp_global_ptr_pt<
                  T>)(offset_grad_input + y_high * width + x_high),
              static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  }
}

at::Tensor roi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(input.is_xpu(), "input must be a XPU tensor");
  TORCH_CHECK(rois.is_xpu(), "rois must be a XPU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  int items_per_group = 512;
  int num_groups =
      std::min((output_size + 512 - 1) / 512, static_cast<int64_t>(4096));

  if (output.numel() == 0)
    return output;

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "roi_align_forward_kernel",
      [&] {
        auto spatial_scale_ = static_cast<scalar_t>(spatial_scale);
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto input_ptr = (scalar_t*)input_.data_ptr();
          auto rois_ptr = (scalar_t*)rois_.data_ptr();
          auto output_ptr = (scalar_t*)output.data_ptr();
          auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
            roi_align_forward_kernel_impl<scalar_t>(
                item,
                output_size,
                input_ptr,
                spatial_scale_,
                channels,
                height,
                width,
                pooled_height,
                pooled_width,
                sampling_ratio,
                aligned,
                rois_ptr,
                output_ptr);
          };
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(num_groups * items_per_group),
                  sycl::range<1>(items_per_group)),
              kfn);
        };
        DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
      });
  return output;
}

at::Tensor roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(grad.is_xpu(), "grad must be a XPU tensor");
  TORCH_CHECK(rois.is_xpu(), "rois must be a XPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  int items_per_group = 512;
  int num_groups =
      std::min((grad.numel() + 512 - 1) / 512, static_cast<int64_t>(4096));

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto rois_ = rois.contiguous();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "roi_align_backward_kernel",
      [&] {
        auto spatial_scale_ = static_cast<scalar_t>(spatial_scale);
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto grad_ptr = (scalar_t*)grad.data_ptr();
          auto grad_input_ptr = (scalar_t*)grad_input.data_ptr();
          auto rois_ptr = (scalar_t*)rois_.data_ptr();
          auto grad_numel = grad.numel();
          auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
            roi_align_backward_kernel_impl<scalar_t>(
                item,
                grad_numel,
                grad_ptr,
                spatial_scale_,
                channels,
                height,
                width,
                pooled_height,
                pooled_width,
                sampling_ratio,
                aligned,
                grad_input_ptr,
                rois_ptr,
                n_stride,
                c_stride,
                h_stride,
                w_stride);
          };
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(num_groups * items_per_group),
                  sycl::range<1>(items_per_group)),
              kfn);
        };
        DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
      });
  return grad_input;
}

at::Tensor roi_align_forward_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  return roi_align_forward_kernel(
             at::autocast::cached_cast(at::kFloat, input, c10::DeviceType::XPU),
             at::autocast::cached_cast(at::kFloat, rois, c10::DeviceType::XPU),
             spatial_scale,
             pooled_height,
             pooled_width,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "roi_align.xpu",
      at::AtenIpexTypeXPU::roi_align_forward_kernel,
      c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "_roi_align_backward.xpu",
      at::AtenIpexTypeXPU::roi_align_backward_kernel,
      c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "roi_align.xpu",
      at::AtenIpexTypeXPU::roi_align_forward_autocast,
      c10::DispatchKey::AutocastXPU);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      c10::DispatchKey::XPU,
      TORCH_FN((&at::AtenIpexTypeXPU::roi_align_forward_kernel)));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
      c10::DispatchKey::XPU,
      TORCH_FN((&at::AtenIpexTypeXPU::roi_align_backward_kernel)));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      c10::DispatchKey::AutocastXPU,
      TORCH_FN((&at::AtenIpexTypeXPU::roi_align_forward_autocast)));
}
} // namespace
