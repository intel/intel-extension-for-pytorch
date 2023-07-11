#include <ATen/native/UpSample.h>
#include <core/MemoryFormat.h>
#include <oneDNN/oneDNN.h>
#include <tensor/Tensor.h>
#include "UpSample.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace at::native;
using namespace at::AtenIpexTypeXPU;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;
using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

struct Nearest_index_op {
  int operator()(const float scale, int dst_index, int input_size) const {
    // index_f32 = (output_index) * scale
    // input_index = round(index_f32)
    // Same as a buggy OpenCV INTER_NEAREST
    // We keep this method for BC and consider as deprecated.
    // See nearest_exact as replacement
    const int src_index =
        min(static_cast<int>(floorf((dst_index)*scale)), input_size - 1);
    return src_index;
  }
};

struct Nearest_bw_index_op {
  int operator()(const float scale, int dst_index, int output_size) const {
    // Equivalent to buggy OpenCV INTER_NEAREST
    // We keep this method for BC and consider as deprecated.
    // See nearest_exact as replacement
    const int src_index =
        min(static_cast<int>(ceilf(dst_index * scale)), output_size);
    return src_index;
  }
};

static int lastPow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max<int>(1, n - (n >> 1));
}

inline size_t idx_cl(
    const size_t n,
    const size_t h,
    const size_t w,
    const size_t c,
    const size_t height,
    const size_t width,
    const size_t channel) {
  return ((n * height + h) * width + w) * channel + c;
}

template <typename scalar_t, typename index_op_t>
void upsample_nearest1d_out_kernel(
    int n,
    const scalar_t* input,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* output,
    float scale_factor,
    index_op_t index_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int dst_idx = item.get_global_linear_id();
      if (dst_idx >= dim_c * dst_dim_w)
        return;

      int c = (dst_idx / dst_dim_w) % dim_c;

      int dst_x = dst_idx % dst_dim_w;
      int src_x = index_op(scale_factor, dst_x, src_dim_w);

      int src_idx = c * src_dim_w + src_x;
      int src_stride = dim_c * src_dim_w;
      int dst_stride = dim_c * dst_dim_w;

      for (int b = 0; b < dim_b; b++) {
        output[dst_idx] = input[src_idx];
        src_idx += src_stride;
        dst_idx += dst_stride;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest1d_backward_out_kernel(
    int n,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float scale_factor,
    index_bw_op_t index_bw_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int dst_idx = item.get_global_linear_id();
      if (dst_idx >= dim_c * dst_dim_w)
        return;

      int c = (dst_idx / (dst_dim_w)) % dim_c;

      int dst_x = dst_idx % dst_dim_w;
      // note that we do not want to clamp src_x to src_dim_w, since we might
      // intentionally want to skip in case of scale_factor < 1.0
      int src_x = index_bw_op(scale_factor, dst_x, src_dim_w);
      int src_x_up = index_bw_op(scale_factor, dst_x + 1, src_dim_w);

      for (int b = 0; b < dim_b; b++) {
        accscalar_t grad = 0;
        int src_idx = b * dim_c * src_dim_w + c * src_dim_w + src_x;
        for (int x = src_x; x < src_x_up; x++) {
          grad += grad_o[src_idx++];
        }
        grad_i[dst_idx] = grad;
        dst_idx += dim_c * dst_dim_w;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_op_t>
void upsample_nearest2d_out_kernel(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t nc,
    const size_t height1, // input height
    const size_t width1,
    const size_t height2, // output height
    const size_t width2,
    float height_scale,
    float width_scale,
    index_op_t index_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int local_x = std::min<int>(lastPow2(width2), work_group_size);
  int local_y = std::min<int>(lastPow2(height2), work_group_size / local_x);
  int local_z = std::min<int>(nc, work_group_size / local_x / local_y);

  int global_x = (width2 + local_x - 1) / local_x * local_x;
  int global_y = (height2 + local_y - 1) / local_y * local_y;
  int z_plane = local_z * 4;
  int global_z = (nc + z_plane - 1) / z_plane * z_plane;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      size_t nc_idx = item.get_global_id(0);
      int h2 = item.get_global_id(1);
      int w2 = item.get_global_id(2);

      if (w2 >= width2 || h2 >= height2) {
        return;
      }

      int nc_range = item.get_global_range(0);

      const size_t h1 =
          height1 == height2 ? h2 : index_op(height_scale, h2, height1);
      const size_t w1 =
          width1 == width2 ? w2 : index_op(width_scale, w2, width1);

      size_t src_index = (nc_idx * height1 + h1) * width1 + w1;
      size_t src_index_stride = nc_range * width1 * height1;
      size_t dst_index = (nc_idx * height2 + h2) * width2 + w2;
      size_t dst_index_stride = nc_range * width2 * height2;

      // iterating over
      while (nc_idx < nc) {
        odata[dst_index] = idata[src_index];
        dst_index += dst_index_stride;
        src_index += src_index_stride;
        nc_idx += nc_range;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_z, global_y, global_x),
            sycl::range<3>(local_z, local_y, local_x)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_op_t>
void upsample_nearest2d_channels_last_kernel(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t channels,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale,
    const size_t out_numel,
    index_op_t index_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (out_numel + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      const int index = item.get_global_linear_id();

      if (index < out_numel) {
        const int c = index % channels;
        const int w2 = (index / channels) % width2;
        const int h2 = (index / channels / width2) % height2;
        const int n = index / channels / width2 / height2;

        const size_t h1 =
            height1 == height2 ? h2 : index_op(height_scale, h2, height1);
        const size_t w1 =
            width1 == width2 ? w2 : index_op(width_scale, w2, width1);

        odata[index] = idata[idx_cl(n, h1, w1, c, height1, width1, channels)];
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest2d_backward_kernel(
    size_t n,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float height_scale,
    float width_scale,
    index_bw_op_t index_bw_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int dst_idx = item.get_global_linear_id();
      if (dst_idx >= dim_c * dst_dim_h * dst_dim_w)
        return;

      int dst_c_stride = dst_dim_h * dst_dim_w;
      int src_c_stride = src_dim_h * src_dim_w;

      int c = (dst_idx / (dst_c_stride)) % dim_c;

      int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
      // note that we do not want to clamp src_y to src_dim_y, since we might
      // intentionally want to skip in case of scale_factor < 1.0
      int src_y = index_bw_op(height_scale, dst_y, src_dim_h);
      int src_y_up = index_bw_op(height_scale, dst_y + 1, src_dim_h);

      int dst_x = dst_idx % dst_dim_w;
      // note that we do not want to clamp src_x to src_dim_w, since we might
      // intentionally want to skip in case of scale_factor < 1.0
      int src_x = index_bw_op(width_scale, dst_x, src_dim_w);
      int src_x_up = index_bw_op(width_scale, dst_x + 1, src_dim_w);

      for (int b = 0; b < dim_b; b++) {
        accscalar_t grad = 0;
        for (int y = src_y; y < src_y_up; y++) {
          for (int x = src_x; x < src_x_up; x++) {
            int src_idx =
                b * dim_c * src_c_stride + c * src_c_stride + y * src_dim_w + x;
            grad += grad_o[src_idx];
          }
        }
        grad_i[dst_idx] = grad;
        dst_idx += dim_c * dst_c_stride;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest2d_backward_channels_last_kernel(
    const scalar_t* go,
    scalar_t* gi,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    const size_t channels,
    const float height_scale,
    const float width_scale,
    const size_t gi_numel,
    index_bw_op_t index_bw_op) {
  // 1 is for grad_output (src)
  // 2 is for grad_input (dst)
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (gi_numel + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int index = item.get_global_linear_id();

      if (index < gi_numel) {
        const int c = index % channels;
        const int w2 = (index / channels) % width2;
        const int h2 = (index / channels / width2) % height2;
        const int n = index / channels / width2 / height2;

        int h1 = index_bw_op(height_scale, h2, height1);
        int h1_up = index_bw_op(height_scale, h2 + 1, height1);

        int w1 = index_bw_op(width_scale, w2, width1);
        int w1_up = index_bw_op(width_scale, w2 + 1, width1);

        accscalar_t grad = 0;
        for (int ih = h1; ih < h1_up; ih++) {
          for (int iw = w1; iw < w1_up; iw++) {
            grad += go[idx_cl(n, ih, iw, c, height1, width1, channels)];
          }
        }
        gi[index] = static_cast<scalar_t>(grad);
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_op_t>
void upsample_nearest3d_out_kernel(
    int n,
    const scalar_t* input,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_d,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_d,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* output,
    float depth_scale,
    float height_scale,
    float width_scale,
    index_op_t index_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int dst_idx = item.get_global_linear_id();

      if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w)
        return;

      int dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
      int src_c_stride = src_dim_d * src_dim_h * src_dim_w;

      int c = (dst_idx / (dst_c_stride)) % dim_c;

      int dst_z = (dst_idx / dst_dim_h / dst_dim_w) % dst_dim_d;
      int src_z = index_op(depth_scale, dst_z, src_dim_d);
      int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
      int src_y = index_op(height_scale, dst_y, src_dim_h);

      int dst_x = dst_idx % dst_dim_w;
      int src_x = index_op(width_scale, dst_x, src_dim_w);

      int src_idx = c * src_c_stride + src_z * src_dim_h * src_dim_w +
          src_y * src_dim_w + src_x;
      for (int b = 0; b < dim_b; b++) {
        output[dst_idx] = input[src_idx];
        src_idx += dim_c * src_c_stride;
        dst_idx += dim_c * dst_c_stride;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest3d_backward_out_kernel(
    int n,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_d,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_d,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float depth_scale,
    float height_scale,
    float width_scale,
    index_bw_op_t index_bw_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int dst_idx = item.get_global_linear_id();

      if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w)
        return;

      int dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
      int src_c_stride = src_dim_d * src_dim_h * src_dim_w;

      int c = (dst_idx / (dst_c_stride)) % dim_c;

      int dst_z = (dst_idx / dst_dim_h / dst_dim_w) % dst_dim_d;
      // note that we do not want to clamp src_z to src_dim_z, since we might
      // intentionally want to skip in case of scale_factor < 1.0
      int src_z = index_bw_op(depth_scale, dst_z, src_dim_d);
      int src_z_up = index_bw_op(depth_scale, dst_z + 1, src_dim_d);

      int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
      // note that we do not want to clamp src_y to src_dim_y, since we might
      // intentionally want to skip in case of scale_factor < 1.0
      int src_y = index_bw_op(height_scale, dst_y, src_dim_h);
      int src_y_up = index_bw_op(height_scale, dst_y + 1, src_dim_h);

      int dst_x = dst_idx % dst_dim_w;
      // note that we do not want to clamp src_x to src_dim_w, since we might
      // intentionally want to skip in case of scale_factor < 1.0
      int src_x = index_bw_op(width_scale, dst_x, src_dim_w);
      int src_x_up = index_bw_op(width_scale, dst_x + 1, src_dim_w);

      for (int b = 0; b < dim_b; b++) {
        accscalar_t grad = 0;
        for (int z = src_z; z < src_z_up; z++) {
          for (int y = src_y; y < src_y_up; y++) {
            for (int x = src_x; x < src_x_up; x++) {
              int src_idx = b * dim_c * src_c_stride + c * src_c_stride +
                  z * src_dim_h * src_dim_w + y * src_dim_w + x;
              grad += grad_o[src_idx];
            }
          }
        }
        grad_i[dst_idx] = grad;
        dst_idx += dim_c * dst_c_stride;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename index_op_t>
static void upsample_nearest1d_out_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales,
    index_op_t index_op) {
  TORCH_CHECK(
      input_.device() == output.device(),
      "expected device '",
      input_.device(),
      "' but got '",
      output.device(),
      "' for nearest output");

  int output_width = output_size[0];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_width = input_.size(2);

  Tensor input = input_.contiguous();

  if (input.numel() == 0) {
    return;
  }

  // upsample_nearest1d meta call makes sure `nbatch != 0`
  unsigned int n = output.numel() / nbatch;
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());
  Tensor output_c = output.contiguous();

  IPEX_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Byte,
      input.scalar_type(),
      "upsample_nearest1d_out_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output_c.data_ptr<scalar_t>();

        const float scale_factor = nearest_compute_scales_value<float>(
            scales, input_width, output_width);

        upsample_nearest1d_out_kernel(
            n,
            idata,
            nbatch,
            channels,
            input_width,
            output_width,
            odata,
            scale_factor,
            index_op);
      });
  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
}

template <typename index_op_t>
static void upsample_nearest1d_backward_out_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    index_op_t index_op) {
  TORCH_CHECK(
      grad_input.device() == grad_output_.device(),
      "expected device '",
      grad_input.device(),
      "' but got '",
      grad_output_.device(),
      "' for nearest backward output");

  int output_width = output_size[0];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_width = input_size[2];

  Tensor grad_output = grad_output_.contiguous();

  if (grad_input.numel() == 0) {
    return;
  }

  // upsample_nearest1d meta call makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

  Tensor grad_input_c = grad_input.contiguous();

  IPEX_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Byte,
      grad_output.scalar_type(),
      "upsample_nearest1d_backward_out_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t>;

        auto idata = grad_input_c.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        const float scale_factor = compute_scales_value_backwards<float>(
            scales, output_width, input_width);

        upsample_nearest1d_backward_out_kernel<scalar_t, accscalar_t>(
            n,
            odata,
            nbatch,
            channels,
            output_width,
            input_width,
            idata,
            scale_factor,
            index_op);
      });
  if (!grad_input.is_contiguous()) {
    grad_input.copy_(grad_input_c);
  }
}

template <typename index_op_t>
static void upsample_nearest2d_out_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    index_op_t index_op) {
  TORCH_CHECK(
      input_.device() == output.device(),
      "expected device '",
      input_.device(),
      "' but got '",
      output.device(),
      "' for nearest output");

  if (input_.numel() == 0) {
    return;
  }
  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  // TODO: upsample_nearest2d meta call makes sure input/output tensor is
  // not empty;
  //
  const float height_scale = nearest_compute_scales_value<float>(
      scales_h, input_height, output_height);
  const float width_scale =
      nearest_compute_scales_value<float>(scales_w, input_width, output_width);

  const auto memory_format = input_.suggest_memory_format();

  if (input_.sizes() == output.sizes()) {
    output.copy_(input_);
    return;
  }

  // heuristic: only use channels_last path when it's faster than the
  // contiguous path
  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 &&
      output.is_contiguous(memory_format)) {
    at::Tensor input = input_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(
        input.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports input tensors with less than INT_MAX elements");
    TORCH_CHECK(
        output.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports output tensors with less than INT_MAX elements");

    IPEX_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        input.scalar_type(),
        "upsample_nearest2d_channes_last_out_kernel",
        [&] {
          const scalar_t* idata = input.data_ptr<scalar_t>();
          scalar_t* odata = output.data_ptr<scalar_t>();
          upsample_nearest2d_channels_last_kernel<scalar_t>(
              idata,
              odata,
              channels,
              input_height,
              input_width,
              output_height,
              output_width,
              height_scale,
              width_scale,
              output.numel(),
              index_op);
        });
  } else {
    // This is needed for non-contiguous tensors.
    Tensor output_c = output.is_contiguous()
        ? output
        : at::empty(output.sizes(), output.options());
    Tensor input = input_.contiguous();

    int nc = nbatch * channels;

    IPEX_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        input.scalar_type(),
        "upsample_nearest2d_out_kernel",
        [&] {
          auto idata = input.data_ptr<scalar_t>();
          auto odata = output_c.data_ptr<scalar_t>();

          upsample_nearest2d_out_kernel<scalar_t>(
              idata,
              odata,
              nc,
              input_height,
              input_width,
              output_height,
              output_width,
              height_scale,
              width_scale,
              index_op);
        });

    if (!output.is_contiguous()) {
      output.copy_(output_c);
    }
  }
}

template <typename index_bw_op_t>
static void upsample_nearest2d_backward_out_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    index_bw_op_t index_bw_op) {
  TORCH_CHECK(
      grad_input.device() == grad_output_.device(),
      "expected device '",
      grad_input.device(),
      "' but got '",
      grad_output_.device(),
      "' for nearest backward output");

  if (grad_input.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  const float height_scale = compute_scales_value_backwards<float>(
      scales_h, output_height, input_height);
  const float width_scale = compute_scales_value_backwards<float>(
      scales_w, output_width, input_width);

  auto memory_format = grad_output_.suggest_memory_format();

  if (grad_output_.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 &&
      grad_input.is_contiguous(memory_format)) {
    Tensor grad_output =
        grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(
        grad_input.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports grad_input tensors with less than INT_MAX elements");
    TORCH_CHECK(
        grad_output.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports grad_output tensors with less than INT_MAX elements");

    IPEX_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        grad_output.scalar_type(),
        "upsample_nearest2d_backward_channels_last_kernel",
        [&] {
          using accscalar_t = acc_type<scalar_t>;

          const scalar_t* go = grad_output.data_ptr<scalar_t>();
          scalar_t* gi = grad_input.data_ptr<scalar_t>();

          upsample_nearest2d_backward_channels_last_kernel<
              scalar_t,
              accscalar_t>(
              go,
              gi,
              output_height,
              output_width,
              input_height,
              input_width,
              channels,
              height_scale,
              width_scale,
              grad_input.numel(),
              index_bw_op);
        });
  } else {
    // This is needed for non-contiguous tensors.
    Tensor grad_input_c = grad_input.is_contiguous()
        ? grad_input
        : at::empty(grad_input.sizes(), grad_input.options());
    Tensor grad_output = grad_output_.contiguous();
    unsigned int n = grad_input.numel() / nbatch;

    // upsample_nearest2d meta call makes sure `nbatch != 0`
    IPEX_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        grad_output.scalar_type(),
        "upsample_nearest2d_backward_kernel",
        [&] {
          using accscalar_t = acc_type<scalar_t>;

          auto idata = grad_input_c.data_ptr<scalar_t>();
          auto odata = grad_output.data_ptr<scalar_t>();

          upsample_nearest2d_backward_kernel<scalar_t, accscalar_t>(
              n,
              odata,
              nbatch,
              channels,
              output_height,
              output_width,
              input_height,
              input_width,
              idata,
              height_scale,
              width_scale,
              index_bw_op);
        });

    if (!grad_input.is_contiguous()) {
      grad_input.copy_(grad_input_c);
    }
  }
}

template <typename index_op_t>
static void upsample_nearest3d_out_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    index_op_t index_op) {
  TORCH_CHECK(
      input_.device() == output.device(),
      "expected device '",
      input_.device(),
      "' but got '",
      output.device(),
      "' for nearest output");
  // TODO: remove this when the kernel is updated to support the
  // channels_last memory format. This is a temporary hack to prevent a silence
  // correctness issue when calling this kernel with tensors in channels_last
  // format.
  auto output_c = output.is_contiguous()
      ? output
      : at::empty(output.sizes(), output.options());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_depth = input_.size(2);
  int input_height = input_.size(3);
  int input_width = input_.size(4);

  Tensor input = input_.contiguous();

  if (input.numel() == 0) {
    return;
  }

  // upsample_nearest3d meta call makes sure `nbatch != 0`
  unsigned int n = output.numel() / nbatch;
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());

  IPEX_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Byte,
      input.scalar_type(),
      "upsample_nearest3d_out_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output_c.data_ptr<scalar_t>();

        const float depth_scale = nearest_compute_scales_value<float>(
            scales_d, input_depth, output_depth);
        const float height_scale = nearest_compute_scales_value<float>(
            scales_h, input_height, output_height);
        const float width_scale = nearest_compute_scales_value<float>(
            scales_w, input_width, output_width);

        upsample_nearest3d_out_kernel<scalar_t>(
            n,
            idata,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            odata,
            depth_scale,
            height_scale,
            width_scale,
            index_op);
      });

  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
}

template <typename index_bw_op_t>
static void upsample_nearest3d_backward_out_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    index_bw_op_t index_bw_op) {
  TORCH_CHECK(
      grad_input.device() == grad_output_.device(),
      "expected device '",
      grad_input.device(),
      "' but got '",
      grad_output_.device(),
      "' for nearest backward output");

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  Tensor grad_output = grad_output_.contiguous();

  if (grad_input.numel() == 0) {
    return;
  }

  // upsample_nearest3d meta call makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

  IPEX_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Byte,
      grad_output.scalar_type(),
      "upsample_nearest3d_backward_out_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t>;

        auto idata = grad_input.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        float depth_scale = compute_scales_value_backwards<float>(
            scales_d, output_depth, input_depth);
        float height_scale = compute_scales_value_backwards<float>(
            scales_h, output_height, input_height);
        float width_scale = compute_scales_value_backwards<float>(
            scales_w, output_width, input_width);

        upsample_nearest3d_backward_out_kernel<scalar_t, accscalar_t>(
            n,
            odata,
            nbatch,
            channels,
            output_depth,
            output_height,
            output_width,
            input_depth,
            input_height,
            input_width,
            idata,
            depth_scale,
            height_scale,
            width_scale,
            index_bw_op);
      });
}

Tensor& _upsample_nearest_exact3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return output;
}

Tensor& upsample_nearest3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  TORCH_CHECK(
      input.device() == output.device(),
      "expected device '",
      input.device(),
      "' but got '",
      output.device(),
      "' for nearest output");

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int input_depth = input.size(2);
  int input_height = input.size(3);
  int input_width = input.size(4);
  bool onednn_path = (output_depth % input_depth == 0) &&
      (output_height % input_height == 0) && (output_width % input_width == 0);

  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_nearest,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
        scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    upsample_nearest3d_out_template(
        output,
        input,
        output_size,
        scales_d,
        scales_h,
        scales_w,
        Nearest_index_op());
  }
  return output;
}

Tensor upsample_nearest3d(
    const Tensor& input,
    c10::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  at::AtenIpexTypeXPU::to_plain_if_needed_(input);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  auto output = at::empty(
      {input.size(0), input.size(1), osize[0], osize[1], osize[2]},
      input.options());

  at::AtenIpexTypeXPU::upsample_nearest3d_out(
      input, osize, scale_d, scale_h, scale_w, output);
  return output;
}

Tensor _upsample_nearest_exact3d(
    const Tensor& input,
    c10::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty({0}, input.options());
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0f,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0f,
      scale_d.has_value() ? static_cast<double>(scale_d.value()) : 0.0f);
  return output;
}

Tensor& _upsample_nearest_exact3d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return grad_input;
}

Tensor& upsample_nearest3d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];
  bool onednn_path =
      ((output_depth % input_depth == 0) &&
       (output_height % input_height == 0) &&
       (output_width % input_width == 0));

  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    xpu::oneDNN::resample_backward(
        grad_input,
        grad_output,
        input_size,
        output_size,
        algorithm::resampling_nearest,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
        scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(grad_output);
    upsample_nearest3d_backward_out_template(
        grad_input,
        grad_output,
        output_size,
        input_size,
        scales_d,
        scales_h,
        scales_w,
        Nearest_bw_index_op());
  }
  return grad_input;
}

Tensor& _upsample_nearest_exact2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  return output;
}

Tensor& upsample_nearest2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);
  bool onednn_path =
      (output_height % input_height == 0) && (output_width % input_width == 0);

  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_nearest,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    upsample_nearest2d_out_template(
        output, input, output_size, scales_h, scales_w, Nearest_index_op());
  }
  return output;
}

Tensor& _upsample_nearest_exact2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  return grad_input;
}

Tensor& upsample_nearest2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  auto compute_eng = Settings::I().get_compute_eng();
  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input_size[2];
  int input_width = input_size[3];
  bool onednn_path =
      (output_height % input_height == 0) && (output_width % input_width == 0);

  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    xpu::oneDNN::resample_backward(
        grad_input,
        grad_output,
        input_size,
        output_size,
        algorithm::resampling_nearest,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(grad_output);
    upsample_nearest2d_backward_out_template(
        grad_input,
        grad_output,
        output_size,
        input_size,
        scales_h,
        scales_w,
        Nearest_bw_index_op());
  }
  return grad_input;
}

Tensor upsample_nearest1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  auto input_size = input.sizes();
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  bool onednn_path = (output_width % input_width == 0);
  if (onednn_path) {
    auto output = at::empty(
        {nbatch, channels, output_width},
        input.options(),
        suggest_memory_format_dpcpp(input));

    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_nearest,
        scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
    return output;
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    auto output = at::empty(
        {nbatch, channels, output_width},
        input.options(),
        suggest_memory_format_dpcpp(input));
    if (suggest_memory_format_dpcpp(input) == CHANNELSLAST1D_DPCPP) {
      auto tmp = output.contiguous(at::MemoryFormat::Contiguous);
      output = convert_tensor_to_channels_last_1d(tmp);
    }
    upsample_nearest1d_out_template(
        output, input, output_size, scales, Nearest_index_op());
    return output;
  }
}

Tensor& upsample_nearest1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    Tensor& output) {
  int64_t output_width = output_size[0];
  int64_t input_width = input.size(2);

  bool onednn_path = (output_width % input_width == 0);
  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_nearest,
        scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    upsample_nearest1d_out_template(
        output, input, output_size, scales, Nearest_index_op());
  }
  return output;
}

Tensor& _upsample_nearest_exact1d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    Tensor& grad_input) {
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
  return grad_input;
}

Tensor& upsample_nearest1d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    Tensor& grad_input) {
  int64_t output_width = output_size[0];
  int64_t input_width = input_size[2];

  bool onednn_path = (output_width % input_width == 0);
  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    xpu::oneDNN::resample_backward(
        grad_input,
        grad_output,
        input_size,
        output_size,
        algorithm::resampling_nearest,
        scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
  } else {
    at::AtenIpexTypeXPU::to_plain_if_needed_(grad_output);
    upsample_nearest1d_backward_out_template(
        grad_input,
        grad_output,
        output_size,
        input_size,
        scales,
        Nearest_bw_index_op());
  }
  return grad_input;
}

Tensor& _upsample_nearest_exact1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    Tensor& output) {
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
  return output;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
template <typename index_op_t>
static Tensor q_upsample_nearest2d_template(
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    index_op_t index_op) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_.dim() == 4,
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input_.sizes());

  if (input_.numel() == 0) {
    return input_;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  // TODO: upsample_nearest2d meta call makes sure input/output tensor is
  // not empty;
  //
  const float height_scale = nearest_compute_scales_value<float>(
      scales_h, input_height, output_height);
  const float width_scale =
      nearest_compute_scales_value<float>(scales_w, input_width, output_width);

  const auto memory_format = input_.suggest_memory_format();

  // heuristic: only use channels_last path when it's faster than the
  // contiguous path
  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4) {
    at::Tensor input = input_.contiguous(at::MemoryFormat::ChannelsLast);
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options()
            .memory_format(input.suggest_memory_format())
            .dtype(toQIntType(input.scalar_type())),
        input.q_scale(),
        input.q_zero_point());

    if (input_height == output_height && input_width == output_width) {
      output.copy_(input);
      return output;
    }

    TORCH_CHECK(
        input.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports input tensors with less than INT_MAX elements");
    TORCH_CHECK(
        output.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports output tensors with less than INT_MAX elements");

    IPEX_DISPATCH_QINT_TYPES(
        input.scalar_type(), "q_upsample_nearest2d_channels_last", [&]() {
          const auto idata =
              reinterpret_cast<underlying_t*>(input.data_ptr<scalar_t>());
          const auto odata =
              reinterpret_cast<underlying_t*>(output.data_ptr<scalar_t>());
          at::AtenIpexTypeXPU::upsample_nearest2d_channels_last_kernel<
              underlying_t>(
              idata,
              odata,
              channels,
              input_height,
              input_width,
              output_height,
              output_width,
              height_scale,
              width_scale,
              output.numel(),
              index_op);
        });
    return output;

  } else {
    // This is needed for non-contiguous tensors.
    Tensor input = input_.contiguous();

    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options().dtype(toQIntType(input.scalar_type())),
        input.q_scale(),
        input.q_zero_point());
    Tensor output_c = output.is_contiguous()
        ? output
        : at::empty(output.sizes(), output.options());
    int nc = nbatch * channels;

    IPEX_DISPATCH_QINT_TYPES(
        input.scalar_type(), "q_upsample_nearest2d", [&]() {
          auto idata =
              reinterpret_cast<underlying_t*>(input.data_ptr<scalar_t>());
          auto odata =
              reinterpret_cast<underlying_t*>(output_c.data_ptr<scalar_t>());

          at::AtenIpexTypeXPU::upsample_nearest2d_out_kernel<underlying_t>(
              idata,
              odata,
              nc,
              input_height,
              input_width,
              output_height,
              output_width,
              height_scale,
              width_scale,
              index_op);
        });

    if (!output.is_contiguous()) {
      output.copy_(output_c);
    }
    return output;
  }
}

Tensor upsample_nearest2d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);
  bool onednn_path =
      (output_height % input_height == 0) && (output_width % input_width == 0);
  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (onednn_path) {
    Tensor output = at::_empty_affine_quantized(
        input.sizes(),
        input.options().dtype(toQIntType(input.scalar_type())),
        input.q_scale(),
        input.q_zero_point());
    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_nearest,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
    return output;
  } else {
    to_plain_if_needed_(input);

    return q_upsample_nearest2d_template(
        input,
        output_size,
        scales_h,
        scales_w,
        at::AtenIpexTypeXPU::Nearest_index_op());
  }
}

Tensor upsample_nearest3d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_INTERNAL_ASSERT(
      false,
      "upsample_nearest3d dosen't support quantized input, we will enable it in the future.");
}

Tensor upsample_nearest3d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  TORCH_INTERNAL_ASSERT(
      false,
      "upsample_nearest3d dosen't support quantized input, we will enable it in the future.");
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
