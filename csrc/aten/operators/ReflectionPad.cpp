#include <ATen/ATen.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

inline std::pair<int64_t, int64_t> get_index_mapping1d(
    int64_t input_w,
    int64_t output_w,
    int64_t output_x,
    int64_t pad_l,
    const sycl::nd_item<3> item) {
  auto input_offset =
      (item.get_group(1) + item.get_group(2) * item.get_group_range(1)) *
      input_w;
  auto output_offset =
      (item.get_group(1) + item.get_group(2) * item.get_group_range(1)) *
      output_w;

  auto i_start_x = Max(int64_t(0), -pad_l);
  auto o_start_x = Max(int64_t(0), pad_l);

  int64_t input_x = Numerics<int64_t>::abs(output_x - pad_l) -
      Numerics<int64_t>::abs(output_x - (input_w + pad_l - 1)) - output_x +
      2 * pad_l + input_w - 1 - o_start_x + i_start_x;

  return std::make_pair<int64_t, int64_t>(
      input_offset + input_x, output_offset + output_x);
}

inline std::pair<int64_t, int64_t> get_index_mapping2d(
    int64_t input_dim_x,
    int64_t input_dim_y,
    int64_t output_dim_x,
    int64_t output_dim_y,
    int64_t pad_l,
    int64_t pad_t,
    int64_t output_xy,
    const sycl::nd_item<3> item) {
  // 3D grid of 1D blocks
  auto input_offset =
      (item.get_group(1) + item.get_group(2) * item.get_group_range(1)) *
      input_dim_x * input_dim_y;
  auto output_offset =
      (item.get_group(1) + item.get_group(2) * item.get_group_range(1)) *
      output_dim_x * output_dim_y;

  auto output_x = output_xy % output_dim_x;
  auto output_y = output_xy / output_dim_x;

  auto i_start_x = Max(int64_t(0), -pad_l);
  auto i_start_y = Max(int64_t(0), -pad_t);
  auto o_start_x = Max(int64_t(0), pad_l);
  auto o_start_y = Max(int64_t(0), pad_t);

  int64_t input_x = Numerics<int64_t>::abs(output_x - pad_l) -
      Numerics<int64_t>::abs(output_x - (input_dim_x + pad_l - 1)) - output_x +
      2 * pad_l + input_dim_x - 1 - o_start_x + i_start_x;

  int64_t input_y = Numerics<int64_t>::abs(output_y - pad_t) -
      Numerics<int64_t>::abs(output_y - (input_dim_y + pad_t - 1)) - output_y +
      2 * pad_t + input_dim_y - 1 - o_start_y + i_start_y;

  return std::make_pair<int64_t, int64_t>(
      input_offset + input_y * input_dim_x + input_x,
      output_offset + output_y * output_dim_x + output_x);
}

template <typename scalar_t>
void reflection_pad1d_out_kernel(
    scalar_t* input,
    scalar_t* output,
    int64_t input_w,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane,
    int64_t output_w) {
  auto& queue = dpcppGetCurrentQueue();
  int work_group_size = output_w > 256 ? 256 : output_w;
  int work_group_num = CeilDiv(output_w, (int64_t)256);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input;
    auto output_data = output;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto input_ptr = input_data;
      auto output_ptr = output_data;
      auto output_x = item.get_global_id(0);

      if (output_x < output_w) {
        // input index and output index mapping
        auto index_pair =
            get_index_mapping1d(input_w, output_w, output_x, pad_l, item);
        output_ptr[index_pair.second] = input_ptr[index_pair.first];
      }
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
            sycl::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void reflection_pad2d_out_kernel(
    scalar_t* input,
    scalar_t* output,
    int64_t input_dim_x,
    int64_t input_dim_y,
    int64_t pad_t,
    int64_t pad_b,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane) {
  int64_t output_dim_x = input_dim_x + pad_l + pad_r;
  int64_t output_dim_y = input_dim_y + pad_t + pad_b;

  auto& queue = dpcppGetCurrentQueue();
  int work_group_size =
      output_dim_x * output_dim_y > 256 ? 256 : output_dim_x * output_dim_y;
  int work_group_num = CeilDiv(output_dim_x * output_dim_y, (int64_t)256);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input;
    auto output_data = output;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto input_ptr = input_data;
      auto output_ptr = output_data;
      auto output_xy = item.get_global_id(0);

      if (output_xy < output_dim_x * output_dim_y) {
        // input index and output index mapping
        auto index_pair = get_index_mapping2d(
            input_dim_x,
            input_dim_y,
            output_dim_x,
            output_dim_y,
            pad_l,
            pad_t,
            output_xy,
            item);
        output_ptr[index_pair.second] = input_ptr[index_pair.first];
      }
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
            sycl::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

void reflection_pad1d_out_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef padding) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  TORCH_CHECK(
      (input_.ndimension() == 2 && input_.size(1) != 0) ||
          (input_.ndimension() == 3 && input_.size(1) != 0 &&
           input_.size(2) != 0),
      "2D or 3D (batch mode) tensor expected for input, but got: ",
      input_);

  if (input_.ndimension() == 3) {
    nbatch = input_.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input_.size(dim_plane);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Padding size should be less "
      "than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input_);

  TORCH_CHECK(
      output_w >= 1,
      "input (W: ",
      input_w,
      ")is too small. Calculated output W: ",
      output_w);

  if (input_.ndimension() == 2) {
    output.resize_({nplane, output_w});
  } else {
    output.resize_({nbatch, nplane, output_w});
  }

  if (output.numel() == 0)
    return;
  Tensor input = input_.contiguous();

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      kHalf, input.scalar_type(), "reflection_pad1d_out_template", [&] {
        reflection_pad1d_out_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_w,
            pad_l,
            pad_r,
            nbatch,
            nplane,
            output_w);
      });
}

void reflection_pad2d_out_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef padding) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  int plane_dim = 0;
  int dim_h = 1;
  int dim_w = 2;
  int nbatch = 1;

  bool valid_dims = input_.size(1) != 0 && input_.size(2) != 0;
  TORCH_CHECK(
      (input_.ndimension() == 3 && valid_dims) ||
          (input_.ndimension() == 4 && valid_dims && input_.size(3) != 0),
      "3D or 4D (batch mode) tensor expected for input, but got: ",
      input_);

  if (input_.ndimension() == 4) {
    nbatch = input_.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input_.size(plane_dim);
  int64_t input_h = input_.size(dim_h);
  int64_t input_w = input_.size(dim_w);

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input_.sizes());

  TORCH_CHECK(
      pad_t < input_h && pad_b < input_h,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_t,
      ", ",
      pad_b,
      ") at dimension ",
      dim_h,
      " of input ",
      input_.sizes());
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w >= 1 || output_h >= 1,
      "input (H: ",
      input_h,
      ", W: ",
      input_w,
      ")is too small.  Calculated "
      "output H: ",
      output_h,
      " W: ",
      output_w);

  if (input_.ndimension() == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    output.resize_({nbatch, nplane, output_h, output_w});
  }

  Tensor input = input_.contiguous();

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      kHalf, input.scalar_type(), "reflection_pad2d_out_template", [&] {
        reflection_pad2d_out_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_w,
            input_h,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            nbatch,
            nplane);
      });
}

template <typename scalar_t>
void reflection_pad1d_backward_out_kernel(
    scalar_t* grad_input,
    scalar_t* grad_output,
    int64_t input_w,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane,
    int64_t output_w) {
  auto& queue = dpcppGetCurrentQueue();
  int work_group_size = output_w > 256 ? 256 : output_w;
  int work_group_num = CeilDiv(output_w, (int64_t)256);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input;
    auto grad_output_data = grad_output;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto grad_input_ptr = grad_input_data;
      auto grad_output_ptr = grad_output_data;
      auto output_x = item.get_global_id(0);

      if (output_x < output_w) {
        // grad input index and grad output index mapping
        auto index_pair =
            get_index_mapping1d(input_w, output_w, output_x, pad_l, item);
        atomicAdd(
            (dpcpp_global_ptr_pt<scalar_t>)&grad_input_ptr[index_pair.first],
            grad_output_ptr[index_pair.second]);
      }
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
            sycl::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void reflection_pad2d_backward_out_kernel(
    scalar_t* grad_input,
    scalar_t* grad_output,
    int64_t input_dim_x,
    int64_t input_dim_y,
    int64_t pad_t,
    int64_t pad_b,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t output_dim_x = input_dim_x + pad_l + pad_r;
  int64_t output_dim_y = input_dim_y + pad_t + pad_b;
  int work_group_size =
      output_dim_x * output_dim_y > 256 ? 256 : output_dim_x * output_dim_y;
  int work_group_num = CeilDiv(output_dim_x * output_dim_y, (int64_t)256);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input;
    auto grad_output_data = grad_output;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto grad_input_ptr = grad_input_data;
      auto grad_output_ptr = grad_output_data;
      auto output_xy = item.get_global_id(0);

      if (output_xy < output_dim_x * output_dim_y) {
        // grad input index and grad output index mapping
        auto index_pair = get_index_mapping2d(
            input_dim_x,
            input_dim_y,
            output_dim_x,
            output_dim_y,
            pad_l,
            pad_t,
            output_xy,
            item);
        atomicAdd(
            (dpcpp_global_ptr_pt<scalar_t>)&grad_input_ptr[index_pair.first],
            grad_output_ptr[index_pair.second]);
      }
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
            sycl::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

void reflection_pad1d_backward_out_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");

  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(grad_output_),
      "input tensor must fit into 32-bit index math");

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  Tensor grad_output = grad_output_.contiguous();

  TORCH_CHECK(
      output_w == grad_output.size(dim_w),
      "gradOutput width unexpected. Expected: ",
      output_w,
      ", Got: ",
      grad_output.size(dim_w));
  IPEX_DISPATCH_ATOMIC_FLOATING_AND_COMPLEX_TYPES(
      grad_input.scalar_type(), "reflection_pad1d_backward_out_template", [&] {
        reflection_pad1d_backward_out_kernel<scalar_t>(
            grad_input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            input_w,
            pad_l,
            pad_r,
            nbatch,
            nplane,
            output_w);
      });
}

void reflection_pad2d_backward_out_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(grad_output_),
      "output gradient tensor must fit into 32-bit index math");

  int64_t plane_dim = 0;
  int64_t dim_h = 1;
  int64_t dim_w = 2;
  int64_t nbatch = 1;

  if (input.ndimension() == 4) {
    nbatch = input.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input.size(plane_dim);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);

  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w == grad_output_.size(dim_w),
      "grad_output width "
      "unexpected. Expected: ",
      output_w,
      ", Got: ",
      grad_output_.size(dim_w));
  TORCH_CHECK(
      output_h == grad_output_.size(dim_h),
      "grad_output height "
      "unexpected. Expected: ",
      output_h,
      ", Got: ",
      grad_output_.size(dim_h));

  Tensor grad_output = grad_output_.contiguous();

  TORCH_CHECK(
      output_w == grad_output.size(dim_w),
      "gradOutput width unexpected. Expected: ",
      output_w,
      ", Got: ",
      grad_output.size(dim_w));
  IPEX_DISPATCH_ATOMIC_FLOATING_AND_COMPLEX_TYPES(
      grad_input.scalar_type(), "reflection_pad2d_backward_out_template", [&] {
        reflection_pad2d_backward_out_kernel<scalar_t>(
            grad_input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            input_w,
            input_h,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            nbatch,
            nplane);
      });
}

template <typename scalar_t, typename F>
inline void parallel_reflection_pad3d(
    PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front,
    const F& f) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t output_plane_size = output.size(2) * output.size(3) * output.size(4);
  int work_group_size = output_plane_size > 256 ? 256 : output_plane_size;
  int work_group_num = CeilDiv(output_plane_size, (int64_t)256);
  int64_t nplane = input.size(1);
  int64_t nbatch = input.size(0);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto output_id = item.get_global_id(0);
      if (output_id > output_plane_size) {
        return;
      }

      int64_t output_x = output_id % output.size(4);
      int64_t output_y = (output_id / output.size(4)) % output.size(3);
      int64_t output_z = output_id / (output.size(3) * output.size(4));

      int64_t i_start_x = Numerics<int64_t>::max(int64_t(0), -pad_left);
      int64_t o_start_x = Numerics<int64_t>::max(int64_t(0), pad_left);
      int64_t i_start_y = Numerics<int64_t>::max(int64_t(0), -pad_top);
      int64_t o_start_y = Numerics<int64_t>::max(int64_t(0), pad_top);
      int64_t i_start_z = Numerics<int64_t>::max(int64_t(0), -pad_front);
      int64_t o_start_z = Numerics<int64_t>::max(int64_t(0), pad_front);

      int64_t input_x = Numerics<int64_t>::abs(output_x - pad_left) -
          Numerics<int64_t>::abs(output_x - (input.size(4) + pad_left - 1)) -
          output_x + 2 * pad_left + input.size(4) - 1 - o_start_x + i_start_x;
      int64_t input_y = Numerics<int64_t>::abs(output_y - pad_top) -
          Numerics<int64_t>::abs(output_y - (input.size(3) + pad_top - 1)) -
          output_y + 2 * pad_top + input.size(3) - 1 - o_start_y + i_start_y;

      int64_t input_z = Numerics<int64_t>::abs(output_z - pad_front) -
          Numerics<int64_t>::abs(output_z - (input.size(2) + pad_front - 1)) -
          output_z + 2 * pad_front + input.size(2) - 1 - o_start_z + i_start_z;

      f(input,
        output,
        item.get_group(1),
        item.get_group(2),
        output_z,
        output_y,
        output_x,
        input_z,
        input_y,
        input_x);
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
            sycl::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void reflection_pad3d_out_kernel(
    PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  parallel_reflection_pad3d(
      input,
      output,
      pad_left,
      pad_top,
      pad_front,
      [&](PackedTensorAccessor64<scalar_t, 5> input,
          PackedTensorAccessor64<scalar_t, 5> output,
          int64_t plane,
          int64_t batch,
          int64_t output_z,
          int64_t output_y,
          int64_t output_x,
          int64_t input_z,
          int64_t input_y,
          int64_t input_x) {
        auto value_to_copy = input[batch][plane][input_z][input_y][input_x];
        output[batch][plane][output_z][output_y][output_x] = value_to_copy;
      });
}

template <typename scalar_t>
void reflection_pad3d_backward_out_kernel(
    PackedTensorAccessor64<scalar_t, 5> grad_input,
    PackedTensorAccessor64<scalar_t, 5> grad_output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  parallel_reflection_pad3d(
      grad_input,
      grad_output,
      pad_left,
      pad_top,
      pad_front,
      [&](PackedTensorAccessor64<scalar_t, 5> grad_input,
          PackedTensorAccessor64<scalar_t, 5> grad_output,
          int64_t plane,
          int64_t batch,
          int64_t output_z,
          int64_t output_y,
          int64_t output_x,
          int64_t input_z,
          int64_t input_y,
          int64_t input_x) {
        auto value_to_add =
            grad_output[batch][plane][output_z][output_y][output_x];
        auto target =
            (dpcpp_global_ptr_pt<scalar_t>)&grad_input[batch][plane][input_z]
                                                      [input_y][input_x];
        atomicAdd(target, value_to_add);
      });
}

} // namespace impl

Tensor& reflection_pad1d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  impl::reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad1d(const Tensor& input, IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  return at::AtenIpexTypeXPU::reflection_pad1d_out(input, padding, output);
}

Tensor& reflection_pad2d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  impl::reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad2d(const Tensor& input, IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  return at::AtenIpexTypeXPU::reflection_pad2d_out(input, padding, output);
}

Tensor& reflection_pad1d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  impl::reflection_pad1d_backward_out_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& reflection_pad2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  impl::reflection_pad2d_backward_out_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad1d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, MemoryFormat::Contiguous);
  impl::reflection_pad1d_backward_out_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, MemoryFormat::Contiguous);
  impl::reflection_pad2d_backward_out_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& reflection_pad3d_out(
    const Tensor& input_,
    IntArrayRef padding,
    Tensor& output) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  if (output.numel() == 0) {
    return output;
  }

  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  auto input = input_.contiguous();
  bool batch_mode = (input.dim() == 5);

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "reflection_pad3d_out_dpcpp", [&] {
        auto input_inner = input;
        auto output_inner = output;
        if (!batch_mode) {
          input_inner = input.unsqueeze(0);
          output_inner = output.unsqueeze(0);
        }

        auto input_packed = input_inner.packed_accessor64<scalar_t, 5>();
        auto output_packed = output_inner.packed_accessor64<scalar_t, 5>();

        impl::reflection_pad3d_out_kernel<scalar_t>(
            input_packed, output_packed, pad_left, pad_top, pad_front);
      });

  return output;
}

Tensor& reflection_pad3d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(grad_output),
      "input tensor must fit into 32-bit index math");

  if (grad_input.numel() == 0) {
    return grad_input;
  }
  grad_input.zero_();

  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "reflection_pad3d_backward_out_dpcpp",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (input.dim() == 4) {
          // non-batch mode
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }

        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 5>();
        auto grad_output_packed = grad_output_.packed_accessor64<scalar_t, 5>();

        impl::reflection_pad3d_backward_out_kernel<scalar_t>(
            grad_input_packed,
            grad_output_packed,
            pad_left,
            pad_top,
            pad_front);
      });
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
