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

inline int imin(int a, int b) {
  return a > b ? b : a;
}
inline int imax(int a, int b) {
  return a > b ? a : b;
}

template <typename scalar_t, typename F>
void parallel_replication_pad1d(
    PackedTensorAccessor64<scalar_t, 3> input,
    PackedTensorAccessor64<scalar_t, 3> output,
    int64_t pad_left,
    int64_t pad_right,
    const F& f) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t output_plane_size = output.size(2);
  int work_group_size = output_plane_size > 256 ? 256 : output_plane_size;
  int work_group_num = CeilDiv(output_plane_size, (int64_t)256);
  int64_t nplane = output.size(1);
  int64_t nbatch = output.size(0);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto output_id = item.get_global_id(0);
      if (output_id > output_plane_size) {
        return;
      }
      int64_t output_x = output_id % output.size(2);
      int64_t i_start_x = imax(0, -pad_left);
      int64_t o_start_x = imax(0, pad_left);
      int64_t input_x =
          imin(imax(pad_left, output_x), input.size(2) + pad_left - 1) -
          o_start_x + i_start_x;

      f(input, output, item.get_group(1), item.get_group(2), output_x, input_x);
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
void replication_pad1d_forward_kernel(
    PackedTensorAccessor64<scalar_t, 3> input,
    PackedTensorAccessor64<scalar_t, 3> output,
    int64_t pad_left,
    int64_t pad_right) {
  parallel_replication_pad1d(
      input,
      output,
      pad_left,
      pad_right,
      [&](PackedTensorAccessor64<scalar_t, 3> input,
          PackedTensorAccessor64<scalar_t, 3> output,
          int64_t plane,
          int64_t batch,
          int64_t output_x,
          int64_t intput_x) {
        auto value_to_copy = input[batch][plane][intput_x];
        output[batch][plane][output_x] = value_to_copy;
      });
}

template <typename scalar_t>
void replication_pad1d_backward_kernel(
    PackedTensorAccessor64<scalar_t, 3> grad_input,
    PackedTensorAccessor64<scalar_t, 3> grad_output,
    int64_t pad_left,
    int64_t pad_right) {
  parallel_replication_pad1d(
      grad_input,
      grad_output,
      pad_left,
      pad_right,
      [&](PackedTensorAccessor64<scalar_t, 3> grad_input,
          PackedTensorAccessor64<scalar_t, 3> grad_output,
          int64_t plane,
          int64_t batch,
          int64_t output_x,
          int64_t intput_x) {
        auto value_to_add = grad_output[batch][plane][output_x];
        auto target =
            (dpcpp_global_ptr_pt<scalar_t>)&grad_input[batch][plane][intput_x];
        atomicAdd(target, value_to_add);
      });
}

template <typename scalar_t>
void replication_pad_forward_kernel2d(
    scalar_t* input,
    scalar_t* output,
    int64_t padT,
    int64_t padB,
    int64_t padL,
    int64_t padR,
    int o0,
    int o1,
    int o2,
    int o3,
    int i0,
    int i1,
    int i2,
    int i3) {
  auto& queue = dpcppGetCurrentQueue();
  int outputPlaneSize = o2 * o3;
  int workgroup_size = outputPlaneSize > 256 ? 256 : outputPlaneSize;
  // clcle
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      int outputPointId =
          item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
      int plane = item.get_group(1);
      int batch = item.get_group(2);
      if (outputPointId >= o2 * o3)
        return;
      int outputPointX = outputPointId % o3;
      int outputPointY = outputPointId / o3;
      int iStartX = imax(0, -padL);
      int iStartY = imax(0, -padT);
      int oStartX = imax(0, padL);
      int oStartY = imax(0, padT);
      int inputPointX =
          imin(imax(padL, outputPointX), i3 + padL - 1) - oStartX + iStartX;
      int inputPointY =
          imin(imax(padT, outputPointY), i2 + padT - 1) - oStartY + iStartY;

      size_t in_ = ((batch * i1 + plane) * i2 + inputPointY) * i3 + inputPointX;
      scalar_t valueToCopy = input[in_];
      size_t out_ =
          ((batch * o1 + plane) * o2 + outputPointY) * o3 + outputPointX;
      output[out_] = valueToCopy;
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(
                CeilDiv(outputPlaneSize, workgroup_size) * workgroup_size,
                o1,
                o0),
            sycl::range<3>(workgroup_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

void replication_pad2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(paddingSize.size() == 4, "padding Size is expected to be 4");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int padT = paddingSize[2];
  int padB = paddingSize[3];
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int numInputDims = input.dim();
  TORCH_CHECK(
      input.numel() && (numInputDims == 3 || numInputDims == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input, but got: ",
      input)

  if (numInputDims == 4) {
    numBatch = input.size(0);
    planeDim++;
    dimh++;
    dimw++;
  }

  int numPlanes = input.size(planeDim);
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);
  int outputH = inputH + padT + padB;
  int outputW = inputW + padL + padR;

  TORCH_CHECK(
      outputW >= 1 || outputH >= 1,
      "input (H: ",
      inputH,
      ", W: ",
      inputW,
      ") is too small."
      " Calculated output H: ",
      outputH,
      " W: ",
      outputW);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "replication_pad2d_out_template",
      [&] {
        if (numInputDims == 3) {
          output.resize_({numPlanes, outputH, outputW});
          auto input_ = input.unsqueeze(0);
          auto output_ = output.unsqueeze(0);
          int o0 = output_.size(0);
          int o1 = output_.size(1);
          int o2 = output_.size(2);
          int o3 = output_.size(3);
          int i0 = input_.size(0);
          int i1 = input_.size(1);
          int i2 = input_.size(2);
          int i3 = input_.size(3);
          replication_pad_forward_kernel2d<scalar_t>(
              input_.data_ptr<scalar_t>(),
              output_.data_ptr<scalar_t>(),
              padT,
              padB,
              padL,
              padR,
              o0,
              o1,
              o2,
              o3,
              i0,
              i1,
              i2,
              i3);

        } else {
          output.resize_({numBatch, numPlanes, outputH, outputW});
          int o0 = output.size(0);
          int o1 = output.size(1);
          int o2 = output.size(2);
          int o3 = output.size(3);
          int i0 = input.size(0);
          int i1 = input.size(1);
          int i2 = input.size(2);
          int i3 = input.size(3);
          replication_pad_forward_kernel2d<scalar_t>(
              input.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              padT,
              padB,
              padL,
              padR,
              o0,
              o1,
              o2,
              o3,
              i0,
              i1,
              i2,
              i3);
        }
      });
}

template <typename scalar_t>
void replication_pad_backward_kernel(
    scalar_t* gradInput,
    scalar_t* gradOutput,
    int padT,
    int padB,
    int padL,
    int padR,
    int go0,
    int go1,
    int go2,
    int go3,
    int gi0,
    int gi1,
    int gi2,
    int gi3) {
  auto& queue = dpcppGetCurrentQueue();
  int outputPlaneSize = go2 * go3;
  int workgroup_size = outputPlaneSize > 256 ? 256 : outputPlaneSize;
  // clcle
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      int outputPointId =
          item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
      int plane = item.get_group(1);
      int batch = item.get_group(2);
      if (outputPointId >= go2 * go3)
        return;
      int outputPointX = outputPointId % go3;
      int outputPointY = outputPointId / go3;
      int iStartX = imax(0, -padL);
      int iStartY = imax(0, -padT);
      int oStartX = imax(0, padL);
      int oStartY = imax(0, padT);
      int inputPointX =
          imin(imax(padL, outputPointX), gi3 + padL - 1) - oStartX + iStartX;
      int inputPointY =
          imin(imax(padT, outputPointY), gi2 + padT - 1) - oStartY + iStartY;
      size_t go_ =
          ((batch * go1 + plane) * go2 + outputPointY) * go3 + outputPointX;
      scalar_t valueToCopy = gradOutput[go_];
      size_t gi_ =
          ((batch * gi1 + plane) * gi2 + inputPointY) * gi3 + inputPointX;
      atomicAdd((dpcpp_global_ptr_pt<scalar_t>)&gradInput[gi_], valueToCopy);
    };
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(
                CeilDiv(outputPlaneSize, workgroup_size) * workgroup_size,
                go1,
                go0),
            sycl::range<3>(workgroup_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

void replication_pad2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(gradOutput),
      "output gradient tensor must fit into 32-bit index math");
  TORCH_CHECK(paddingSize.size() == 4, "padding Size is expected to be 4");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int padT = paddingSize[2];
  int padB = paddingSize[3];
  int dimh = 1;
  int dimw = 2;

  int numInputDims = input.dim();
  if (numInputDims == 4) {
    dimh++;
    dimw++;
  }
  int iheight = input.size(dimh);
  int iwidth = input.size(dimw);
  int oheight = iheight + padT + padB;
  int owidth = iwidth + padL + padR;

  TORCH_CHECK(
      owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ",
      owidth,
      ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(
      oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ",
      oheight,
      ", Got: ",
      gradOutput.size(dimh));

  gradInput.resize_as_(input);
  gradInput.zero_();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "replication_pad2d_backward_out_template",
      [&] {
        auto gradInput_ = gradInput;
        auto gradOutput_ = gradOutput;
        if (numInputDims == 3) {
          gradInput_ = gradInput.unsqueeze(0);
          gradOutput_ = gradOutput.unsqueeze(0);
        }
        int go0 = gradOutput_.size(0);
        int go1 = gradOutput_.size(1);
        int go2 = gradOutput_.size(2);
        int go3 = gradOutput_.size(3);
        int gi0 = gradInput_.size(0);
        int gi1 = gradInput_.size(1);
        int gi2 = gradInput_.size(2);
        int gi3 = gradInput_.size(3);
        replication_pad_backward_kernel<scalar_t>(
            gradInput_.data_ptr<scalar_t>(),
            gradOutput_.data_ptr<scalar_t>(),
            padT,
            padB,
            padL,
            padR,
            go0,
            go1,
            go2,
            go3,
            gi0,
            gi1,
            gi2,
            gi3);
      });
}

template <typename scalar_t, typename F>
void parallel_replication_pad3d(
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
  int64_t nplane = output.size(1);
  int64_t nbatch = output.size(0);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto output_id = item.get_global_id(0);
      if (output_id > output_plane_size) {
        return;
      }
      int64_t output_x = output_id % output.size(4);
      int64_t output_y = (output_id / output.size(4)) % output.size(3);
      int64_t output_z = output_id / (output.size(3) * output.size(4));

      int64_t i_start_x = imax(0, -pad_left);
      int64_t i_start_y = imax(0, -pad_top);
      int64_t i_start_z = imax(0, -pad_front);
      int64_t o_start_x = imax(0, pad_left);
      int64_t o_start_y = imax(0, pad_top);
      int64_t o_start_z = imax(0, pad_front);

      int64_t input_x =
          imin(imax(pad_left, output_x), input.size(4) + pad_left - 1) -
          o_start_x + i_start_x;
      int64_t input_y =
          imin(imax(pad_top, output_y), input.size(3) + pad_top - 1) -
          o_start_y + i_start_y;
      int64_t input_z =
          imin(imax(pad_front, output_z), input.size(2) + pad_front - 1) -
          o_start_z + i_start_z;

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
void replication_pad3d_forward_kernel(
    PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  parallel_replication_pad3d(
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
          int64_t intput_z,
          int64_t intput_y,
          int64_t intput_x) {
        auto value_to_copy = input[batch][plane][intput_z][intput_y][intput_x];
        output[batch][plane][output_z][output_y][output_x] = value_to_copy;
      });
}

template <typename scalar_t>
void replication_pad3d_backward_kernel(
    PackedTensorAccessor64<scalar_t, 5> grad_input,
    PackedTensorAccessor64<scalar_t, 5> grad_output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  parallel_replication_pad3d(
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
          int64_t intput_z,
          int64_t intput_y,
          int64_t intput_x) {
        auto value_to_add =
            grad_output[batch][plane][output_z][output_y][output_x];
        auto target =
            (dpcpp_global_ptr_pt<scalar_t>)&grad_input[batch][plane][intput_z]
                                                      [intput_y][intput_x];
        atomicAdd(target, value_to_add);
      });
}

static inline void shapeAndGradOutputCheck3d(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t pad_front,
    int64_t pad_back) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  int64_t num_input_dims = input.dim();

  bool valid_dims =
      input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
  TORCH_CHECK(
      (num_input_dims == 4 && valid_dims) ||
          (num_input_dims == 5 && valid_dims && input.size(4) != 0),
      "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

  int plane_dim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  if (num_input_dims == 5) {
    plane_dim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int64_t num_planes = input.size(plane_dim);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pad_front + pad_back;
  int64_t oheight = iheight + pad_top + pad_bottom;
  int64_t owidth = iwidth + pad_left + pad_right;
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

  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(grad_output),
      "output gradient tensor must fit into 32-bit index math");

  TORCH_CHECK(
      num_planes == grad_output.size(plane_dim),
      "grad_output width unexpected. Expected: ",
      num_planes,
      ", Got: ",
      grad_output.size(plane_dim));
  TORCH_CHECK(
      owidth == grad_output.size(dimw),
      "grad_output width unexpected. Expected: ",
      owidth,
      ", Got: ",
      grad_output.size(dimw));
  TORCH_CHECK(
      oheight == grad_output.size(dimh),
      "grad_output height unexpected. Expected: ",
      oheight,
      ", Got: ",
      grad_output.size(dimh));
  TORCH_CHECK(
      odepth == grad_output.size(dimd),
      "grad_output depth unexpected. Expected: ",
      odepth,
      ", Got: ",
      grad_output.size(dimd));
}

void replication_pad3d_backward_out_dpcpp_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  grad_input.resize_as_(input);
  if (grad_input.numel() == 0) {
    return;
  }
  grad_input.zero_();

  TORCH_CHECK(padding.size() == 6, "padding Size is expected to be 6");
  int pad_left = padding[0];
  int pad_right = padding[1];
  int pad_top = padding[2];
  int pad_bottom = padding[3];
  int pad_front = padding[4];
  int pad_back = padding[5];
  shapeAndGradOutputCheck3d(
      input,
      grad_output,
      pad_left,
      pad_right,
      pad_top,
      pad_bottom,
      pad_front,
      pad_back);
  int num_input_dims = input.dim();

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "replication_pad3d_backward_dpcpp",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (num_input_dims == 4) {
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }
        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 5>();
        auto grad_output_packed = grad_output_.packed_accessor64<scalar_t, 5>();
        replication_pad3d_backward_kernel<scalar_t>(
            grad_input_packed,
            grad_output_packed,
            pad_left,
            pad_top,
            pad_front);
      });
}

} // namespace impl

Tensor& replication_pad1d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");

  if (output.numel() == 0) {
    return output;
  }

  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t num_input_dims = input.dim();

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "replication_pad1d_dpcpp", [&] {
        auto input_ = input;
        auto output_ = output;
        if (num_input_dims == 2) {
          input_ = input.unsqueeze(0);
          output_ = output.unsqueeze(0);
        }

        auto input_packed = input_.packed_accessor64<scalar_t, 3>();
        auto output_packed = output_.packed_accessor64<scalar_t, 3>();

        impl::replication_pad1d_forward_kernel<scalar_t>(
            input_packed, output_packed, pad_left, pad_right);
      });

  return output;
}

Tensor& replication_pad1d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      xpu::dpcpp::detail::canUse32BitIndexMath(grad_output),
      "output gradient tensor must fit into 32-bit index math");

  if (grad_input.numel() == 0) {
    return grad_input;
  }
  grad_input.zero_();

  int pad_left = padding[0];
  int pad_right = padding[1];
  int num_input_dims = input.ndimension();

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "replication_pad1d_backward_dpcpp",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (num_input_dims == 2) {
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }
        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 3>();
        auto grad_output_packed = grad_output_.packed_accessor64<scalar_t, 3>();

        impl::replication_pad1d_backward_kernel<scalar_t>(
            grad_input_packed, grad_output_packed, pad_left, pad_right);
      });
  return grad_input;
}

Tensor& replication_pad2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize) {
  impl::replication_pad2d_out_template(output, input, paddingSize);
  return output;
}

Tensor replication_pad2d(const Tensor& input, IntArrayRef paddingSize) {
  auto output = at::empty({0}, input.options());
  impl::replication_pad2d_out_template(output, input, paddingSize);
  return output;
}

Tensor& replication_pad2d_backward_out(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize) {
  impl::replication_pad2d_backward_out_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize) {
  auto gradInput = at::zeros_like(input, MemoryFormat::Contiguous);
  impl::replication_pad2d_backward_out_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor& replication_pad3d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  if (output.numel() == 0) {
    return output;
  }
  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  int64_t num_input_dims = input.dim();

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "replication_pad3d_dpcpp", [&] {
        auto input_ = input;
        auto output_ = output;
        if (num_input_dims == 4) {
          input_ = input.unsqueeze(0);
          output_ = output.unsqueeze(0);
        }

        auto input_packed = input_.packed_accessor64<scalar_t, 5>();
        auto output_packed = output_.packed_accessor64<scalar_t, 5>();

        impl::replication_pad3d_forward_kernel<scalar_t>(
            input_packed, output_packed, pad_left, pad_top, pad_front);
      });
  return output;
}

Tensor& replication_pad3d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  impl::replication_pad3d_backward_out_dpcpp_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor replication_pad3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  impl::replication_pad3d_backward_out_dpcpp_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
