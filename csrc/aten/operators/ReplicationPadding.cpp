#include <ATen/ATen.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

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
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item) {
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
        DPCPP::nd_range<3>(
            DPCPP::range<3>(
                CeilDiv(outputPlaneSize, workgroup_size) * workgroup_size,
                o1,
                o0),
            DPCPP::range<3>(workgroup_size, 1, 1)),
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
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item) {
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
        DPCPP::nd_range<3>(
            DPCPP::range<3>(
                CeilDiv(outputPlaneSize, workgroup_size) * workgroup_size,
                go1,
                go0),
            DPCPP::range<3>(workgroup_size, 1, 1)),
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
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;

  int numInputDims = input.dim();
  if (numInputDims == 4) {
    planeDim++;
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

} // namespace impl

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

} // namespace AtenIpexTypeXPU
} // namespace at
