#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename accscalar_t>
inline int get_interval(
    accscalar_t sample,
    int index,
    int inputSize,
    int outputSize,
    int poolSize) {
  accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
      static_cast<accscalar_t>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int>((index + sample) * alpha) -
        static_cast<int>(sample * alpha);
  }
}

template <typename scalar_t>
void fractional_max_pool2d_out_frame(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    scalar_t* samples,
    int numBatch,
    int numPlane,
    int inputSizeH,
    int inputSizeW,
    int outputSizeH,
    int outputSizeW,
    int poolSizeH,
    int poolSizeW) {
  using accscalar_t = acc_type<scalar_t>;
  auto& queue = dpcppGetCurrentQueue();
  int outputPlaneSize = outputSizeH * outputSizeW;
  int work_group_size = outputPlaneSize > 128 ? 128 : outputPlaneSize;
  int work_group_num = (outputPlaneSize + 127) / 128;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input;
    auto output_data = output;
    auto indices_data = indices;
    auto samples_data = samples;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item) {
      auto input_ptr = input_data;
      auto output_ptr = output_data;
      auto indices_ptr = indices_data;
      auto samples_ptr = samples_data;

      int ourOutputPoint = item.get_global_id()[0];
      int plane = item.get_group()[1];
      int batch = item.get_group()[2];

      if (ourOutputPoint < outputSizeH * outputSizeW) {
        int outputW = ourOutputPoint % outputSizeW;
        int outputH = ourOutputPoint / outputSizeW;

        int poolW = get_interval<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(
                samples_ptr
                    [batch * numPlane * 2 + plane * 2] /*[batch][plane][0] */),
            outputW,
            inputSizeW,
            outputSizeW,
            poolSizeW);
        int poolH = get_interval<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(samples_ptr
                                         [batch * numPlane * 2 + plane * 2 +
                                          1] /*[batch][plane][1] */),
            outputH,
            inputSizeH,
            outputSizeH,
            poolSizeH);

        scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
        int maxIndex = -1;

        for (int h = poolH; h < poolH + poolSizeH; ++h) {
          if (poolSizeW < 2 || poolSizeW > 7) {
            for (int w = poolW; w < poolW + poolSizeW; ++w) {
              scalar_t val = input_ptr
                  [batch * numPlane * inputSizeH * inputSizeW +
                   plane * inputSizeH * inputSizeW + h * inputSizeW +
                   w] /* [batch][plane][h][w]*/;
              if (val > maxVal) {
                maxIndex = h * inputSizeW + w;
                maxVal = val;
              }
            }
          } else {
            for (int i = 0; i < poolSizeW; ++i) {
              int w = i + poolW;
              scalar_t val = input_ptr
                  [batch * numPlane * inputSizeH * inputSizeW +
                   plane * inputSizeH * inputSizeW + h * inputSizeW +
                   w] /*[batch][plane][h][w] */;
              if (val > maxVal) {
                maxIndex = h * inputSizeW + w;
                maxVal = val;
              }
            }
          }
        }

        indices_ptr
            [batch * numPlane * outputSizeH * outputSizeW +
             plane * outputSizeH * outputSizeW + outputH * outputSizeW +
             outputW] /*[batch][plane][outputH][outputW] */
            = maxIndex;
        output_ptr
            [batch * numPlane * outputSizeH * outputSizeW +
             plane * outputSizeH * outputSizeW + outputH * outputSizeW +
             outputW] /*[batch][plane][outputH][outputW]*/
            = maxVal;
      }
    };
    cgh.parallel_for(
        DPCPP::nd_range<3>(
            DPCPP::range<3>(
                work_group_size * work_group_num, numPlane, numBatch),
            DPCPP::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void fractional_max_pool2d_backward_out_frame(
    scalar_t* gradInput,
    scalar_t* gradOutput,
    int64_t* indices,
    int numBatch,
    int numPlane,
    int gradInputSizeH,
    int gradInputSizeW,
    int gradOutputSizeH,
    int gradOutputSizeW) {
  auto& queue = dpcppGetCurrentQueue();
  int gradOutputPlaneSize = gradOutputSizeH * gradOutputSizeW;
  int work_group_size = gradOutputPlaneSize > 128 ? 128 : gradOutputPlaneSize;
  int work_group_num = (gradOutputPlaneSize + 127) / 128;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = gradInput;
    auto gradOutput_data = gradOutput;
    auto indices_data = indices;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item) {
      auto gradInput_ptr = gradInput_data;
      auto gradOutput_ptr = gradOutput_data;
      auto indices_ptr = indices_data;

      int ourOutputPoint = item.get_global_id()[0];
      int plane = item.get_group()[1];
      int batch = item.get_group()[2];

      if (ourOutputPoint < gradOutputPlaneSize) {
        int outputW = ourOutputPoint % gradOutputSizeW;
        int outputH = ourOutputPoint / gradOutputSizeW;

        int index = indices_ptr
            [batch * numPlane * gradOutputSizeH * gradOutputSizeW +
             plane * gradOutputSizeH * gradOutputSizeW +
             outputH * gradOutputSizeW +
             outputW] /* [batch][plane][outputH][outputW]*/;
        int inputW = index % gradInputSizeW;
        int inputH = index / gradInputSizeW;

        atomicAdd(
            (dpcpp_global_ptr_pt<scalar_t>)&gradInput_ptr
                [batch * numPlane * gradInputSizeH * gradInputSizeW +
                 plane * gradInputSizeH * gradInputSizeW +
                 inputH * gradInputSizeW +
                 inputW] /*[batch][plane][inputH][inputW] */,
            gradOutput_ptr
                [batch * numPlane * gradOutputSizeH * gradOutputSizeW +
                 plane * gradOutputSizeH * gradOutputSizeW +
                 outputH * gradOutputSizeW +
                 outputW] /*[batch][plane][outputH][outputW]*/
        );
      }
    };
    cgh.parallel_for(
        DPCPP::nd_range<3>(
            DPCPP::range<3>(
                work_group_size * work_group_num, numPlane, numBatch),
            DPCPP::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

void fractional_max_pool2d_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& randomSamples) {
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int ndims = input.ndimension();
  TORCH_CHECK(
      input.numel() > 0,
      "fractional_max_pool2d(): expected input to have non-empty ",
      "spatial dimensions.");

  TORCH_CHECK(
      (ndims == 3 || ndims == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int numPlanes = input.size(planeDim);
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);

  int outputH = output_size[0];
  int outputW = output_size[1];
  int poolSizeH = pool_size[0];
  int poolSizeW = pool_size[1];

  TORCH_CHECK(
      outputH + poolSizeH - 1 <= inputH,
      "fractional_max_pool2d(): pool_size height ",
      poolSizeH,
      " too large relative to input height ",
      inputH);
  TORCH_CHECK(
      outputW + poolSizeW - 1 <= inputW,
      "pool_size width ",
      poolSizeW,
      " too large relative to input width ",
      inputW);

  if (ndims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW});
    indices.resize_({numBatch, numPlanes, outputH, outputW});
  }

  auto output_ = output;
  auto input_ = input.contiguous();
  auto indices_ = indices;

  if (ndims == 3) {
    output_ = output_.reshape({1, numPlanes, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputH, outputW});
    input_ = input_.reshape({1, input.size(0), input.size(1), input.size(2)});
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "fractional_max_pool2d_out_frame", [&] {
        fractional_max_pool2d_out_frame<scalar_t>(
            output_.data_ptr<scalar_t>(),
            indices_.data_ptr<int64_t>(),
            input_.data_ptr<scalar_t>(),
            randomSamples.data_ptr<scalar_t>(),
            input_.size(0),
            input_.size(1),
            inputH,
            inputW,
            outputH,
            outputW,
            poolSizeH,
            poolSizeW);
      });
}

void fractional_max_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices) {
  int dimh = 1;
  int dimw = 2;

  int ndims = input.ndimension();
  if (ndims == 4) {
    dimh++;
    dimw++;
  }

  /* sizes */
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);

  int outputH = output_size[0];
  int outputW = output_size[1];

  TORCH_CHECK(
      outputH == gradOutput.size(dimh),
      "fractional_max_pool2d(): gradOutput height unexpected");
  TORCH_CHECK(
      outputW == gradOutput.size(dimw),
      "fractional_max_pool2d(): gradOutput width unexpected");

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput.contiguous();
  auto indices_ = indices;

  if (ndims == 3) {
    gradInput_ = gradInput_.reshape({1, input.size(0), inputH, inputW});
    gradOutput_ =
        gradOutput_.reshape({1, gradOutput.size(0), outputH, outputW});
    indices_ = indices_.reshape({1, indices_.size(0), outputH, outputW});
  }

  fractional_max_pool2d_backward_out_frame<float>(
      gradInput_.data_ptr<float>(),
      gradOutput_.data_ptr<float>(),
      indices_.data_ptr<int64_t>(),
      gradInput_.size(0),
      gradInput_.size(1),
      inputH,
      inputW,
      outputH,
      outputW);
}

} // namespace impl

std::tuple<Tensor&, Tensor&> fractional_max_pool2d_out(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples) {
  impl::fractional_max_pool2d_out_template(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> fractional_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples) {
  Tensor output = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  impl::fractional_max_pool2d_out_template(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& fractional_max_pool2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices) {
  impl::fractional_max_pool2d_backward_out_template(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

Tensor fractional_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices) {
  Tensor grad_input = at::empty({0}, self.options());
  impl::fractional_max_pool2d_backward_out_template(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
