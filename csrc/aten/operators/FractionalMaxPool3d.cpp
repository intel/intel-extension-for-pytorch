#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <runtime/DPCPPUtils.h>
#include <core/Memory.h>
#include "comm/NumericLimits.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

DPCPP_DEF_K2(FractionalMaxPool3d, typename scalar_t);
DPCPP_DEF_K2(FractionalMaxPool3dBackward, typename scalar_t);

template <typename scalar_t, typename accscalar_t>
inline int64_t get_intervals(
    accscalar_t sample,
    int64_t index,
    int64_t inputSize,
    int64_t outputSize,
    int64_t poolSize) {
  accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
      static_cast<accscalar_t>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int64_t>((index + sample) * alpha) -
        static_cast<int64_t>(sample * alpha);
  }
}

template <typename scalar_t>
void fractional_max_pool3d_out_frame(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    scalar_t* samples,
    int numBatch,
    int numPlane,
    int inputSizeT,
    int inputSizeH,
    int inputSizeW,
    int outputSizeT,
    int outputSizeH,
    int outputSizeW,
    int poolSizeT,
    int poolSizeH,
    int poolSizeW) {
  using accscalar_t = acc_type<scalar_t>;
  auto queue = dpcppGetCurrentQueue();
  int outputPlaneSize = outputSizeT * outputSizeH * outputSizeW;
  int work_group_size = outputPlaneSize > 128 ? 128 : outputPlaneSize;
  int work_group_num = (outputPlaneSize + 127) / 128;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = get_buffer<dpcpp_r_mode>(cgh, input);
    auto output_data = get_buffer<dpcpp_w_mode>(cgh, output);
    auto indices_data = get_buffer<dpcpp_w_mode>(cgh, indices);
    auto samples_data = get_buffer<dpcpp_r_mode>(cgh, samples);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item) {
      auto input_ptr = get_pointer(input_data);
      auto output_ptr = get_pointer(output_data);
      auto indices_ptr = get_pointer(indices_data);
      auto samples_ptr = get_pointer(samples_data);

      int ourOutputPoint = item.get_global_id()[0];
      int plane = item.get_group()[1];
      int batch = item.get_group()[2];

      if (ourOutputPoint < outputPlaneSize) {
        int64_t outputT = ourOutputPoint / (outputSizeH * outputSizeW);
        int64_t outputH = (ourOutputPoint / outputSizeW) % outputSizeH;
        int64_t outputW = ourOutputPoint % outputSizeW;

        int64_t poolT = get_intervals<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(
                samples_ptr
                    [batch * numPlane * 3 + plane * 3] /*[batch][plane][0]*/),
            outputT,
            inputSizeT,
            outputSizeT,
            poolSizeT);
        int64_t poolH = get_intervals<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(samples_ptr
                                         [batch * numPlane * 3 + plane * 3 +
                                          1] /*[batch][plane][1]*/),
            outputH,
            inputSizeH,
            outputSizeH,
            poolSizeH);
        int64_t poolW = get_intervals<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(samples_ptr
                                         [batch * numPlane * 3 + plane * 3 +
                                          2] /*[batch][plane][2]*/),
            outputW,
            inputSizeW,
            outputSizeW,
            poolSizeW);

        scalar_t maxVal = at::numeric_limits<scalar_t>::lowest();
        int64_t maxIndex = -1;

        for (int64_t t = poolT; t < poolT + poolSizeT; ++t) {
          for (int64_t h = poolH; h < poolH + poolSizeH; ++h) {
            if (poolSizeW < 2 || poolSizeW > 7) {
              for (int64_t w = poolW; w < poolW + poolSizeW; ++w) {
                scalar_t val = input_ptr
                    [batch * numPlane * inputSizeT * inputSizeH * inputSizeW +
                     plane * inputSizeT * inputSizeH * inputSizeW +
                     t * inputSizeH * inputSizeW + h * inputSizeW +
                     w] /*[batch][plane][t][h][w]*/;
                if (val > maxVal) {
                  maxIndex = t * inputSizeH * inputSizeW + h * inputSizeW + w;
                  maxVal = val;
                }
              }
            } else {
              for (int64_t i = 0; i < poolSizeW; ++i) {
                int64_t w = i + poolW;
                scalar_t val = input_ptr
                    [batch * numPlane * inputSizeT * inputSizeH * inputSizeW +
                     plane * inputSizeT * inputSizeH * inputSizeW +
                     t * inputSizeH * inputSizeW + h * inputSizeW +
                     w] /*[batch][plane][t][h][w]*/;
                if (val > maxVal) {
                  maxIndex = t * inputSizeH * inputSizeW + h * inputSizeW + w;
                  maxVal = val;
                }
              }
            }
          }
        }

        indices_ptr
            [batch * numPlane * outputSizeT * outputSizeH * outputSizeW +
             plane * outputSizeT * outputSizeH * outputSizeW +
             outputT * outputSizeH * outputSizeW + outputH * outputSizeW +
             outputW] /*[batch][plane][outputT][outputH][outputW]*/
            = maxIndex;
        output_ptr
            [batch * numPlane * outputSizeT * outputSizeH * outputSizeW +
             plane * outputSizeT * outputSizeH * outputSizeW +
             outputT * outputSizeH * outputSizeW + outputH * outputSizeW +
             outputW] /*[batch][plane][outputT][outputH][outputW]*/
            = maxVal;
      }
    };
    cgh.parallel_for<DPCPP_K(FractionalMaxPool3d, scalar_t)>(
        DPCPP::nd_range<3>(
            DPCPP::range<3>(
                work_group_size * work_group_num, numPlane, numBatch),
            DPCPP::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void fractional_max_pool3d_backward_out_frame(
    scalar_t* gradInput,
    scalar_t* gradOutput,
    int64_t* indices,
    int numBatch,
    int numPlane,
    int gradInputSizeT,
    int gradInputSizeH,
    int gradInputSizeW,
    int gradOutputSizeT,
    int gradOutputSizeH,
    int gradOutputSizeW) {
  auto queue = dpcppGetCurrentQueue();
  int gradOutputPlaneSize = gradOutputSizeT * gradOutputSizeH * gradOutputSizeW;
  int work_group_size = gradOutputPlaneSize > 128 ? 128 : gradOutputPlaneSize;
  int work_group_num = (gradOutputPlaneSize + 127) / 128;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = get_buffer<dpcpp_w_mode>(cgh, gradInput);
    auto gradOutput_data = get_buffer<dpcpp_r_mode>(cgh, gradOutput);
    auto indices_data = get_buffer<dpcpp_r_mode>(cgh, indices);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item) {
      auto gradInput_ptr = get_pointer(gradInput_data);
      auto gradOutput_ptr = get_pointer(gradOutput_data);
      auto indices_ptr = get_pointer(indices_data);

      int ourOutputPoint = item.get_global_id()[0];
      int plane = item.get_group()[1];
      int batch = item.get_group()[2];

      if (ourOutputPoint < gradOutputPlaneSize) {
        int64_t outputW = ourOutputPoint % gradOutputSizeW;
        int64_t outputH = (ourOutputPoint / gradOutputSizeW % gradOutputSizeH);
        int64_t outputT = ourOutputPoint / (gradOutputSizeH * gradOutputSizeW);

        int64_t index = indices_ptr
            [batch * numPlane * gradOutputSizeT * gradOutputSizeH *
                 gradOutputSizeW +
             plane * gradOutputSizeT * gradOutputSizeH * gradOutputSizeW +
             outputT * gradOutputSizeH * gradOutputSizeW +
             outputH * gradOutputSizeW +
             outputW] /*[batch][plane][outputT][outputH][outputW]*/;
        int64_t inputW = index % gradInputSizeW;
        int64_t inputH = (index / gradInputSizeW % gradInputSizeH);
        int64_t inputT = index / (gradInputSizeH * gradInputSizeW);

        atomicAdd(
          (dpcpp_global_ptr_pt<scalar_t>)&gradInput_ptr
                [batch * numPlane * gradInputSizeT * gradInputSizeH *
                     gradInputSizeW +
                 plane * gradInputSizeT * gradInputSizeH * gradInputSizeW +
                 inputT * gradInputSizeH * gradInputSizeW +
                 inputH * gradInputSizeW +
                 inputW] /*[batch][plane][inputT][inputH][inputW]*/,
            gradOutput_ptr
                [batch * numPlane * gradOutputSizeT * gradOutputSizeH *
                     gradOutputSizeW +
                 plane * gradOutputSizeT * gradOutputSizeH * gradOutputSizeW +
                 outputT * gradOutputSizeH * gradOutputSizeW +
                 outputH * gradOutputSizeW +
                 outputW] /*[batch][plane][outputT][outputH][outputW]*/
        );
      }
    };
    cgh.parallel_for<DPCPP_K(FractionalMaxPool3dBackward, scalar_t)>(
        DPCPP::nd_range<3>(
            DPCPP::range<3>(
                work_group_size * work_group_num, numPlane, numBatch),
            DPCPP::range<3>(work_group_size, 1, 1)),
        kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

void fractional_max_pool3d_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& randomSamples) {
  int64_t planeDim = 0;
  int64_t dimt = 1;
  int64_t dimh = 2;
  int64_t dimw = 3;
  int64_t numBatch = 1;

  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];
  int64_t poolSizeT = pool_size[0];
  int64_t poolSizeH = pool_size[1];
  int64_t poolSizeW = pool_size[2];

  int64_t ndims = input.ndimension();
  TORCH_CHECK(
      input.numel() != 0 && (ndims == 4 || ndims == 5),
      "fractional_max_pool3d_out_template(): ",
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ",
      ndims);

  if (ndims == 5) {
    numBatch = input.size(0);
    planeDim++;
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputT = input.size(dimt);
  int64_t inputH = input.size(dimh);
  int64_t inputW = input.size(dimw);

  TORCH_CHECK(
      outputT + poolSizeT - 1 < inputT,
      "fractional_max_pool3d_out_template(): ",
      "pool time (",
      poolSizeT,
      ") too large relative to input time (",
      inputT,
      ")");
  TORCH_CHECK(
      outputH + poolSizeH - 1 < inputH,
      "fractional_max_pool3d_out_template(): ",
      "pool height (",
      poolSizeH,
      ") too large relative to input height (",
      inputH,
      ")");
  TORCH_CHECK(
      outputW + poolSizeW - 1 < inputW,
      "fractional_max_pool3d_out_template(): ",
      "pool width (",
      poolSizeW,
      ") too large relative to input width (",
      inputW,
      ")");

  if (ndims == 4) {
    /* resize output */
    output.resize_({numPlanes, outputT, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputT, outputH, outputW});
  } else {
    /* resize output */
    output.resize_({numBatch, numPlanes, outputT, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numBatch, numPlanes, outputT, outputH, outputW});
  }

  auto output_ = output;
  auto indices_ = indices;
  auto input_ = input.contiguous();
  if (ndims == 4) {
    output_ = output_.reshape({1, numPlanes, outputT, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputT, outputH, outputW});
    input_ = input_.reshape({1, numPlanes, inputT, inputH, inputW});
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "fractional_max_pool3d_out_frame", [&] {
        fractional_max_pool3d_out_frame<scalar_t>(
            output_.data_ptr<scalar_t>(),
            indices_.data_ptr<int64_t>(),
            input_.data_ptr<scalar_t>(),
            randomSamples.data_ptr<scalar_t>(),
            input_.size(0),
            input_.size(1),
            inputT,
            inputH,
            inputW,
            outputT,
            outputH,
            outputW,
            poolSizeT,
            poolSizeH,
            poolSizeW);
      });
}

void fractional_max_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices) {
  int64_t dimt = 1;
  int64_t dimh = 2;
  int64_t dimw = 3;

  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];

  int64_t ndims = input.ndimension();
  if (ndims == 5) {
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t inputT = input.size(dimt);
  int64_t inputH = input.size(dimh);
  int64_t inputW = input.size(dimw);

  TORCH_CHECK(
      outputT == gradOutput.size(dimt),
      "fractional_max_pool3d_backward_out_template(): ",
      "gradOutput time unexpected");
  TORCH_CHECK(
      outputH == gradOutput.size(dimh),
      "fractional_max_pool3d_backward_out_template(): ",
      "gradOutput height unexpected");
  TORCH_CHECK(
      outputW == gradOutput.size(dimw),
      "fractional_max_pool3d_backward_out_template(): ",
      "gradOutput width unexpected");

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput.contiguous();
  auto indices_ = indices;

  if (ndims == 4) {
    gradInput_ =
        gradInput_.reshape({1, gradInput.size(0), inputT, inputH, inputW});
    gradOutput_ =
        gradOutput_.reshape({1, gradOutput.size(0), outputT, outputH, outputW});
    indices_ =
        indices_.reshape({1, indices.size(0), outputT, outputH, outputW});
  }

  fractional_max_pool3d_backward_out_frame<float>(
      gradInput_.data_ptr<float>(),
      gradOutput_.data_ptr<float>(),
      indices_.data_ptr<int64_t>(),
      gradInput_.size(0),
      gradInput_.size(1),
      inputT,
      inputH,
      inputW,
      outputT,
      outputH,
      outputW);
}

} // namespace impl

std::tuple<Tensor&, Tensor&> fractional_max_pool3d_out(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples) {
  impl::fractional_max_pool3d_out_template(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> fractional_max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples) {
  Tensor output = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  impl::fractional_max_pool3d_out_template(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& fractional_max_pool3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices) {
  impl::fractional_max_pool3d_backward_out_template(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

Tensor fractional_max_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices) {
  Tensor grad_input = at::empty({0}, self.options());
  impl::fractional_max_pool3d_backward_out_template(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
