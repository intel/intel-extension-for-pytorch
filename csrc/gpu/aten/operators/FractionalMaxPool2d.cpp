#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <core/Memory.h>
#include <core/MemoryFormat.h>
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
    int poolSizeW,
    const bool is_channels_last) {
  using accscalar_t = acc_type<scalar_t>;
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int outputSize = numBatch * numPlane * outputSizeH * outputSizeW;
  int work_group_size = outputSize > max_wg_size ? max_wg_size : outputSize;
  // One full device launch could launch en_num * SMID32 * HD threads as below
  const auto target_global_size = dpcppMaxWorkItemsPerTile(dev_id);
  // Each work group size is work_group_size, one full device launch is
  // target_global_size, so we can calculate max work group num as below
  const int max_work_group_num = target_global_size / work_group_size;
  int work_group_num = outputSize / work_group_size < max_work_group_num
      ? outputSize / work_group_size
      : max_work_group_num;
  int draft_work_group_num =
      (outputSize + work_group_size - 1) / work_group_size;
  // work item in each work group calculates loops' elements
  int loops = draft_work_group_num / work_group_num + 1;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input;
    auto output_data = output;
    auto indices_data = indices;
    auto samples_data = samples;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto input_ptr = input_data;
      auto output_ptr = output_data;
      auto indices_ptr = indices_data;
      auto samples_ptr = samples_data;

      int linearIndex = item.get_global_id()[0];
      for (int l = 0; l < loops; ++l) {
        int outputIndex = linearIndex + l * (work_group_size * work_group_num);
        int batch = outputIndex / (numPlane * outputSizeH * outputSizeW);
        int plane = is_channels_last
            ? outputIndex % numPlane
            : (outputIndex / outputSizeH / outputSizeW) % numPlane;
        int outputH = is_channels_last
            ? outputIndex / numPlane / outputSizeW % outputSizeH
            : outputIndex / outputSizeW % outputSizeH;
        int outputW = is_channels_last ? outputIndex / numPlane % outputSizeW
                                       : outputIndex % outputSizeW;

        if (batch < numBatch && plane < numPlane && outputH < outputSizeH &&
            outputW < outputSizeW) {
          int poolW = get_interval<scalar_t, accscalar_t>(
              static_cast<accscalar_t>(samples_ptr
                                           [batch * numPlane * 2 +
                                            plane * 2] /*[batch][plane][0] */),
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
            for (int w = poolW; w < poolW + poolSizeW; ++w) {
              int64_t load_offset = is_channels_last
                  ? batch * inputSizeH * inputSizeW * numPlane + plane +
                      h * inputSizeW * numPlane + w * numPlane
                  : batch * numPlane * inputSizeH * inputSizeW +
                      plane * inputSizeH * inputSizeW + h * inputSizeW + w;
              scalar_t val = input_ptr[load_offset];
              if (val > maxVal) {
                maxIndex = h * inputSizeW + w;
                maxVal = val;
              }
            }
          }

          int64_t store_offset = is_channels_last
              ? batch * outputSizeH * outputSizeW * numPlane + plane +
                  outputH * outputSizeW * numPlane + outputW * numPlane
              : batch * numPlane * outputSizeH * outputSizeW +
                  plane * outputSizeH * outputSizeW + outputH * outputSizeW +
                  outputW;
          indices_ptr[store_offset] = maxIndex;
          output_ptr[store_offset] = maxVal;
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * work_group_num),
            sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, bool is_channels_last>
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
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkItemsPerEU(dev_id);
  int64_t gradOutputSize =
      numBatch * numPlane * gradOutputSizeH * gradOutputSizeW;
  int work_group_size =
      gradOutputSize > max_wg_size ? max_wg_size : gradOutputSize;
  int global_range =
      ((gradOutputSize - 1) / work_group_size + 1) * work_group_size;
  auto out_cf_c_stride = gradOutputSizeH * gradOutputSizeW;
  auto in_cf_c_stride = gradInputSizeH * gradInputSizeW;
  auto out_n_stride = numPlane * out_cf_c_stride;
  auto in_n_stride = numPlane * in_cf_c_stride;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = gradInput;
    auto gradOutput_data = gradOutput;
    auto indices_data = indices;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto gradInput_ptr = gradInput_data;
      auto gradOutput_ptr = gradOutput_data;
      auto indices_ptr = indices_data;

      int64_t outputIndex = item.get_global_id()[0];
      if (outputIndex < gradOutputSize) {
        int batch = outputIndex / out_n_stride;
        if constexpr (is_channels_last) {
          int plane = outputIndex % numPlane;
          int64_t index = indices_ptr[outputIndex];
          int64_t gI_offset = batch * in_n_stride + plane + index * numPlane;
          atomicAdd(
              (dpcpp_global_ptr_pt<scalar_t>)&gradInput_ptr[gI_offset],
              gradOutput_ptr[outputIndex]);
        } else {
          int plane = outputIndex / out_cf_c_stride % numPlane;
          int64_t index = indices_ptr[outputIndex];
          int inputW = index % gradInputSizeW;
          int inputH = index / gradInputSizeW;
          int64_t gI_offset =
              batch * in_n_stride + plane * in_cf_c_stride + index;
          atomicAdd(
              (dpcpp_global_ptr_pt<scalar_t>)&gradInput_ptr[gI_offset],
              gradOutput_ptr[outputIndex]);
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
} // namespace impl

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

  auto smf = (3 == ndims) ? at::MemoryFormat::Contiguous
                          : input.suggest_memory_format();

  if (ndims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW}, smf);
    indices.resize_({numBatch, numPlanes, outputH, outputW}, smf);
  }

  auto output_ = output;
  auto input_ = input.contiguous(smf);
  auto indices_ = indices;

  if (ndims == 3) {
    output_ = output_.reshape({1, numPlanes, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputH, outputW});
    input_ = input_.reshape({1, input.size(0), input.size(1), input.size(2)});
  }
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fractional_max_pool2d_out_frame",
      [&] {
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
            poolSizeW,
            is_smf_channels_last(input));
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

  auto smf = (3 == ndims) ? at::MemoryFormat::Contiguous
                          : input.suggest_memory_format();

  /* resize */
  gradInput.resize_as_(input, smf);
  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput.contiguous(smf);
  auto indices_ = indices;
  if (ndims == 3) {
    gradInput_ = gradInput_.reshape({1, input.size(0), inputH, inputW});
    gradOutput_ =
        gradOutput_.reshape({1, gradOutput.size(0), outputH, outputW});
    indices_ = indices_.reshape({1, indices_.size(0), outputH, outputW});
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "fractional_max_pool2d_backward_out_frame",
      [&] {
        if (is_smf_channels_last(input)) {
          fractional_max_pool2d_backward_out_frame<scalar_t, true>(
              gradInput_.data_ptr<scalar_t>(),
              gradOutput_.data_ptr<scalar_t>(),
              indices_.data_ptr<int64_t>(),
              gradInput_.size(0),
              gradInput_.size(1),
              inputH,
              inputW,
              outputH,
              outputW);
        } else {
          fractional_max_pool2d_backward_out_frame<scalar_t, false>(
              gradInput_.data_ptr<scalar_t>(),
              gradOutput_.data_ptr<scalar_t>(),
              indices_.data_ptr<int64_t>(),
              gradInput_.size(0),
              gradInput_.size(1),
              inputH,
              inputW,
              outputH,
              outputW);
        }
      });
}

} // namespace impl

std::tuple<Tensor&, Tensor&> fractional_max_pool2d_out(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples,
    Tensor& output,
    Tensor& indices) {
  impl::fractional_max_pool2d_out_template(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& fractional_max_pool2d_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices,
    Tensor& grad_input) {
  impl::fractional_max_pool2d_backward_out_template(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
