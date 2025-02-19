#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <core/Memory.h>
#include <core/MemoryFormat.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"

#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include <utils/CustomOperatorRegistration.h>
#include "comm/RegisterUtils.h"
#endif

using namespace torch_ipex::xpu::dpcpp;

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

template <typename scalar_t, typename accscalar_t>
struct FractionalMaxPool2dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
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
  }
  FractionalMaxPool2dOutFrameKernelFunctor(
      scalar_t* output_data_,
      int64_t* indices_data_,
      scalar_t* input_data_,
      scalar_t* samples_data_,
      int numBatch_,
      int numPlane_,
      int inputSizeH_,
      int inputSizeW_,
      int outputSizeH_,
      int outputSizeW_,
      int poolSizeH_,
      int poolSizeW_,
      const bool is_channels_last_,
      int work_group_size_,
      int work_group_num_,
      int loops_)
      : output_data(output_data_),
        indices_data(indices_data_),
        input_data(input_data_),
        samples_data(samples_data_),
        numBatch(numBatch_),
        numPlane(numPlane_),
        inputSizeH(inputSizeH_),
        inputSizeW(inputSizeW_),
        outputSizeH(outputSizeH_),
        outputSizeW(outputSizeW_),
        poolSizeH(poolSizeH_),
        poolSizeW(poolSizeW_),
        is_channels_last(is_channels_last_),
        work_group_size(work_group_size_),
        work_group_num(work_group_num_),
        loops(loops_) {}

 private:
  scalar_t* output_data;
  int64_t* indices_data;
  scalar_t* input_data;
  scalar_t* samples_data;
  int numBatch;
  int numPlane;
  int inputSizeH;
  int inputSizeW;
  int outputSizeH;
  int outputSizeW;
  int poolSizeH;
  int poolSizeW;
  const bool is_channels_last;
  int work_group_size;
  int work_group_num;
  int loops;
};

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
    FractionalMaxPool2dOutFrameKernelFunctor<scalar_t, accscalar_t> kfn(
        output_data,
        indices_data,
        input_data,
        samples_data,
        numBatch,
        numPlane,
        inputSizeH,
        inputSizeW,
        outputSizeH,
        outputSizeW,
        poolSizeH,
        poolSizeW,
        is_channels_last,
        work_group_size,
        work_group_num,
        loops);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * work_group_num),
            sycl::range<1>(work_group_size)),
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

  if (output_.numel() == 0) {
    return;
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

} // namespace impl

#ifdef USE_OVERRIDE_OP
void fractional_max_pool2d_meta(
    const at::Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const at::Tensor& randomSamples,
    Tensor& output,
    Tensor& indices) {
  TORCH_CHECK(
      pool_size.size() == 2,
      "fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints")
  TORCH_CHECK(
      output_size.size() == 2,
      "fractional_max_pool2d: output_size must either be a single Int or tuple of Ints")
  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t heightDim = 1;
  int64_t widthDim = 2;
  int64_t outputH = output_size[0];
  int64_t outputW = output_size[1];
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  int64_t ndims = input.ndimension();
  TORCH_CHECK(
      ndims == 3 || ndims == 4,
      "fractional_max_pool2d(): Expected 3D or 4D tensor, but got: ",
      input.sizes());
  for (const auto i : c10::irange(1, ndims)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "fractional_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but got",
        input.sizes(),
        " with dimension ",
        i,
        " being empty.");
  }

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(heightDim);
  auto inputW = input.size(widthDim);

  TORCH_CHECK(
      outputH + poolSizeH - 1 <= inputH,
      "fractional_max_pool2d(): pool height ",
      poolSizeH,
      " too large relative to input height ",
      inputH);
  TORCH_CHECK(
      outputW + poolSizeW - 1 <= inputW,
      "fractional_max_pool2d(): pool width ",
      poolSizeW,
      " too large relative to input width ",
      inputW);

  if (ndims == 3) {
    if (output.defined()) {
      at::AtenIpexTypeXPU::resize_out(
          output, {numPlanes, outputH, outputW}, {}, input.options());
    } else {
      output = at::AtenIpexTypeXPU::create_out(
          {numPlanes, outputH, outputW}, {}, input.options());
    }
    /* indices will contain the locations for each output point */
    if (indices.defined()) {
      at::AtenIpexTypeXPU::resize_out(
          indices,
          {numPlanes, outputH, outputW},
          {},
          input.options().dtype(kLong));
    } else {
      indices = at::AtenIpexTypeXPU::create_out(
          {numPlanes, outputH, outputW}, {}, input.options().dtype(kLong));
    }
  } else {
    if (output.defined()) {
      at::AtenIpexTypeXPU::resize_out(
          output, {numBatch, numPlanes, outputH, outputW}, {}, input.options());
    } else {
      output = at::AtenIpexTypeXPU::create_out(
          {numBatch, numPlanes, outputH, outputW}, {}, input.options());
    }
    /* indices will contain the locations for each output point */
    if (indices.defined()) {
      at::AtenIpexTypeXPU::resize_out(
          indices,
          {numBatch, numPlanes, outputH, outputW},
          {},
          input.options().dtype(kLong));
    } else {
      indices = at::AtenIpexTypeXPU::create_out(
          {numBatch, numPlanes, outputH, outputW},
          {},
          input.options().dtype(kLong));
    }
  }
}

std::tuple<Tensor, Tensor> fractional_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples) {
  Tensor output;
  Tensor indices;
  fractional_max_pool2d_meta(
      self, kernel_size, output_size, random_samples, output, indices);
  impl::fractional_max_pool2d_out_template(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}
#endif

} // namespace AtenIpexTypeXPU
} // namespace at
#ifdef USE_OVERRIDE_OP

namespace {

::std::tuple<at::Tensor, at::Tensor> wrapper_XPU_fractional_max_pool2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef output_size,
    const at::Tensor& random_samples) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_fractional_max_pool2d", "self");
  c10::impl::check_and_update_common_device(
      common_device,
      random_samples,
      "wrapper_XPU_fractional_max_pool2d",
      "random_samples");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::fractional_max_pool2d(
      self, kernel_size, output_size, random_samples);
}

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(
      "fractional_max_pool2d", TORCH_FN((&wrapper_XPU_fractional_max_pool2d)));
}
} // namespace
#endif
