#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include "BatchKernel.h"
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"
#include "comm/ParamUtils.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"

#include <tuple>

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode) {
  std::vector<int64_t> output_size(input_size.size());

  output_size[0] = input_size[0];
  output_size[1] = input_size[1];

  for (size_t i = 2; i < input_size.size(); ++i) {
    output_size[i] = pooling_output_shape_pad_lr<int64_t>(
        input_size[i],
        kernel_size[i - 2],
        padding_l[i - 2],
        padding_r[i - 2],
        stride[i - 2],
        dilation[i - 2],
        ceil_mode);
  }

  return output_size;
}

static inline int p_start(
    int size,
    int pad,
    int kernel,
    int dilation,
    int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1))
      ? 0
      : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static inline int p_end(int size, int pad, int pooled_size, int stride) {
  return std::min((size + pad) / stride + 1, pooled_size);
}

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg.problem_) {
        int outputIndex = desc.glb_problem;
        int batch = outputIndex / stride;
        int plane, outputH, outputW;
        int64_t load_offset, store_offset;
        if constexpr (is_channels_last) {
          plane = outputIndex % numPlane;
          outputH = outputIndex / numPlane / outputSizeW % outputSizeH;
          outputW = outputIndex / numPlane % outputSizeW;
          store_offset = batch * outputSizeH * outputSizeW * numPlane + plane +
              outputH * outputSizeW * numPlane + outputW * numPlane;
        } else {
          plane = (outputIndex / outputSizeH / outputSizeW) % numPlane;
          outputH = outputIndex / outputSizeW % outputSizeH;
          outputW = outputIndex % outputSizeW;
          store_offset = batch * numPlane * outputSizeH * outputSizeW +
              plane * outputSizeH * outputSizeW + outputH * outputSizeW +
              outputW;
        }
        scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
        int maxIndex = -1;
        int StartH = outputH * dH - padH;
        int StartW = outputW * dW - padW;
        int EndH =
            Numerics<int>::min(StartH + (kH - 1) * dilationH + 1, inputSizeH);
        int EndW =
            Numerics<int>::min(StartW + (kW - 1) * dilationW + 1, inputSizeW);
        while (StartH < 0)
          StartH += dilationH;
        while (StartW < 0)
          StartW += dilationW;
#pragma unroll
        for (int h = StartH; h < EndH; h += dilationH) {
#pragma unroll
          for (int w = StartW; w < EndW; w += dilationW) {
            if constexpr (is_channels_last) {
              load_offset = batch * inputSizeH * inputSizeW * numPlane + plane +
                  h * inputSizeW * numPlane + w * numPlane;
            } else {
              load_offset = batch * numPlane * inputSizeH * inputSizeW +
                  plane * inputSizeH * inputSizeW + h * inputSizeW + w;
            }
            scalar_t val = input[load_offset];
            if (val > maxVal) {
              maxIndex = h * inputSizeW + w;
              maxVal = val;
            }
          }
        }
        indices[store_offset] = maxIndex;
        output[store_offset] = maxVal;
      }
    } while (cfg.next(item, desc));
  }
  MaxPool2dOutFrameKernelFunctor(
      scalar_t* output_,
      int64_t* indices_,
      scalar_t* input_,
      int numBatch_,
      int numPlane_,
      int inputSizeH_,
      int inputSizeW_,
      int outputSizeH_,
      int outputSizeW_,
      int kH_,
      int kW_,
      int dH_,
      int dW_,
      int padH_,
      int padW_,
      int dilationH_,
      int dilationW_,
      int outputSize_,
      int stride_,
      BatchKernelConfig cfg_)
      : output(output_),
        indices(indices_),
        input(input_),
        numBatch(numBatch_),
        numPlane(numPlane_),
        inputSizeH(inputSizeH_),
        inputSizeW(inputSizeW_),
        outputSizeH(outputSizeH_),
        outputSizeW(outputSizeW_),
        kH(kH_),
        kW(kW_),
        dH(dH_),
        dW(dW_),
        padH(padH_),
        padW(padW_),
        dilationH(dilationH_),
        dilationW(dilationW_),
        outputSize(outputSize_),
        stride(stride_),
        cfg(cfg_) {}

 private:
  scalar_t* output;
  int64_t* indices;
  scalar_t* input;
  int numBatch;
  int numPlane;
  int inputSizeH;
  int inputSizeW;
  int outputSizeH;
  int outputSizeW;
  int kH;
  int kW;
  int dH;
  int dW;
  int padH;
  int padW;
  int dilationH;
  int dilationW;
  int outputSize;
  int stride;
  BatchKernelConfig cfg;
};

template <typename scalar_t, bool is_channels_last>
void max_pool2d_out_frame(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    int numBatch,
    int numPlane,
    int inputSizeH,
    int inputSizeW,
    int outputSizeH,
    int outputSizeW,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW) {
  auto& queue = dpcppGetCurrentQueue();
  int outputSize = numBatch * numPlane * outputSizeH * outputSizeW;
  int stride = numPlane * outputSizeH * outputSizeW;
  BatchKernelConfig cfg = {
      1, outputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
  auto cgf = DPCPP_Q_CGF(cgh) {
    MaxPool2dOutFrameKernelFunctor<scalar_t, is_channels_last> kfn(
        output,
        indices,
        input,
        numBatch,
        numPlane,
        inputSizeH,
        inputSizeW,
        outputSizeH,
        outputSizeW,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        dilationH,
        dilationW,
        outputSize,
        stride,
        cfg);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dBackwardOutKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg.problem_) {
        int batch = desc.glb_problem / out_n_stride;
        int outputIndex = desc.glb_problem;
        if constexpr (is_channels_last) {
          int plane = outputIndex % numPlane;
          int64_t index = indices[outputIndex];
          int64_t gI_offset = batch * in_n_stride + plane + index * numPlane;
          atomicAdd(
              (dpcpp_global_ptr_pt<scalar_t>)&gradInput[gI_offset],
              gradOutput[outputIndex]);
        } else {
          int plane = outputIndex / out_cf_c_stride % numPlane;
          int64_t index = indices[outputIndex];
          int inputW = index % gradInputSizeW;
          int inputH = index / gradInputSizeW;
          int64_t gI_offset =
              batch * in_n_stride + plane * in_cf_c_stride + index;
          atomicAdd(
              (dpcpp_global_ptr_pt<scalar_t>)&gradInput[gI_offset],
              gradOutput[outputIndex]);
        }
      }
    } while (cfg.next(item, desc));
  }
  MaxPool2dBackwardOutKernelFunctor(
      scalar_t* gradInput_,
      scalar_t* gradOutput_,
      int64_t* indices_,
      int numPlane_,
      int gradInputSizeH_,
      int gradInputSizeW_,
      int gradOutputSizeH_,
      int gradOutputSizeW_,
      int64_t gradOutputSize_,
      int out_cf_c_stride_,
      int in_cf_c_stride_,
      int out_n_stride_,
      int in_n_stride_,
      BatchKernelConfig cfg_)
      : gradInput(gradInput_),
        gradOutput(gradOutput_),
        indices(indices_),
        numPlane(numPlane_),
        gradInputSizeH(gradInputSizeH_),
        gradInputSizeW(gradInputSizeW_),
        gradOutputSizeH(gradOutputSizeH_),
        gradOutputSizeW(gradOutputSizeW_),
        gradOutputSize(gradOutputSize_),
        out_cf_c_stride(out_cf_c_stride_),
        in_cf_c_stride(in_cf_c_stride_),
        out_n_stride(out_n_stride_),
        in_n_stride(in_n_stride_),
        cfg(cfg_) {}

 private:
  scalar_t* gradInput;
  scalar_t* gradOutput;
  int64_t* indices;
  int numPlane;
  int gradInputSizeH;
  int gradInputSizeW;
  int gradOutputSizeH;
  int gradOutputSizeW;
  int64_t gradOutputSize;
  int out_cf_c_stride;
  int in_cf_c_stride;
  int out_n_stride;
  int in_n_stride;
  BatchKernelConfig cfg;
};

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dBackwardOutDeterministicKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg.get_item_desc(item);
    do {
      if (desc.glb_problem < cfg.problem_) {
        int inputIndex = desc.glb_problem;
        int batch = inputIndex / in_n_stride;
        int plane;
        int64_t input_hw_index;
        if constexpr (is_channels_last) {
          plane = inputIndex % numPlane;
          input_hw_index = ((inputIndex % in_n_stride) - plane) / numPlane;
        } else {
          plane = inputIndex / in_cf_c_stride % numPlane;
          input_hw_index = ((inputIndex % in_n_stride)) % in_cf_c_stride;
        }
        int inputW = input_hw_index % gradInputSizeW;
        int inputH = input_hw_index / gradInputSizeW;
        int phstart = p_start(inputH, pad_h, kernel_h, dilation_h, stride_h);
        int phend = p_end(inputH, pad_h, gradOutputSizeH, stride_h);
        int pwstart = p_start(inputW, pad_w, kernel_w, dilation_w, stride_w);
        int pwend = p_end(inputW, pad_w, gradOutputSizeW, stride_w);
        if constexpr (is_channels_last) {
          int offset = batch * out_n_stride + plane;
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (indices[offset + (ph * gradOutputSizeW + pw) * numPlane] ==
                  input_hw_index) {
                gradInput[inputIndex] += static_cast<scalar_t>(
                    gradOutput
                        [offset + (ph * gradOutputSizeW + pw) * numPlane]);
              }
            }
          }
        } else {
          int offset = batch * out_n_stride + plane * out_cf_c_stride;
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (indices[offset + ph * gradOutputSizeW + pw] ==
                  input_hw_index) {
                gradInput[inputIndex] += static_cast<scalar_t>(
                    gradOutput[offset + ph * gradOutputSizeW + pw]);
              }
            }
          }
        }
      }
    } while (cfg.next(item, desc));
  }
  MaxPool2dBackwardOutDeterministicKernelFunctor(
      scalar_t* gradInput_,
      scalar_t* gradOutput_,
      int64_t* indices_,
      int numPlane_,
      int gradInputSizeH_,
      int gradInputSizeW_,
      int gradOutputSizeH_,
      int gradOutputSizeW_,
      int64_t gradInputSize_,
      int out_cf_c_stride_,
      int in_cf_c_stride_,
      int out_n_stride_,
      int in_n_stride_,
      int kernel_h_,
      int kernel_w_,
      int stride_h_,
      int stride_w_,
      int pad_h_,
      int pad_w_,
      int dilation_h_,
      int dilation_w_,
      BatchKernelConfig cfg_)
      : gradInput(gradInput_),
        gradOutput(gradOutput_),
        indices(indices_),
        numPlane(numPlane_),
        gradInputSizeH(gradInputSizeH_),
        gradInputSizeW(gradInputSizeW_),
        gradOutputSizeH(gradOutputSizeH_),
        gradOutputSizeW(gradOutputSizeW_),
        gradInputSize(gradInputSize_),
        out_cf_c_stride(out_cf_c_stride_),
        in_cf_c_stride(in_cf_c_stride_),
        out_n_stride(out_n_stride_),
        in_n_stride(in_n_stride_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        dilation_h(dilation_h_),
        dilation_w(dilation_w_),
        cfg(cfg_) {}

 private:
  scalar_t* gradInput;
  scalar_t* gradOutput;
  int64_t* indices;
  int numPlane;
  int gradInputSizeH;
  int gradInputSizeW;
  int gradOutputSizeH;
  int gradOutputSizeW;
  int64_t gradInputSize;
  int out_cf_c_stride;
  int in_cf_c_stride;
  int out_n_stride;
  int in_n_stride;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int dilation_h;
  int dilation_w;
  BatchKernelConfig cfg;
};

template <typename scalar_t, bool is_channels_last>
void max_pool2d_backward_out_frame(
    scalar_t* gradInput,
    scalar_t* gradOutput,
    int64_t* indices,
    int numBatch,
    int numPlane,
    int gradInputSizeH,
    int gradInputSizeW,
    int gradOutputSizeH,
    int gradOutputSizeW,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t gradOutputSize =
      numBatch * numPlane * gradOutputSizeH * gradOutputSizeW;
  int64_t gradInputSize = numBatch * numPlane * gradInputSizeH * gradInputSizeW;
  auto out_cf_c_stride = gradOutputSizeH * gradOutputSizeW;
  auto in_cf_c_stride = gradInputSizeH * gradInputSizeW;
  auto out_n_stride = numPlane * out_cf_c_stride;
  auto in_n_stride = numPlane * in_cf_c_stride;
  if (globalContext().deterministicAlgorithms()) {
    BatchKernelConfig cfg = {
        1, gradInputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
    auto cgf = DPCPP_Q_CGF(cgh) {
      MaxPool2dBackwardOutDeterministicKernelFunctor<scalar_t, is_channels_last>
          kfn(gradInput,
              gradOutput,
              indices,
              numPlane,
              gradInputSizeH,
              gradInputSizeW,
              gradOutputSizeH,
              gradOutputSizeW,
              gradInputSize,
              out_cf_c_stride,
              in_cf_c_stride,
              out_n_stride,
              in_n_stride,
              kernel_h,
              kernel_w,
              stride_h,
              stride_w,
              pad_h,
              pad_w,
              dilation_h,
              dilation_w,
              cfg);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf);
  } else {
    BatchKernelConfig cfg = {
        1, gradOutputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
    auto cgf = DPCPP_Q_CGF(cgh) {
      MaxPool2dBackwardOutKernelFunctor<scalar_t, is_channels_last> kfn(
          gradInput,
          gradOutput,
          indices,
          numPlane,
          gradInputSizeH,
          gradInputSizeW,
          gradOutputSizeH,
          gradOutputSizeW,
          gradOutputSize,
          out_cf_c_stride,
          in_cf_c_stride,
          out_n_stride,
          in_n_stride,
          cfg);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf);
  }
} // namespace impl

void max_pool2d_with_indices_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  const int64_t dims = 2;
  auto kernel_size_vec =
      expand_param_if_needed(kernel_size, "kernel_size", dims);
  std::vector<int64_t> empty_stride_vec = {dH, dW};
  auto stride_vec = stride.empty()
      ? empty_stride_vec
      : expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);

  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  std::vector<int64_t> output_sizes;
  int64_t outputHeight, outputWidth;

  /*
    For the pooling output shap with ceil_mode, oneDNN computes as (src -
    ker_range + pad_l + pad_r) / str + 1 and PyTorch ceil_mode comptues as (src
    - ker_range + pl + pr + stride -1) /stride +1. The code following here is to
    adjust right padding of pooling to satisfy oneDNN fromula according to
    https://jira.devtools.intel.com/browse/MFDNN-6759. The code is based on
    https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mkldnn/Pooling.cpp#L222.
  */
  if (ceil_mode) {
    // Unsqueeze 3D input(C, H, W) -> 4D input(1, C, H, W)
    auto input_4d = (input.ndimension() == 3) ? (input.unsqueeze(0)) : (input);
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input_4d.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true);

    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input_4d.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false);

      all_equal = true;
      for (size_t i = 2; i < input_4d.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
          padding_vec_r[i - 2]++;
          all_equal = false;
        }
      }
    }
    outputHeight = output_sizes[2];
    outputWidth = output_sizes[3];
  } else {
    outputHeight = pooling_output_shape<int64_t>(
        inputHeight, kH, padH, dH, dilationH, ceil_mode);
    outputWidth = pooling_output_shape<int64_t>(
        inputWidth, kW, padW, dW, dilationW, ceil_mode);
  }

  /* PyTorch support two cases of MaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */

  /* get contiguous input */
  Tensor input_ = input.ndimension() == 3
      ? input.contiguous()
      : contiguous_if_needed(input, input.suggest_memory_format());
  auto smf = input_.suggest_memory_format();

  pool2d_shape_check(
      input_,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      smf);

  /* resize output/indices */
  if (input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth}, smf);
    indices.resize_({nInputPlane, outputHeight, outputWidth}, smf);
  } else {
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
    indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
  }

  auto real_eng = choose_compute_eng(xpu::COMPUTE_ENG::BASIC, input);

  if (xpu::COMPUTE_ENG::ONEDNN == real_eng || input.is_quantized()) {
    // per oneDNN definition, no dilation means dilation ratio is 0.
    // Since dilation is already designed in the output size, no dilation
    // is used in ::xpu::oneDNN::pooling
    dilation_vec = {0, 0};

    ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_max>(
        output,
        indices,
        input_,
        nbatch,
        nInputPlane,
        0,
        inputHeight,
        inputWidth,
        0,
        outputHeight,
        outputWidth,
        stride_vec,
        kernel_size_vec,
        dilation_vec,
        padding_vec_l,
        padding_vec_r);
  } else {
    // if it's global max pooling, reuse max op for better parallelism
    if (outputHeight == 1 && outputWidth == 1 && inputHeight <= kH &&
        inputWidth <= kW) {
      if (input.ndimension() == 4) {
        input_.resize_({nbatch, nInputPlane, 1, inputHeight * inputWidth}, smf);
        output.resize_(
            {nbatch, nInputPlane, 1, outputHeight * outputWidth}, smf);
        indices.resize_(
            {nbatch, nInputPlane, 1, outputHeight * outputWidth}, smf);
      }
      at::AtenIpexTypeXPU::max_out(input_, 3, true, output, indices);
      if (input.ndimension() == 4) {
        input_.resize_({nbatch, nInputPlane, inputHeight, inputWidth}, smf);
        output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
        indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
      }
    } else if (is_smf_channels_last(input_)) {
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          input.scalar_type(),
          "max_pool2d_out_frame",
          [&] {
            max_pool2d_out_frame<scalar_t, true>(
                output.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                input_.data_ptr<scalar_t>(),
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                dilationH,
                dilationW);
          });
    } else {
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          input.scalar_type(),
          "max_pool2d_out_frame",
          [&] {
            max_pool2d_out_frame<scalar_t, false>(
                output.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                input_.data_ptr<scalar_t>(),
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                dilationH,
                dilationW);
          });
    }
  }
}

Tensor& max_pool2d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  const int64_t dims = 2;
  auto kernel_size_vec =
      expand_param_if_needed(kernel_size, "kernel_size", dims);
  std::vector<int64_t> empty_stride_vec = {dH, dW};
  auto stride_vec = stride.empty()
      ? empty_stride_vec
      : expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);

  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  std::vector<int64_t> output_sizes;
  int64_t outputHeight, outputWidth;

  // Unsqueeze 3D input(C, H, W) -> 4D input(1, C, H, W)
  auto input_4d = (input.ndimension() == 3) ? (input.unsqueeze(0)) : (input);
  /*
    For the pooling output shap with ceil_mode, oneDNN computes as (src -
    ker_range + pad_l + pad_r) / str + 1 and PyTorch ceil_mode comptues as
    (src
    - ker_range + pl + pr + stride -1) /stride +1. The code following here is
    to adjust right padding of pooling to satisfy oneDNN fromula according to
    https://jira.devtools.intel.com/browse/MFDNN-6759. The code is based on
    https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mkldnn/Pooling.cpp#L222.
  */
  if (ceil_mode) {
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input_4d.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true);

    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input_4d.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false);

      all_equal = true;
      for (size_t i = 2; i < input_4d.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
          padding_vec_r[i - 2]++;
          all_equal = false;
        }
      }
    }
    outputHeight = output_sizes[2];
    outputWidth = output_sizes[3];
  } else {
    outputHeight = pooling_output_shape<int64_t>(
        inputHeight, kH, padH, dH, dilationH, ceil_mode);
    outputWidth = pooling_output_shape<int64_t>(
        inputWidth, kW, padW, dW, dilationW, ceil_mode);
  }

  auto memory_format = input.suggest_memory_format();
  max_pool2d_backward_shape_check(
      input,
      gradOutput,
      indices,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  auto real_eng = choose_compute_eng(xpu::COMPUTE_ENG::BASIC, input);
  if ((input.is_quantized()) || real_eng == xpu::COMPUTE_ENG::ONEDNN) {
    // per oneDNN definition, no dilation means dilation ratio is 0.
    // Since dilation is already designed in the output size, no dilation
    // is used in ::xpu::oneDNN::pooling
    dilation_vec = {0, 0};
    ::xpu::oneDNN::pooling_backward<::xpu::oneDNN::alg::pooling_max>(
        gradInput,
        gradOutput,
        input,
        indices,
        nbatch,
        nInputPlane,
        0,
        inputHeight,
        inputWidth,
        0,
        outputHeight,
        outputWidth,
        stride_vec,
        kernel_size_vec,
        dilation_vec,
        padding_vec_l,
        padding_vec_r);
  } else {
    auto gradOutput_ = gradOutput.contiguous(memory_format);
    gradInput.zero_();
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gradOutput.scalar_type(),
        "max_pool2d_backward_out_frame",
        [&] {
          if (is_smf_channels_last(input)) {
            max_pool2d_backward_out_frame<scalar_t, true>(
                gradInput.data_ptr<scalar_t>(),
                gradOutput_.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                dilationH,
                dilationW);
          } else {
            max_pool2d_backward_out_frame<scalar_t, false>(
                gradInput.data_ptr<scalar_t>(),
                gradOutput_.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                dilationH,
                dilationW);
          }
        });
  }
  return gradInput;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  impl::max_pool2d_with_indices_out_template(
      output,
      indices,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& max_pool2d_with_indices_backward_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices_,
    Tensor& grad_input) {
  /* PyTorch support two cases of MaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, indices;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input.zero_();
  } else {
    auto smf = self_.suggest_memory_format();
    self = contiguous_if_needed(self_, smf);
    grad_output = contiguous_if_needed(grad_output_, smf);
    indices = contiguous_if_needed(indices_, smf);
    grad_input.zero_();
  }

  impl::max_pool2d_with_indices_backward_out_template(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
