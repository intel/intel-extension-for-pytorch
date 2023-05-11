#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/Pool.h>

#include <oneDNN/oneDNN.h>
#include <quantized/QUtils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include <vector>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace {
template <
    typename scalar_t,
    typename accscalar_t,
    bool is_channels_last,
    bool is_quantized = false>
void adaptive_avg_pool2d_kernel(
    PackedTensorAccessor64<scalar_t, 4> input,
    PackedTensorAccessor64<scalar_t, 4> output,
    std::tuple<accscalar_t, int64_t> quantizer = {1, 0}) {
  int ih = input.size(2);
  int iw = input.size(3);
  int ob = output.size(0);
  int oc = output.size(1);
  int oh = output.size(2);
  int ow = output.size(3);

  accscalar_t scale = std::get<0>(quantizer);
  int64_t zp = std::get<1>(quantizer);

  int64_t numel = ob * oc * oh * ow;
  int total_item = std::min(numel, dpcppMaxWorkItemsPerTile());
  int local_range = dpcppMaxWorkItemsPerEU();
  int global_range = total_item < local_range
      ? local_range
      : (total_item / local_range) * local_range;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int64_t gi = item.get_global_linear_id();
      for (int64_t i = gi; i < numel; i += global_range) {
        int64_t _ow, _oh, _oc, _ob;
        if constexpr (is_channels_last) {
          _oc = i % oc;
          _ow = i / oc % ow;
          _oh = i / oc / ow % oh;
          _ob = i / oc / ow / oh;
        } else {
          _ow = i % ow;
          _oh = i / ow % oh;
          _oc = i / ow / oh % oc;
          _ob = i / ow / oh / oc;
        }

        int64_t _ih0 = native::start_index(_oh, oh, ih);
        int64_t _ih1 = native::end_index(_oh, oh, ih);
        int64_t _iw0 = native::start_index(_ow, ow, iw);
        int64_t _iw1 = native::end_index(_ow, ow, iw);
        int64_t kh = _ih1 - _ih0;
        int64_t kw = _iw1 - _iw0;
        int64_t _ib = _ob;
        int64_t _ic = _oc;

        accscalar_t sum = 0;
        for (int _ih = _ih0; _ih < _ih1; _ih++) {
          for (int _iw = _iw0; _iw < _iw1; _iw++) {
            if constexpr (is_quantized) {
              sum += accscalar_t(
                  ((accscalar_t)input[_ib][_ic][_ih][_iw] - (accscalar_t)zp) *
                  scale);
            } else {
              sum += accscalar_t(input[_ib][_ic][_ih][_iw]);
            }
          }
        }
        accscalar_t avg = sum / kh / kw;

        const auto store = [](PackedTensorAccessor64<scalar_t, 4> oacc,
                              int64_t _ob,
                              int64_t _oc,
                              int64_t _oh,
                              int64_t _ow,
                              scalar_t res) { oacc[_ob][_oc][_oh][_ow] = res; };
        if constexpr (is_quantized) {
          scalar_t qavg = quantize_val<scalar_t>(scale, zp, avg);
          store(output, _ob, _oc, _oh, _ow, qavg);
        } else {
          store(output, _ob, _oc, _oh, _ow, avg);
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

void adaptive_avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_average_pool2d_dpcpp(): expected input to have non-empty spatial "
        "dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_average_pool2d: internal error: output_size.size() must be 2");

  auto outputWidth = output_size[1];
  auto outputHeight = output_size[0];

  if (!input.is_quantized() && outputWidth == 1 && outputHeight == 1) {
    // in this case, adaptive pooling is just computing mean over hw
    // dimensions, which can be done more efficiently

    output = input.mean({-1, -2}, /* keepdim = */ true);
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
      // assert ndim == 4, since ndim = 3 doesn't give channels_last
      const int n = input.size(0);
      const int c = input.size(1);
      output.as_strided_({n, c, 1, 1}, {c, 1, c, c});
    }
    return;
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> stride_vec = {dH, dW};

  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> kernel_size_vec = {kH, kW};

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;
  std::vector<int64_t> padding_vec = {padH, padW};

  Tensor input_;
  if (input.ndimension() == 3) {
    input_ = input.contiguous();
    output.resize_({nInputPlane, outputHeight, outputWidth});
  } else {
    auto smf = input.suggest_memory_format();
    input_ = contiguous_if_needed(input, smf);
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
  }

  if (xpu::oneDNN::is_valid_pooling(
          {inputHeight, inputWidth},
          {outputHeight, outputWidth},
          {kH, kW},
          {dH, dW},
          {padH, padW})) {
    /* PyTorch support two cases of AdaptiveAvgPool2d:
       1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
       This case does not support channel last format. For a 3-dim tensor,
       the suggest_memory_format can only be Contiguous or ChannelsLast1D
       (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
       H, W) case. Then the suggest_memory_format can only be Contiguous.
       2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
       This case supports Contiguous and ChannelsLast2D memory_format. */
    xpu::oneDNN::pooling<xpu::oneDNN::alg::pooling_avg_exclude_padding>(
        output,
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
        padding_vec,
        padding_vec);
  } else {
    TORCH_CHECK(
        !is_opaque_u8(input),
        "XPU opaque u8 tensor is not supported in SYCL kernel ...");

    input_ = to_plain_if_needed(input_);

    bool is_3d = input_.ndimension() == 3;
    if (is_3d) {
      input_.resize_({1, nInputPlane, inputHeight, inputWidth});
      output.resize_({1, nInputPlane, outputHeight, outputWidth});
    }

    if (input_.is_quantized()) {
      float scale = input.scalar_type() == kQUInt8 ? input.q_scale() / 2.0f
                                                   : input.q_scale();

      IPEX_DISPATCH_QTYPE_ONLY_WITH_UNDERLYING(
          input_.scalar_type(), "aten::adpative_avg_pooled", 0, [&]() {
            auto iacc = input_.packed_accessor64<scalar_t_0, 4>();
            auto oacc = output.packed_accessor64<scalar_t_0, 4>();
            if (is_smf_channels_last(output)) {
              adaptive_avg_pool2d_kernel<scalar_t_0, float, true, true>(
                  iacc, oacc, {scale, 0 /* TODO: Asymm */});
            } else {
              adaptive_avg_pool2d_kernel<scalar_t_0, float, false, true>(
                  iacc, oacc, {scale, 0 /* TODO: Asymm */});
            }
          });
    } else {
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          input_.scalar_type(),
          "aten::adaptive_avg_pool2d",
          [&]() {
            using accscalar_t = acc_type<scalar_t>;
            auto iacc = input_.packed_accessor64<scalar_t, 4>();
            auto oacc = output.packed_accessor64<scalar_t, 4>();
            if (is_smf_channels_last(output)) {
              adaptive_avg_pool2d_kernel<scalar_t, accscalar_t, true>(
                  iacc, oacc);
            } else {
              adaptive_avg_pool2d_kernel<scalar_t, accscalar_t, false>(
                  iacc, oacc);
            }
          });
    }

    if (is_3d) {
      input_.resize_({nInputPlane, inputHeight, inputWidth});
      output.resize_({nInputPlane, outputHeight, outputWidth});
    }
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    bool is_channels_last,
    bool using_shared>
class adaptive_avg_pool2d_backward_kernel {
 public:
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc) {}
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
class adaptive_avg_pool2d_backward_kernel<
    scalar_t,
    accscalar_t,
    is_channels_last,
    false> {
 public:
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc) {
    int ib = gxacc.size(0);
    int ic = gxacc.size(1);
    int ih = gxacc.size(2);
    int iw = gxacc.size(3);
    int oh = gyacc.size(2);
    int ow = gyacc.size(3);

    int64_t numel = ib * ic * ih * iw;
    int total_item = std::min(numel, dpcppMaxWorkItemsPerTile());
    int local_range = dpcppMaxWorkItemsPerEU();
    int global_range = total_item < local_range
        ? local_range
        : (total_item / local_range) * local_range;

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        int64_t gi = item.get_global_linear_id();
        int64_t li = item.get_local_id(0);

        for (int64_t i = gi; i < numel; i += global_range) {
          int64_t _iw, _ih, _ic, _ib;
          if constexpr (is_channels_last) {
            _ic = i % ic;
            _iw = i / ic % iw;
            _ih = i / ic / iw % ih;
            _ib = i / ic / iw / ih;
          } else {
            _iw = i % iw;
            _ih = i / iw % ih;
            _ic = i / iw / ih % ic;
            _ib = i / iw / ih / ic;
          }

          int64_t _oh0 = native::start_index(_ih, ih, oh);
          int64_t _oh1 = native::end_index(_ih, ih, oh);
          int64_t _ow0 = native::start_index(_iw, iw, ow);
          int64_t _ow1 = native::end_index(_iw, iw, ow);
          int64_t _ob = _ib;
          int64_t _oc = _ic;

          accscalar_t gx = 0;
          accscalar_t _ikh, _ikw;
          for (int _oh = _oh0; _oh < _oh1; _oh++) {
            _ikh = accscalar_t(1.0) /
                (accscalar_t)(native::end_index(_oh, oh, ih) - native::start_index(_oh, oh, ih));
            for (int _ow = _ow0; _ow < _ow1; _ow++) {
              _ikw = accscalar_t(1.0) /
                  (accscalar_t)(native::end_index(_ow, ow, iw) - native::start_index(_ow, ow, iw));
              gx += gyacc[_ob][_oc][_oh][_ow] * _ikh * _ikw;
            }
          }

          const auto store = [](PackedTensorAccessor64<scalar_t, 4> gxacc,
                                int64_t _ib,
                                int64_t _ic,
                                int64_t _ih,
                                int64_t _iw,
                                scalar_t res) {
            gxacc[_ib][_ic][_ih][_iw] = res;
          };
          store(gxacc, _ib, _ic, _ih, _iw, (scalar_t)gx);
        }
      };
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(global_range), sycl::range<1>(local_range)),
          kfn);
    };

    DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
  }
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
class adaptive_avg_pool2d_backward_kernel<
    scalar_t,
    accscalar_t,
    is_channels_last,
    true> {
 public:
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc) {
    int ib = gxacc.size(0);
    int ic = gxacc.size(1);
    int ih = gxacc.size(2);
    int iw = gxacc.size(3);
    int oh = gyacc.size(2);
    int ow = gyacc.size(3);

    int64_t numel = ib * ic * ih * iw;
    int total_item = std::min(numel, dpcppMaxWorkItemsPerTile());

    // Not use dpcppMaxWorkItemsPerEU to improve shared local memory usage.
    // Size of local memory is fixed (ih/iw/oh/ow) in the case.
    // Using max work group size to make more work items share same local
    // memory.
    int local_range = dpcppMaxWorkGroupSize();
    int global_range = total_item < local_range
        ? local_range
        : (total_item / local_range) * local_range;

    // trade-off occupancy and slm leverage
    int64_t ohw01_shared_size = ((iw + ih) * 2) * sizeof(int);
    int64_t ikhw_shared_size = (oh + ow) * sizeof(accscalar_t);

    auto cgf = DPCPP_Q_CGF(cgh) {
      dpcpp_local_acc_t<int> _oh0_cached(ih * sizeof(int), cgh);
      dpcpp_local_acc_t<int> _oh1_cached(ih * sizeof(int), cgh);
      dpcpp_local_acc_t<int> _ow0_cached(iw * sizeof(int), cgh);
      dpcpp_local_acc_t<int> _ow1_cached(iw * sizeof(int), cgh);
      dpcpp_local_acc_t<accscalar_t> _ikh_cached(oh * sizeof(accscalar_t), cgh);
      dpcpp_local_acc_t<accscalar_t> _ikw_cached(ow * sizeof(accscalar_t), cgh);
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        int64_t gi = item.get_global_linear_id();
        int64_t li = item.get_local_id(0);

        // for-loop order: oh*ow->ih->iw
        // reuse oh*ow(oh0, oh1, ow0, ow1), ih(ikh), iw(ikw) in inner loop.
        for (int _ih = li; _ih < ih; _ih += local_range) {
          _oh0_cached[_ih] = (int)native::start_index(_ih, ih, oh);
          _oh1_cached[_ih] = (int)native::end_index(_ih, ih, oh);
        }
        for (int _iw = li; _iw < iw; _iw += local_range) {
          _ow0_cached[_iw] = (int)native::start_index(_iw, iw, ow);
          _ow1_cached[_iw] = (int)native::end_index(_iw, iw, ow);
        }
        for (int _oh = li; _oh < oh; _oh += local_range) {
          _ikh_cached[_oh] = accscalar_t(1.0) /
              (accscalar_t)(native::end_index(_oh, oh, ih) -
                            native::start_index(_oh, oh, ih));
        }
        for (int _ow = li; _ow < ow; _ow += local_range) {
          _ikw_cached[_ow] = accscalar_t(1.0) /
              (accscalar_t)(native::end_index(_ow, ow, iw) -
                            native::start_index(_ow, ow, iw));
        }

        item.barrier(dpcpp_local_fence);

        for (int64_t i = gi; i < numel; i += global_range) {
          int64_t _iw, _ih, _ic, _ib;
          if constexpr (is_channels_last) {
            _ic = i % ic;
            _iw = i / ic % iw;
            _ih = i / ic / iw % ih;
            _ib = i / ic / iw / ih;
          } else {
            _iw = i % iw;
            _ih = i / iw % ih;
            _ic = i / iw / ih % ic;
            _ib = i / iw / ih / ic;
          }

          int64_t _oh0, _oh1, _ow0, _ow1;
          _oh0 = _oh0_cached[_ih];
          _oh1 = _oh1_cached[_ih];
          _ow0 = _ow0_cached[_iw];
          _ow1 = _ow1_cached[_iw];
          int64_t _ob = _ib;
          int64_t _oc = _ic;

          accscalar_t gx = 0;
          accscalar_t _ikh, _ikw;
          for (int _oh = _oh0; _oh < _oh1; _oh++) {
            _ikh = _ikh_cached[_oh];
            for (int _ow = _ow0; _ow < _ow1; _ow++) {
              _ikw = _ikw_cached[_ow];
              gx += gyacc[_ob][_oc][_oh][_ow] * _ikh * _ikw;
            }
          }

          const auto store = [](PackedTensorAccessor64<scalar_t, 4> gxacc,
                                int64_t _ib,
                                int64_t _ic,
                                int64_t _ih,
                                int64_t _iw,
                                scalar_t res) {
            gxacc[_ib][_ic][_ih][_iw] = res;
          };
          store(gxacc, _ib, _ic, _ih, _iw, (scalar_t)gx);
        }
      };
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(global_range), sycl::range<1>(local_range)),
          kfn);
    };

    DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
  }
};

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input) {
  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  auto outputHeight = gradOutput.size(-2);
  auto outputWidth = gradOutput.size(-1);

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> stride_vec = {dH, dW};

  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      std::floor((float)inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      std::floor((float)inputWidth / outputWidth);
  std::vector<int64_t> kernel_size_vec = {kH, kW};

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;
  std::vector<int64_t> padding_vec = {padH, padW};

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};
  if (xpu::oneDNN::is_valid_pooling(
          {inputHeight, inputWidth},
          {outputHeight, inputHeight},
          {kH, kW},
          {dH, dW},
          {padH, padW})) {
    xpu::oneDNN::pooling_backward<
        xpu::oneDNN::alg::pooling_avg_exclude_padding>(
        gradInput,
        gradOutput,
        input,
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
        padding_vec,
        padding_vec);
  } else {
    auto gradOutput_ = to_plain_if_needed(gradOutput);

    bool is_3d = gradOutput_.ndimension() == 3;
    if (is_3d) {
      gradOutput_.resize_({1, nInputPlane, outputHeight, outputWidth});
      gradInput.resize_({1, nInputPlane, inputHeight, inputWidth});
    }

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        gradOutput_.scalar_type(),
        "aten::adaptive_avg_pool2d_backward",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          auto gyacc = gradOutput_.packed_accessor64<scalar_t, 4>();
          auto gxacc = gradInput.packed_accessor64<scalar_t, 4>();

          int64_t ohw01_shared_size =
              ((inputHeight + inputWidth) * 2) * sizeof(int);
          int64_t ikhw_shared_size =
              (outputHeight + outputWidth) * sizeof(accscalar_t);
          bool using_shared =
              dpcppLocalMemSize() >= ohw01_shared_size + ikhw_shared_size;

          if (is_smf_channels_last(gradOutput)) {
            if (using_shared) {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  true,
                  true>()(gyacc, gxacc);
            } else {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  true,
                  false>()(gyacc, gxacc);
            }
          } else {
            if (using_shared) {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  false,
                  true>()(gyacc, gxacc);
            } else {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  false,
                  false>()(gyacc, gxacc);
            }
          }
        });

    if (is_3d) {
      gradOutput_.resize_({nInputPlane, outputHeight, outputWidth});
      gradInput.resize_({nInputPlane, inputHeight, inputWidth});
    }
  }
}
} // namespace

namespace at {
namespace AtenIpexTypeXPU {

Tensor& adaptive_avg_pool2d_out(
    const Tensor& self,
    IntArrayRef output_size,
    Tensor& out) {
  adaptive_avg_pool2d_out_template(out, self, output_size);
  return out;
}

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if (self.is_quantized()) {
    output = at::_empty_affine_quantized(
        {0}, self.options(), self.q_scale(), self.q_zero_point());
  } else {
    output = at::empty({0}, self.options());
  }

  adaptive_avg_pool2d_out_template(output, self, output_size);
  return output;
}

Tensor adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  if (self.is_quantized()) {
    output = at::_empty_affine_quantized(
        {0}, self.options(), self.q_scale(), self.q_zero_point());
  } else {
    output = at::empty({0}, self.options());
  }

  adaptive_avg_pool2d_out_template(output, self, output_size);
  return output;
}

Tensor _adaptive_avg_pool2d_backward(
    const Tensor& grad_output_,
    const Tensor& self_) {
  /* PyTorch support two cases of AdaptiveAvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the
     sementics of Input (C, H, W) case. Then the suggest_memory_format can
     only be Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, grad_input;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input = at::empty_like(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = contiguous_if_needed(self_, smf);
    grad_output = contiguous_if_needed(grad_output_, smf);
    grad_input = at::empty_like(self_, smf);
  }

  adaptive_avg_pool2d_backward_out_template(grad_input, grad_output, self);
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor _adaptive_avg_pool2d(const Tensor& self, IntArrayRef output_size) {
  Tensor output;
  output = at::_empty_affine_quantized(
      {0},
      self.options(),
      self.q_scale(),
      self.q_zero_point(),
      MemoryFormat::Contiguous);
  adaptive_avg_pool2d_out_template(output, self, output_size);
  return output;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
