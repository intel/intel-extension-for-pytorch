#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/custom_class.h>
#include <torch/custom_class_detail.h>
#include <utils/LRUCache.h>

#include <torch/custom_class.h>
#include <utils/Macros.h>

#include <aten/core/DeviceAllocator.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <operators/comm/Numerics.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/DPCPP.h>

namespace xpu {
namespace dpcpp {
// Note: [Opaque u8 tensor]
// Due to the difference between oneDNN and PyTorch u8 quantization, we quant
// tensor with kQUint8 and 128 zp to memory::data_type::s8 and 0 zp inside. This
// utils is used for checking this kind of QTensor. More details can see
// Quantization.cpp quantizer_tensor_per_tenser_affine function.
static inline bool is_opaque_u8(const Tensor& qx) {
  auto qx_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(qx);
  if (!qx_ctx.is_plain()) {
    return (
        (qx.scalar_type() == kQUInt8) &&
        (qx_ctx.meta().get_data_type() == dnnl::memory::data_type::s8));
  } else {
    return false;
  }
}

template <typename T>
inline T Round(const T x) {
  return std::nearbyint(x);
}

static inline at::Tensor u8tos8(const at::Tensor& u8) {
  auto s8 = at::_empty_affine_quantized(
      u8.sizes(),
      ScalarType::QInt8,
      c10::nullopt,
      u8.device(),
      c10::nullopt,
      u8.q_scale(),
      0,
      u8.suggest_memory_format());

  auto& dpcpp_queue = xpu::dpcpp::dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    uint8_t* u8_ptr = (uint8_t*)u8.data_ptr();
    int8_t* s8_ptr = (int8_t*)s8.data_ptr();
    cgh.parallel_for(sycl::range<1>(u8.numel()), [=](sycl::item<1> item) {
      auto id = item.get_linear_id();
      auto s8_val = (float)Round(static_cast<float>(u8_ptr[id]) / 2.f);
      s8_val =
          Numerics<float>::min(Numerics<float>::max(s8_val, -128.f), 127.f);
      s8_ptr[id] = static_cast<int8_t>(s8_val);
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  return s8;
}

template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  int64_t qvalue;
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();

  qvalue = static_cast<int64_t>(zero_point + Round(value / scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

template <typename T>
T quantize_val(float scale, int64_t zero_point, float value) {
  float qvalue;
  constexpr float qmin = (float)std::numeric_limits<T>::min();
  constexpr float qmax = (float)std::numeric_limits<T>::max();

  qvalue = (float)zero_point + Round(value / scale);
  qvalue = std::max<float>(qvalue, qmin);
  qvalue = std::min<float>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

/*
  Note: [Usage on XPUQUantizerBase]
  XPUQuantizerBase is a class designed to hold scale/zeropoint on XPU for
  QTensor. It is intended to co-work with LRU during the whole lifetime of
  process. The deconstructor is not needed for this situation.

  Likewise, if you want to hold XPUQuantBase in a variable with
  procedure-level lifeteime, please use `need_dtor=false`, otherwise, use
  default template argument as follows.
*/
template <typename scale_t_, typename zp_t_, bool need_dtor = true>
class XPUQuantizerBase {
 public:
  using scale_t = scale_t_;
  using zp_t = zp_t_;
  using scale_ptr_t = std::shared_ptr<scale_t>;
  using zp_ptr_t = std::shared_ptr<zp_t>;

 public:
  XPUQuantizerBase() = default;
  XPUQuantizerBase(size_t size, sycl::queue& q) {
    void* sc_ptr =
        DeviceAllocator::Instance()->raw_allocate(size * sizeof(scale_t));
    void* zp_ptr =
        DeviceAllocator::Instance()->raw_allocate(size * sizeof(zp_t));
    void (*deleter)(void*) = nullptr;
    if (need_dtor) {
      scale_ptr_ = scale_ptr_t((scale_t*)sc_ptr, DeviceAllocator::deleter);
      zp_ptr_ = zp_ptr_t((zp_t*)zp_ptr, DeviceAllocator::deleter);
    } else {
      scale_ptr_ =
          scale_ptr_t((scale_t*)sc_ptr, [=](scale_t*) { /*Do Nothing*/ });
      zp_ptr_ = zp_ptr_t((zp_t*)zp_ptr, [=](zp_t*) { /*Do Nothing*/ });
    }
  }

  scale_t* scale_ptr() {
    return scale_ptr_.get();
  }

  zp_t* zero_point_ptr() {
    return zp_ptr_.get();
  }

 private:
  scale_ptr_t scale_ptr_;
  zp_ptr_t zp_ptr_;
};

static inline XPUQuantizerBase<float, int32_t, false>
fetch_cached_quantizer_base(float dnn_sc, int32_t dnn_zp) {
  using key_t = xpu::dpcpp::lru_key_t;
  key_t key_sc_zp;

  // 0 here means dnn zero_point
  // TODO: remove after asymmetric is enabled
  dnn_zp = 0;
  xpu::dpcpp::create_key(key_sc_zp, dnn_sc, dnn_zp);

  bool sc_zp_key_found = xpu::dpcpp::find_key<
      XPUQuantizerBase<float, int32_t, false>,
      xpu::dpcpp::lru_key_t,
      /*capacity*/ 256>(key_sc_zp);

  XPUQuantizerBase<float, int32_t, false> quant_base;
  float* sc_ptr;
  int32_t* zp_ptr;
  if (sc_zp_key_found) {
    quant_base = xpu::dpcpp::fetch_m<
        XPUQuantizerBase<float, int32_t, false>,
        key_t,
        /*capacity*/ 256>(key_sc_zp);
    sc_ptr = quant_base.scale_ptr();
    zp_ptr = quant_base.zero_point_ptr();
  } else {
    quant_base = xpu::dpcpp::create_and_fetch_m<
        XPUQuantizerBase<float, int32_t, false>,
        key_t,
        /*capacity*/ 256>(key_sc_zp, /*size*/ 1, dpcppGetCurrentQueue());
    sc_ptr = quant_base.scale_ptr();
    float _scale = (float)dnn_sc;
    dpcppGetCurrentQueue().single_task([=]() { sc_ptr[0] = _scale; });

    zp_ptr = quant_base.zero_point_ptr();
    int32_t _zp = 0;
    dpcppGetCurrentQueue().single_task([=]() { zp_ptr[0] = _zp; });
  }
  return quant_base;
}

static inline std::pair<float*, int32_t*> q_get_sc_zp_gpu_ptr(const Tensor qx) {
  float dnn_scale;
  if (is_opaque_u8(qx)) {
    dnn_scale = qx.q_scale();
  } else {
    dnn_scale = (qx.scalar_type() == kQUInt8) ? qx.q_scale() / 2 : qx.q_scale();
  }
  // TODO: Use correct zp after aymmetric is enabled
  auto quant_base = fetch_cached_quantizer_base(dnn_scale, /*dnn_zp=*/0);

  return {quant_base.scale_ptr(), quant_base.zero_point_ptr()};
}
} // namespace dpcpp
} // namespace xpu
namespace at {
namespace AtenIpexTypeQuantizedXPU {
template <int kSpatialDim>
struct PackedConvWeightQDPCPP : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightQDPCPP(
      Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose) {}

  Tensor weight;
  c10::optional<Tensor> bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    // This is just align with Pytorch INT8 designe.
    Tensor output;
    return output;
  }

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    // This is just align with Pytorch INT8 designe.
    Tensor output;
    return output;
  }

  at::Tensor apply_dynamic(const at::Tensor& input, bool reduce_range)
      override {
    Tensor output;
    return output;
  }

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override {
    return std::tuple<at::Tensor, c10::optional<at::Tensor>>(weight, bias);
  }

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  bool transpose() const override {
    return transpose_;
  }

  int64_t groups() const override {
    return groups_;
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);
};

struct PackedLinearWeightQDPCPP : public LinearPackedParamsBase {
  PackedLinearWeightQDPCPP(Tensor weight, c10::optional<at::Tensor> bias)
      : weight(std::move(weight)), bias_(std::move(bias)) {}
  Tensor weight;
  c10::optional<at::Tensor> bias_;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override {
    // This is just align with Pytorch INT8 designe.
    Tensor output;
    return output;
  }
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override {
    // This is just align with Pytorch INT8 designe.
    Tensor output;
    return output;
  }

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override {
    Tensor output;
    return output;
  }
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override {
    Tensor output;
    return output;
  }

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override {
    return std::tuple<at::Tensor, c10::optional<at::Tensor>>(weight, bias_);
  };

  c10::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);
};

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at

#ifdef BUILD_JIT_QUANTIZATION_SAVE

// Repeat torch type definition here again
using ConvParamsSerializationTypeV2 = std::tuple<
    // version, for versions 2 and up
    std::string,
    // non-optional tensors
    std::vector<at::Tensor>,
    // optional tensors
    std::vector<c10::optional<at::Tensor>>>;
using ConvParamsSerializationTypeV3 = std::tuple<
    // version, int for versions 3 and up
    int64_t,
    // configuration values
    std::vector<int64_t>,
    // optional tensors
    std::vector<c10::optional<at::Tensor>>>;

using ConvParamsSerializationType = ConvParamsSerializationTypeV2;

template <uint32_t kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> deserialize_conv_dpcpp(
    ConvParamsSerializationTypeV3 state) {
  int64_t version;
  std::vector<int64_t> config_vals;
  std::vector<c10::optional<at::Tensor>> tensors;

  std::tie(version, config_vals, tensors) = state;
  TORCH_INTERNAL_ASSERT(
      version == 3, "Unexpected serialized qconv version: ", version);

  TORCH_CHECK(tensors.size() == 3, "Wrong number of tensors", tensors.size());
  c10::optional<at::Tensor> weight = tensors[1];
  c10::optional<at::Tensor> bias = tensors[2];
  TORCH_INTERNAL_ASSERT(
      weight, "Weight should always be present in serialized qconv.");

  torch::List<int64_t> stride, padding, output_padding, dilation;
  // skip kSpatialDim
  int idx = 1;
  for (const auto i : c10::irange(kSpatialDim)) {
    (void)i; // Suppress unused variable
    stride.emplace_back(config_vals.at(idx));
    idx++;
  }
  for (const auto i : c10::irange(kSpatialDim)) {
    (void)i; // Suppress unused variable
    padding.emplace_back(config_vals.at(idx));
    idx++;
  }
  for (const auto i : c10::irange(kSpatialDim)) {
    (void)i; // Suppress unused variable
    dilation.emplace_back(config_vals.at(idx));
    idx++;
  }
  for (const auto i : c10::irange(kSpatialDim)) {
    (void)i; // Suppress unused variable
    output_padding.emplace_back(config_vals.at(idx));
    idx++;
  }
  int64_t groups = config_vals.at(idx);
  idx++;
  int64_t flags = config_vals.at(idx);
  idx++;
  TORCH_INTERNAL_ASSERT(
      idx == static_cast<int64_t>(config_vals.size()),
      "Unexpected length of config_vals, expected ",
      idx,
      " got ",
      config_vals.size());

  bool transpose = flags & (1 << 0);

  int64_t other_flags = flags & ~(1 << 0);
  TORCH_INTERNAL_ASSERT(
      other_flags == 0, "Unexpected flags set in ", flags, ".");

  return at::AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP<kSpatialDim>::
      prepack(
          weight.value(),
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          transpose);
}
#endif
