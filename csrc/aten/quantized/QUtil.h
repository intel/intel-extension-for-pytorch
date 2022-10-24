#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/custom_class.h>
#include <torch/custom_class_detail.h>

#include <torch/custom_class.h>
#include <utils/Macros.h>
#include <oneapi/dpl/cmath>

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

template <typename T>
inline T Round(const T x) {
  return oneapi::dpl::nearbyint(x);
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