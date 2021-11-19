#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>

#include <torch/custom_class.h>
#include <utils/Macros.h>

namespace at {
namespace AtenIpexTypeQuantizedXPU {

template <int kSpatialDim>
struct PackedConvWeightQDPCPP : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightQDPCPP(
      Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        dilation_(std::move(dilation)),
        groups_(groups) {}

  Tensor weight;
  c10::optional<Tensor> bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;

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
    AT_ERROR("not implemented yet");
  }

  bool transpose() const override {
    AT_ERROR("not implemented yet");
  }

  int64_t groups() const override {
    return groups_;
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups);
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
