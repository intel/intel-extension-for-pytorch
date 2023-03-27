#pragma once

#include <ATen/quantized/Quantizer.h>
#include <quantized/DeQuantization.h>
#include <quantized/QTensor.h>
#include <quantized/Quantization.h>
#include <utils/LRUCache.h>

namespace at {
namespace AtenIpexTypeQuantizedXPU {

struct DPCPPPerTensorAffineQuantizer : public AffineQuantizer {
  explicit DPCPPPerTensorAffineQuantizer(
      ScalarType scalar_type,
      double scale,
      int64_t zero_point)
      : AffineQuantizer(scalar_type), scale_(scale), zero_point_(zero_point) {
    xpu::dpcpp::lru_key_t key_sc_zp;
    float dnn_scale = scale;
    if (scalar_type == kQUInt8) {
      dnn_scale = scale / 2.f;
    }
    // TODO: Modify this line after asymmetric enabled
    xpu::dpcpp::create_key(key_sc_zp, dnn_scale, 0);
    bool key_found = xpu::dpcpp::find_key<std::pair<Tensor, Tensor>>(key_sc_zp);
    if (key_found) {
      std::tie(scale_tensor_, zero_point_tensor_) =
          xpu::dpcpp::fetch_m<std::pair<Tensor, Tensor>>(key_sc_zp);
    } else {
      scale_tensor_ = at::empty({1}, at::dtype(kFloat).device(at::kXPU))
                          .fill_(static_cast<float>(dnn_scale));
      // TODO: Modify this line after asymmetric enabled
      zero_point_tensor_ = at::zeros({1}, at::dtype(kInt).device(at::kXPU));
      xpu::dpcpp::fetch_or_create_m<std::pair<Tensor, Tensor>>(
          key_sc_zp, scale_tensor_, zero_point_tensor_);
    }
  }

  Tensor quantize(const Tensor& rtensor) override {
    TORCH_CHECK(
        rtensor.scalar_type() == kFloat,
        "quantize only works on Float Tensor.");
    Tensor qtensor = AtenIpexTypeXPU::new_qtensor(
        rtensor.sizes(),
        rtensor.options()
            .dtype(scalar_type_)
            .memory_format(rtensor.suggest_memory_format()),
        intrusive_from_this());
    auto rtensor_contig = rtensor.contiguous(rtensor.suggest_memory_format());

    return at::AtenIpexTypeXPU::quantize_tensor_per_tensor_affine(
        qtensor, rtensor_contig, scale_, zero_point_);
  }

  Tensor dequantize(const Tensor& qtensor) override {
    if (!qtensor.is_quantized()) {
      return qtensor;
    }

    Tensor rtensor = at::empty(
        qtensor.sizes(),
        qtensor.options()
            .dtype(at::kFloat)
            .memory_format(qtensor.suggest_memory_format()));
    auto qtensor_contig = qtensor.contiguous(qtensor.suggest_memory_format());

    return at::AtenIpexTypeXPU::dequantize_tensor_per_tensor_affine(
        rtensor, qtensor_contig, scale_, zero_point_);
  }

  QScheme qscheme() const override {
    return kPerTensorAffine;
  }

  double scale() const {
    return scale_;
  }

  int64_t zero_point() const {
    return zero_point_;
  }

  Tensor scale_tensor() {
    return scale_tensor_;
  }

  Tensor zero_point_tensor() {
    return zero_point_tensor_;
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerTensorAffine) {
      return false;
    }
    auto* other_per_tensor_affine =
        static_cast<DPCPPPerTensorAffineQuantizer*>(other.get());
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

  Tensor& dequantize_out(Tensor& out, const Tensor& t) override {
    TORCH_CHECK(false, "not implemented");
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
  Tensor scale_tensor_;
  Tensor zero_point_tensor_;
};

struct DPCPPPerChannelAffineQuantizer : public AffineQuantizer {
  explicit DPCPPPerChannelAffineQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : AffineQuantizer(scalar_type),
        scales_(scales),
        zero_points_(zero_points),
        axis_(axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffine;
  }

  Tensor scales() const {
    return scales_;
  }

  Tensor zero_points() const {
    return zero_points_;
  }

  int64_t axis() const {
    return axis_;
  }

  Tensor quantize(const Tensor& rtensor) override {
    TORCH_CHECK(
        rtensor.scalar_type() == kFloat,
        "quantize only works on Float Tensor.");

    Tensor qtensor = AtenIpexTypeXPU::new_qtensor(
        rtensor.sizes(),
        rtensor.options()
            .dtype(scalar_type_)
            .memory_format(rtensor.suggest_memory_format()),
        intrusive_from_this());

    auto rtensor_contig = rtensor.contiguous(rtensor.suggest_memory_format());

    return at::AtenIpexTypeXPU::quantize_tensor_per_channel_affine(
        qtensor, rtensor_contig, scales_, zero_points_, axis_);
  }

  Tensor dequantize(const Tensor& qtensor) override {
    if (!qtensor.is_quantized()) {
      return qtensor;
    }

    Tensor rtensor = at::empty(
        qtensor.sizes(),
        qtensor.options()
            .dtype(at::kFloat)
            .memory_format(qtensor.suggest_memory_format()));
    auto qtensor_contig = qtensor.contiguous(qtensor.suggest_memory_format());

    return at::AtenIpexTypeXPU::dequantize_tensor_per_channel_affine(
        rtensor, qtensor_contig, scales_, zero_points_, axis_);
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffine) {
      return false;
    }
    auto* other_per_channel_affine =
        static_cast<DPCPPPerChannelAffineQuantizer*>(other.get());
    return scalar_type() == other_per_channel_affine->scalar_type() &&
        scales().equal(other_per_channel_affine->scales()) &&
        zero_points().equal(other_per_channel_affine->zero_points()) &&
        axis() == other_per_channel_affine->axis();
  }

  Tensor& dequantize_out(Tensor& out, const Tensor& t) override {
    TORCH_CHECK(false, "not implemented");
  }

 private:
  Tensor scales_;
  Tensor zero_points_;
  const int64_t axis_;
};

static inline QuantizerPtr dpcpp_make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<DPCPPPerTensorAffineQuantizer>(
      scalar_type, scale, zero_point);
}

static inline QuantizerPtr dpcpp_make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match");
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");
  TORCH_CHECK(
      isIntegralType(zero_points.scalar_type(), false /*includeBool*/),
      "zero_points tensor must have integral type");
  Tensor scales_double = scales.to(kDouble).contiguous();
  Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
  return c10::make_intrusive<DPCPPPerChannelAffineQuantizer>(
      scalar_type, scales_double, zero_points_int64, axis);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
