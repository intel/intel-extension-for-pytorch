#pragma once

#include <ATen/quantized/Quantizer.h>
#include <quantized/DeQuantization.h>
#include <quantized/QTensor.h>
#include <quantized/Quantization.h>
#include <runtime/Utils.h>
#include <utils/LRUCache.h>

namespace at {
namespace AtenIpexTypeQuantizedXPU {

using namespace xpu::dpcpp;

template <typename scale_t_, typename zp_t_>
class XPUQuantizerBase {
 public:
  using scale_t = scale_t_;
  using zp_t = zp_t_;
  using scale_ptr_t = std::shared_ptr<scale_t>;
  using zp_ptr_t = std::shared_ptr<zp_t>;

 public:
  XPUQuantizerBase() = default;

  XPUQuantizerBase(size_t size, sycl::queue& q)
      : scale_ptr_(
            sycl::malloc_device<scale_t>(size * sizeof(scale_t), q),
            [=](scale_t* ptr) { sycl::free(ptr, q); }),
        zp_ptr_(
            sycl::malloc_device<zp_t>(size * sizeof(zp_t), q),
            [=](zp_t* ptr) { sycl::free(ptr, q); }) {}
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

struct DPCPPPerTensorAffineQuantizer : public AffineQuantizer {
  using QuantizerBaseType = XPUQuantizerBase<float, int32_t>;
  using scale_t = QuantizerBaseType::scale_t;
  using zp_t = QuantizerBaseType::zp_t;

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
    bool key_found = xpu::dpcpp::find_key<QuantizerBaseType>(key_sc_zp);
    if (key_found) {
      base_ = xpu::dpcpp::fetch_m<QuantizerBaseType>(key_sc_zp);
    } else {
      base_ = QuantizerBaseType(1, dpcppGetCurrentQueue());

      scale_t* sc_ptr = base_.scale_ptr();
      scale_t _scale = (scale_t)dnn_scale;
      dpcppGetCurrentQueue().single_task([=]() { sc_ptr[0] = _scale; });

      zp_t* zp_ptr = base_.zero_point_ptr();
      zp_t _zp = (zp_t)0;
      dpcppGetCurrentQueue().single_task([=]() { zp_ptr[0] = _zp; });

      xpu::dpcpp::fetch_or_create_m<QuantizerBaseType>(key_sc_zp, base_);
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

  scale_t* scale_ptr() {
    return base_.scale_ptr();
  }

  zp_t* zero_point_ptr() {
    return base_.zero_point_ptr();
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
  QuantizerBaseType base_;
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
