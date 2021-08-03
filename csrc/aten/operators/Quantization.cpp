#include <ATen/ATen.h>
#include <ATen/quantized/QTensorImpl.h>

#include <intrinsic/ipex_intrinsic.h>
#include <oneDNN/oneDNN.h>
#include <quantized/Quantizer.h>
#include "Loops.h"
#include "comm/ATDispatch.h"

DPCPP_DEF_K1(make_per_tensor_quantized_tensor_dpcpp);
DPCPP_DEF_K1(make_per_channel_quantized_tensor_dpcpp);

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

using namespace at::AtenIpexTypeQuantizedXPU;

Tensor _make_per_tensor_quantized_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);
  Tensor self_contig = self.contiguous();
  IPEX_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_tensor_quantized_tensor_dpcpp", [&]() {
        auto iter = TensorIteratorConfig()
                        .add_output(dst)
                        .add_input(self)
                        .check_all_same_dtype(false)
                        .build();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(
            make_per_tensor_quantized_tensor_dpcpp)>(
            iter,
            [=](underlying_t value) -> scalar_t { return scalar_t(value); });
      });
  return dst;
}

Tensor _make_per_channel_quantized_tensor(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  Tensor self_contig = self.contiguous();
  IPEX_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_channel_quantized_tensor_dpcpp", [&]() {
        auto iter = TensorIteratorConfig()
                        .add_output(dst)
                        .add_input(self)
                        .check_all_same_dtype(false)
                        .build();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(
            make_per_channel_quantized_tensor_dpcpp)>(
            iter,
            [=](underlying_t value) -> scalar_t { return scalar_t(value); });
      });
  return dst;
}

Tensor quantize_tensor_per_channel_affine(
    Tensor& qtensor,
    const Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  ReorderAttr rattr = ReorderAttr();
  int mask_0 = 1 << axis;
  int mask_1 = 0;
  std::vector<float> scls;
  std::vector<int> zps;
  for (int i = 0; i < scales.numel(); i++) {
    scls.push_back(1.0f / scales[i].item().to<float>());
  }
  zps.push_back(zero_points[0].item().to<float>());

  rattr.set_dst_sc_and_zp(mask_0, scls, mask_1, zps);
  xpu::oneDNN::reorder(rtensor, qtensor, rattr);

  return qtensor;
}

Tensor quantize_tensor_per_tensor_affine(
    Tensor& qtensor,
    const Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  auto r_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(rtensor);
  if (Settings::I().is_onednn_layout_enabled()) {
    // this is a temporary implementation for forcing linear to fp32 path
    // and get better performance,due to oneDNN int8 kernel slower than fp32
    // kernel by currently.
    if (rtensor.dim() == 2) {
      return rtensor;
    }
    // for int8 weight cache
    if (!r_ctx.is_plain() &&
        (r_ctx.meta().data_type() != memory::data_type::f32)) {
      return rtensor;
    }
  }

  memory::dims r_dims = rtensor.dim() == 4
      ? memory::dims(
            {rtensor.size(0),
             rtensor.size(1),
             rtensor.size(2),
             rtensor.size(3)})
      : rtensor.dim() == 2 ? memory::dims({rtensor.size(0), rtensor.size(1)})
                           : memory::dims({rtensor.size(0)});
  memory::format_tag r_fmt = rtensor.dim() == 4
      ? memory::format_tag::nchw
      : rtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;

  memory::dims q_dims = r_dims;
  memory::data_type q_dt = get_onednn_dtype(qtensor);
  memory::format_tag q_fmt = r_fmt;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);

  Tensor qtensor_opt = qtensor;
  if (!r_ctx.is_plain() && Settings::I().is_onednn_layout_enabled()) {
    if (rtensor.is_quantized())
      return rtensor;
    auto q_type = qtensor.scalar_type();
    auto quantizer =
        dpcpp_make_per_tensor_affine_quantizer(scale, zero_point, q_type);
    qtensor_opt =
        AtenIpexTypeXPU::empty_opaque_qtensor(q_md, c10::nullopt, quantizer);
  }

  ReorderAttr rattr = ReorderAttr();
  int mask = 0;
  std::vector<float> scls = {static_cast<float>(1.0f / scale)};
  std::vector<int> zps = {static_cast<int>(zero_point)};

  rattr.set_dst_sc_and_zp(mask, scls, mask, zps);
  xpu::oneDNN::reorder(rtensor, qtensor_opt, rattr);

  if (!r_ctx.is_plain() && Settings::I().is_onednn_layout_enabled()) {
    auto q_opt_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::release_tensor_ctx(
            qtensor_opt);
    at::AtenIpexTypeXPU::DPCPPTensorContext::set_tensor_ctx(
        rtensor, std::move(q_opt_ctx));
    return rtensor;
  }

  return qtensor;
}

Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  if (self.is_quantized()) {
    return self;
  }
  auto quantizer =
      dpcpp_make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_per_channel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  auto quantizer =
      dpcpp_make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);
  return quantizer->quantize(self);
}

Tensor _empty_affine_quantized(
    IntArrayRef size,
    const TensorOptions& options_,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; "
      "please delete "
      "the redundant setter.");
  auto options = options_.memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  return AtenIpexTypeXPU::new_qtensor(
      size,
      options,
      dpcpp_make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor _empty_per_channel_affine_quantized(
    IntArrayRef size,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    const TensorOptions& options_,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; "
      "please delete "
      "the redundant setter.");
  auto options = options_.memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  TORCH_CHECK(
      options.dtype() == kQInt8 || options.dtype() == kQUInt8,
      "Supported data type for tensor creation is int8 or uint8");
  return AtenIpexTypeXPU::new_qtensor(
      size,
      options,
      dpcpp_make_per_channel_affine_quantizer(
          scales, zero_points, axis, typeMetaToScalarType(options.dtype())));
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor _empty_affine_quantized(
    IntArrayRef size,
    const TensorOptions& options_,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; "
      "please delete "
      "the redundant setter.");
  auto options = options_.memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  return AtenIpexTypeXPU::new_qtensor(
      size,
      options,
      dpcpp_make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  if (self.is_quantized()) {
    return self;
  }
  auto quantizer =
      dpcpp_make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_per_channel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  auto quantizer =
      dpcpp_make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);
  return quantizer->quantize(self);
}

} // namespace AtenIpexTypeQuantizedXPU

} // namespace at
