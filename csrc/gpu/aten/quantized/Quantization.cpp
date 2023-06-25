#include <ATen/InitialTensorOptions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/detail/ListUtils.h>
#include <oneDNN/oneDNN.h>
#include <quantized/QTensor.h>
#include <quantized/Quantization.h>
#include <quantized/Quantizer.h>
#include <utils/DPCPP.h>

using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

using namespace at::AtenIpexTypeQuantizedXPU;

Tensor quantize_per_tensor_dynamic(
    const Tensor& self,
    ScalarType dtype,
    bool reduce_range = false) {
  TORCH_CHECK(
      (dtype == ScalarType::QInt8 || dtype == ScalarType::QUInt8 ||
       dtype == ScalarType::Half),
      "dtype ",
      dtype,
      "not supported");
  auto input_contig = self.contiguous();
  if (dtype == ScalarType::Half) {
    return input_contig.to(ScalarType::Half);
  }
  float x_min = input_contig.min().item<float>();
  float x_max = input_contig.max().item<float>();

  int qmin;
  int qmax;

  if (dtype == ScalarType::QInt8) {
    qmin = -128;
    qmax = 127;
  } else {
    // for now, this branch executes for dtype == ScalarType::QUInt8
    // additional cases will be added when quantization support for other dtypes
    // becomes available
    qmin = 0;
    qmax = 255;
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/qmin,
      /*qmax=*/qmax,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  return at::AtenIpexTypeXPU::quantize_per_tensor(
      self, q_params.scale, q_params.zero_point, dtype);
}

Tensor quantize_tensor_per_channel_affine(
    Tensor& qtensor,
    const Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  xpu::oneDNN::ReorderAttr rattr = xpu::oneDNN::ReorderAttr();
  int mask = (1 << axis);
  // See [Note: Scale setting for reorder]
  memory::dims scale_zp_sz = scales.sizes().vec();
  memory::dims scale_zp_st = scales.strides().vec();
  rattr.set_dst_sc_mask(mask);
  xpu::oneDNN::quantized_reorder(
      rtensor,
      qtensor,
      /*src_scale=*/nullptr,
      /*src_zero_point=*/nullptr,
      (float*)scales.data_ptr(),
      (int32_t*)zero_points.data_ptr(),
      scale_zp_sz,
      scale_zp_st,
      rattr);

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
        (r_ctx.meta().get_data_type() != memory::data_type::f32)) {
      return rtensor;
    }
  }

  xpu::oneDNN::ReorderAttr rattr = xpu::oneDNN::ReorderAttr();
  int mask = 0;
  rattr.set_dst_sc_mask(mask);
  bool asymmetric = false;
  if (asymmetric && zero_point != 0)
    rattr.set_dst_zp_mask(mask);
  const memory::dim scale_zp_sz = 1;
  const memory::dim scale_zp_st = 1;
  float dnn_scale = scale;
  // TODO: Remove workaround for dnnl symmetric quantization
  int dnn_zero_point = 0;
  auto quant_base = fetch_cached_quantizer_base(dnn_scale, dnn_zero_point);
  // See [Note: Scale setting for reorder]
  // rattr.set_dst_sc_and_zp_mask(mask);
  if (qtensor.scalar_type() == kQUInt8 && zero_point == 128) {
    Tensor qtensor_opt = qtensor;
    memory::dims q_dims = xpu::oneDNN::get_onednn_dims(rtensor);
    memory::format_tag q_fmt = xpu::oneDNN::get_dnnl_default_format(
        rtensor.dim(), is_smf_channels_last(rtensor));

    // We will force to specify s8 as quantization data type to meet the
    // requirement of pytorch calibration with unified data type. Dueing to
    // PyTorch use zp=128 for u8 symmetric quantization, while oneDNN use 0. We
    // need forcely quant input to a s8 tensor.
    memory::data_type q_dt = memory::data_type::s8;
    memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
    auto quantizer = dpcpp_make_per_tensor_affine_quantizer(scale, 0, kQInt8);

    qtensor_opt =
        AtenIpexTypeXPU::empty_opaque_qtensor(q_md, c10::nullopt, quantizer);
    xpu::oneDNN::quantized_reorder(
        rtensor,
        qtensor_opt,
        /*src_scale*/ nullptr,
        /*src_zero_point*/ nullptr,
        quant_base.scale_ptr(),
        quant_base.zero_point_ptr(),
        {scale_zp_sz},
        {scale_zp_st},
        rattr);
    auto q_opt_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::release_tensor_ctx(
            qtensor_opt);
    at::AtenIpexTypeXPU::DPCPPTensorContext::set_tensor_ctx(
        qtensor, std::move(q_opt_ctx));
  } else {
    xpu::oneDNN::quantized_reorder(
        rtensor,
        qtensor,
        /*src_scale=*/nullptr,
        /*srd_zero_point=*/nullptr,
        quant_base.scale_ptr(),
        quant_base.zero_point_ptr(),
        {scale_zp_sz},
        {scale_zp_st},
        rattr);
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

Tensor quantize_per_tensor(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    ScalarType dtype) {
  if (self.is_quantized()) {
    return self;
  }

  auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
      scale.item().toDouble(), zero_point.item().toLong(), dtype);
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
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
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
          scale, 0, typeMetaToScalarType(options.dtype())));
}

Tensor _empty_per_channel_affine_quantized(
    IntArrayRef size,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
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

using namespace at::AtenIpexTypeXPU;

Tensor _empty_affine_quantized(
    IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
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
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
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

Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  if (self.is_quantized()) {
    return self;
  }
  auto quantizer = dpcpp_make_per_tensor_affine_quantizer(scale, 0, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_per_tensor(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    ScalarType dtype) {
  if (self.is_quantized()) {
    return self;
  }
  auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
      scale.item().toDouble(), zero_point.item().toLong(), dtype);
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
