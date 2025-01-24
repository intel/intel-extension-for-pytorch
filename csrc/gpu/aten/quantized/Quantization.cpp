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
using namespace torch_ipex::xpu::dpcpp;

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
  torch_ipex::xpu::oneDNN::ReorderAttr rattr =
      torch_ipex::xpu::oneDNN::ReorderAttr();
  int mask = (1 << axis);
  // See [Note: Scale setting for reorder]
  memory::dims scale_zp_sz = scales.sizes().vec();
  memory::dims scale_zp_st = scales.strides().vec();
  rattr.set_dst_sc_mask(mask);
  rattr.set_dst_zp_mask(mask);
  torch_ipex::xpu::oneDNN::quantized_reorder(
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

  torch_ipex::xpu::oneDNN::ReorderAttr rattr =
      torch_ipex::xpu::oneDNN::ReorderAttr();
  int mask = 0;
  rattr.set_dst_sc_mask(mask);
  if (zero_point != 0)
    rattr.set_dst_zp_mask(mask);
  const memory::dim scale_zp_sz = 1;
  const memory::dim scale_zp_st = 1;
  float dnn_scale = scale;
  auto quant_base = fetch_cached_quantizer_base(dnn_scale, zero_point);
  torch_ipex::xpu::oneDNN::quantized_reorder(
      rtensor,
      qtensor,
      /*src_scale=*/nullptr,
      /*srd_zero_point=*/nullptr,
      quant_base.scale_ptr(),
      quant_base.zero_point_ptr(),
      {scale_zp_sz},
      {scale_zp_st},
      rattr);
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

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
