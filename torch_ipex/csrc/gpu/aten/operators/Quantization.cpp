#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>

#include <ATen/aten_ipex_type_dpcpp.h>
#include <ATen/ipex_type_dpcpp_customized.h>
#include <core/Context.h>
#include <core/TensorImplUtils.h>
#include <utils/ATDispatch.h>
#include <utils/Numerics.h>

#include <core/Runtime.h>
#include "Loops.h"

DPCPP_DEF_K1(make_per_tensor_quantized_tensor_dpcpp);
DPCPP_DEF_K1(make_per_channel_quantized_tensor_dpcpp);

using namespace mkldnn;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

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
        auto iter = TensorIterator();
        iter.add_output(dst);
        iter.add_input(self);
        iter.dont_compute_common_dtype();
        iter.build();
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
        auto iter = TensorIterator();
        iter.add_output(dst);
        iter.add_input(self);
        iter.dont_compute_common_dtype();
        iter.build();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(
            make_per_channel_quantized_tensor_dpcpp)>(
            iter,
            [=](underlying_t value) -> scalar_t { return scalar_t(value); });
      });
  return dst;
}

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<at::PerTensorAffineQuantizer>(
      scalar_type, scale, zero_point);
}

QuantizerPtr make_per_channel_affine_quantizer_dpcpp(
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
  return c10::make_intrusive<at::PerChannelAffineQuantizer>(
      scalar_type, scales_double, zero_points_int64, axis);
}

void quantize_tensor_per_channel_affine_kernel(
    Tensor rtensor,
    Tensor qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  auto stream = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kDPCPP, current_device());
  auto r_eng = GpuEngineManager::Instance().get_engine(curDevice);

  memory::dims r_dims = qtensor.dim() == 4
      ? memory::dims({qtensor.size(0),
                      qtensor.size(1),
                      qtensor.size(2),
                      qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type r_dt = dt_to_dnnl(rtensor.scalar_type());
  memory::format_tag r_fmt = qtensor.dim() == 4
      ? memory::format_tag::nchw
      : qtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);
  memory r_m = dpcpp_onednn_memory(r_md, r_eng, rtensor.data_ptr());

  memory::dims q_dims = r_dims;
  memory::data_type q_dt = dt_to_dnnl(qtensor.scalar_type());
  memory::format_tag q_fmt = r_fmt;
  engine q_eng = r_eng;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
  memory q_m = dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr());

  primitive_attr attr;
  int mask_0 = 1 << axis;
  int mask_1 = 0;
  std::vector<float> scls;
  std::vector<int> zps;
  for (int i = 0; i < scales.numel(); i++) {
    scls.push_back(static_cast<float>(scales.data_ptr<double>()[i]));
  }
  zps.push_back(static_cast<int>(
      zero_points.data_ptr<long>()[0])); // oneDNN only support single
                                         // zero_point by currently.

  attr.set_output_scales(mask_0, {scls});
  attr.set_zero_points(DNNL_ARG_DST, mask_1, {zps});
  reorder reorder_p = reorder(r_m, q_m, attr);

  DPCPP_ONEDNN_EXEC(reorder_p, stream, r_m, q_m);
}

void quantize_tensor_per_tensor_affine_dpcpp_kernel(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point) {
  auto stream = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kDPCPP, current_device());
  auto r_eng = GpuEngineManager::Instance().get_engine(curDevice);

  memory::dims r_dims = rtensor.dim() == 4
      ? memory::dims({rtensor.size(0),
                      rtensor.size(1),
                      rtensor.size(2),
                      rtensor.size(3)})
      : rtensor.dim() == 2 ? memory::dims({rtensor.size(0), rtensor.size(1)})
                           : memory::dims({rtensor.size(0)});
  memory::data_type r_dt = dt_to_dnnl(rtensor.scalar_type());
  memory::format_tag r_fmt = rtensor.dim() == 4
      ? memory::format_tag::nchw
      : rtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);
  memory r_m = dpcpp_onednn_memory(r_md, r_eng, rtensor.data_ptr());

  memory::dims q_dims = r_dims;
  memory::data_type q_dt = dt_to_dnnl(qtensor.scalar_type());
  memory::format_tag q_fmt = r_fmt;
  engine q_eng = r_eng;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
  memory q_m = dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr());

  primitive_attr attr;
  int mask = 0;
  attr.set_output_scales(mask, {static_cast<float>(1.0f / scale)});
  attr.set_zero_points(DNNL_ARG_DST, mask, {static_cast<int>(zero_point)});
  reorder reorder_p = reorder(r_m, q_m, attr);

  DPCPP_ONEDNN_EXEC(reorder_p, stream, r_m, q_m);
}

Tensor quantize_tensor_per_tensor_affine_dpcpp(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point) {
  quantize_tensor_per_tensor_affine_dpcpp_kernel(
      rtensor, qtensor, scale, zero_point);
  return qtensor;
}

Tensor quantize_tensor_per_channel_affine_dpcpp(
    Tensor rtensor,
    Tensor qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  quantize_tensor_per_channel_affine_kernel(
      rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_per_channel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  auto quantizer =
      make_per_channel_affine_quantizer_dpcpp(scales, zero_points, axis, dtype);
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
  auto options =
      options_.merge_in(TensorOptions().memory_format(optional_memory_format));
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  return AtenIpexTypeDPCPP::new_qtensor(
      size,
      options,
      make_per_tensor_affine_quantizer(
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
  auto options =
      options_.merge_in(TensorOptions().memory_format(optional_memory_format));
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  TORCH_CHECK(
      options.dtype() == kQInt8 || options.dtype() == kQUInt8,
      "Supported data type for tensor creation is int8 or uint8");
  return AtenIpexTypeDPCPP::new_qtensor(
      size,
      options,
      make_per_channel_affine_quantizer_dpcpp(
          scales, zero_points, axis, typeMetaToScalarType(options.dtype())));
}

} // namespace AtenIpexTypeDPCPP
} // namespace at

namespace at {

Tensor PerTensorAffineQuantizer::quantize(Tensor rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat, "quantize only works on Float Tensor.");
  Tensor qtensor = AtenIpexTypeDPCPP::new_qtensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());

  rtensor = rtensor.contiguous();

  AtenIpexTypeDPCPP::quantize_tensor_per_tensor_affine_dpcpp(
      rtensor, qtensor, scale_, zero_point_);

  return qtensor;
}

Tensor PerChannelAffineQuantizer::quantize(Tensor rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat, "quantize only works on Float Tensor.");

  Tensor qtensor = AtenIpexTypeDPCPP::new_qtensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());

  rtensor = rtensor.contiguous();

  AtenIpexTypeDPCPP::quantize_tensor_per_channel_affine_dpcpp(
      rtensor, qtensor, scales_, zero_points_, axis_);

  return qtensor;
}

} // namespace at
