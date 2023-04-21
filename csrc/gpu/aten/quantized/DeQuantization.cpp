#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>

#include <oneDNN/oneDNN.h>
#include <quantized/DeQuantization.h>

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
using namespace at::AtenIpexTypeQuantizedXPU;

namespace at {
namespace AtenIpexTypeXPU {

Tensor dequantize_tensor_per_tensor_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  ReorderAttr rattr = ReorderAttr();
  int mask = 0;
  rattr.set_src_sc_and_zp_mask(mask);

  Tensor rtensor_ = at::empty(qtensor.sizes(), rtensor.options());
  if (is_opaque_u8(qtensor)) {
    Tensor dnn_scale =
        at::empty({1}, at::dtype(at::kFloat).device(at::kXPU)).fill_(scale);
    Tensor dnn_zero_point =
        at::zeros({1}, at::dtype(at::kInt).device(at::kXPU));

    // See [Note: Scale setting for reorder]
    xpu::oneDNN::quantized_reorder(
        qtensor,
        rtensor_,
        dnn_scale,
        dnn_zero_point,
        /*dst_scale=*/Tensor(),
        /*dst_zero_point=*/Tensor(),
        rattr);
  } else {
    // See [Note: Scale setting for reorder]
    xpu::oneDNN::quantized_reorder(
        qtensor,
        rtensor_,
        q_scale_ptr(qtensor),
        q_zero_point_ptr(qtensor),
        /*dst_scale=*/nullptr,
        /*dst_zero_point=*/nullptr,
        {1},
        {1},
        rattr);
  }

  return rtensor_;
}

Tensor dequantize_tensor_per_channel_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  auto q_eng =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});

  memory::dims q_dims = qtensor.dim() == 4 ? memory::dims(
                                                 {qtensor.size(0),
                                                  qtensor.size(1),
                                                  qtensor.size(2),
                                                  qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type q_dt = get_onednn_dtype(qtensor);
  memory::format_tag q_fmt = qtensor.dim() == 4 ? memory::format_tag::nchw
      : qtensor.dim() == 2                      ? memory::format_tag::nc
                                                : memory::format_tag::x;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
  memory q_m = dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr());

  memory::dims r_dims = q_dims;
  memory::data_type r_dt = get_onednn_dtype(rtensor);
  memory::format_tag r_fmt = q_fmt;
  engine r_eng = q_eng;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);

  int num_channels = qtensor.size(axis);
  ReorderAttr rattr;
  int mask = (1 << axis);
  // See [Note: Scale setting for reorder]
  Tensor dnn_scale =
      at::ones({num_channels}, at::dtype(at::kFloat).device(at::kXPU)) *
      scales.to(at::kFloat);
  // TODO: Remove workaround for dnnl symmetric quantization
  Tensor dnn_zero_point =
      at::zeros({num_channels}, at::dtype(at::kInt).device(at::kXPU));

  rattr.set_src_sc_and_zp_mask(mask);

  Tensor rtensor_ = empty_opaque_tensor(r_md, rtensor.options(), c10::nullopt);
  xpu::oneDNN::quantized_reorder(
      qtensor,
      rtensor_,
      dnn_scale,
      dnn_zero_point,
      /*dst_scale=*/Tensor(),
      /*dst_zero_point=*/Tensor(),
      rattr);

  return rtensor_;
}

Tensor dequantize(const Tensor& self) {
  if (!self.is_quantized()) {
    return self;
  }
  auto qtensor = static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
  return qtensor->quantizer()->dequantize(self);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor dequantize(const Tensor& self) {
  if (!self.is_quantized()) {
    return self;
  }
  auto qtensor = static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
  return qtensor->quantizer()->dequantize(self);
}
} // namespace AtenIpexTypeQuantizedXPU

} // namespace at
