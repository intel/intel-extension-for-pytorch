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

namespace at {
namespace AtenIpexTypeXPU {

Tensor dequantize_tensor_per_tensor_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  ReorderAttr rattr = ReorderAttr();
  int mask = 0;
  auto q_ctx = DPCPPTensorContext::get_tensor_ctx(qtensor);
  std::vector<float> scls = {
      ((q_ctx.is_plain() ? get_onednn_dtype(qtensor)
                         : q_ctx.meta().data_type()) == memory::data_type::u8 &&
       qtensor.q_zero_point() == 128)
          ? static_cast<float>(scale / 2)
          : static_cast<float>(scale)};
  std::vector<int> zps = {static_cast<int>(0)};
  rattr.set_src_sc_and_zp(mask, scls, mask, zps);

  Tensor rtensor_ = at::empty(qtensor.sizes(), rtensor.options());
  xpu::oneDNN::reorder(qtensor, rtensor_, rattr);

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

  ReorderAttr rattr = ReorderAttr();
  int mask_0 = 1 << axis;
  int mask_1 = 0;
  std::vector<float> scls;
  std::vector<int> zps;
  for (int i = 0; i < scales.numel(); i++) {
    scls.push_back(scales[i].item().to<float>());
  }

  // oneDNN only support single zero_point by currently.
  zps.push_back(zero_points[0].item().to<float>());

  rattr.set_src_sc_and_zp(mask_0, scls, mask_1, zps);

  Tensor rtensor_ = empty_opaque_tensor(r_md, rtensor.options(), c10::nullopt);
  xpu::oneDNN::reorder(qtensor, rtensor_, rattr);

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
