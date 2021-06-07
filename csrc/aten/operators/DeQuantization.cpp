#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>
#include <intrinsic/ipex_intrinsic.h>

#include <core/Context.h>
#include <core/TensorImplUtils.h>
#include "comm/Numerics.h"
#include <oneDNN/oneDNN.h>

#include "Loops.h"


DPCPP_DEF_K1(intrepr);
DPCPP_DEF_K1(make_per_tensor_quantized_tensor_dpcpp);

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
  auto q_eng = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto stream = GpuStreamManager::Instance().get_stream();
  memory::dims q_dims = qtensor.dim() == 4
      ? memory::dims({qtensor.size(0), qtensor.size(1), qtensor.size(2), qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type q_dt = get_onednn_dtype(qtensor);
  memory::format_tag q_fmt = qtensor.dim() == 4 ? memory::format_tag::nchw
      : qtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;

  auto q_md = memory::desc(q_dims, q_dt, q_fmt);
  auto q_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(qtensor);
  auto q_m = q_ctx.is_plain() ? dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr())
      : dpcpp_onednn_memory({q_ctx.meta()}, q_eng, qtensor.data_ptr());

  memory::dims r_dims = q_m.get_desc().dims();
  memory::data_type r_dt = get_onednn_dtype(rtensor);
  memory::format_tag r_fmt = r_dims.size() == 4 ? memory::format_tag::nchw
      : qtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;
  engine r_eng = q_eng;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);
  memory r_m = dpcpp_onednn_memory(r_md, r_eng, rtensor.data_ptr());

  primitive_attr attr;
  int mask = 0;
  attr.set_output_scales(mask, {static_cast<float>(scale)});
  attr.set_zero_points(DNNL_ARG_SRC, mask, {static_cast<int>(zero_point)});

#ifdef USE_PRIMITIVE_CACHE
     lru_key_t key;
     create_key(key, q_md, r_md, scale);
     auto reorder_p = fetch_or_create_m<dnnl::reorder>(key, q_m, r_m, attr);
#else
     auto reorder_p = dnnl::reorder(q_m, r_m, attr);
#endif

  DPCPP_ONEDNN_EXEC(reorder_p, stream, {{DNNL_ARG_FROM, q_m}, {DNNL_ARG_TO, r_m}});
  return rtensor;
}

Tensor dequantize_tensor_per_channel_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  auto q_eng = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto stream = GpuStreamManager::Instance().get_stream();

  memory::dims q_dims = qtensor.dim() == 4
      ? memory::dims({qtensor.size(0), qtensor.size(1), qtensor.size(2), qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type q_dt = get_onednn_dtype(qtensor);
  memory::format_tag q_fmt = qtensor.dim() == 4 ? memory::format_tag::nchw
      : qtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
  memory q_m = dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr());

  memory::dims r_dims = q_dims;
  memory::data_type r_dt = get_onednn_dtype(rtensor);
  memory::format_tag r_fmt = q_fmt;
  engine r_eng = q_eng;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);
  memory r_m = dpcpp_onednn_memory(r_md, r_eng, rtensor.data_ptr());

  primitive_attr attr;
  int mask_0 = 1 << axis;
  int mask_1 = 0;
  std::vector<float> scls;
  std::vector<int> zps;
  for (int i = 0; i < scales.numel(); i++) {
    scls.push_back(scales[i].item().to<float>());
  }

  // oneDNN only support single zero_point by currently.
  zps.push_back(zero_points[0] .item() .to<float>());

  attr.set_output_scales(mask_0, {scls});
  attr.set_zero_points(DNNL_ARG_SRC, mask_1, {zps});
  auto reorder_p = dnnl::reorder(q_m, r_m, attr);

  DPCPP_ONEDNN_EXEC(reorder_p, stream, {{DNNL_ARG_FROM, q_m}, {DNNL_ARG_TO, r_m}});
  return rtensor;
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
