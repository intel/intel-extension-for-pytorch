#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>

#include <ATen/aten_ipex_type_dpcpp.h>
#include <core/Context.h>
#include <core/TensorImplUtils.h>
#include <utils/Numerics.h>

#include "Loops.h"

#include <core/Runtime.h>

DPCPP_DEF_K1(intrepr);
DPCPP_DEF_K1(make_per_tensor_quantized_tensor_dpcpp);

using namespace mkldnn;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor dequantize_tensor_per_tensor_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  auto stream = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kDPCPP, current_device());
  auto q_eng = GpuEngineManager::Instance().get_engine(curDevice);

  memory::dims q_dims = qtensor.dim() == 4
      ? memory::dims({qtensor.size(0),
                      qtensor.size(1),
                      qtensor.size(2),
                      qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type q_dt = dt_to_dnnl(qtensor.scalar_type());
  memory::format_tag q_fmt = qtensor.dim() == 4
      ? memory::format_tag::nchw
      : qtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;

  auto q_md = memory::desc(q_dims, q_dt, q_fmt);
  auto q_ctx =
      at::AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(qtensor);
  auto q_m = q_ctx.is_plain()
      ? dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr())
      : dpcpp_onednn_memory({q_ctx.meta()}, q_eng, qtensor.data_ptr());

  memory::dims r_dims = q_dims;
  memory::data_type r_dt = dt_to_dnnl(rtensor.scalar_type());
  memory::format_tag r_fmt = q_fmt;
  engine r_eng = q_eng;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);
  memory r_m = dpcpp_onednn_memory(r_md, r_eng, rtensor.data_ptr());

  primitive_attr attr;
  int mask = 0;
  attr.set_output_scales(mask, {static_cast<float>(scale)});
  attr.set_zero_points(DNNL_ARG_SRC, mask, {static_cast<int>(zero_point)});
  reorder reorder_p = reorder(q_m, r_m, attr);

  DPCPP_ONEDNN_EXEC(reorder_p, stream, {{DNNL_ARG_FROM, q_m}, {DNNL_ARG_TO, r_m}});
  return rtensor;
}

Tensor dequantize_tensor_per_channel_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  auto stream = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kDPCPP, current_device());
  auto q_eng = GpuEngineManager::Instance().get_engine(curDevice);
  memory::dims q_dims = qtensor.dim() == 4
      ? memory::dims({qtensor.size(0),
                      qtensor.size(1),
                      qtensor.size(2),
                      qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type q_dt = dt_to_dnnl(qtensor.scalar_type());
  memory::format_tag q_fmt = qtensor.dim() == 4
      ? memory::format_tag::nchw
      : qtensor.dim() == 2 ? memory::format_tag::nc : memory::format_tag::x;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
  memory q_m = dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr());

  memory::dims r_dims = q_dims;
  memory::data_type r_dt = dt_to_dnnl(rtensor.scalar_type());
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
  zps.push_back(
      zero_points[0]
          .item()
          .to<float>()); // oneDNN only support single zero_point by currently.
  attr.set_output_scales(mask_0, {scls});
  attr.set_zero_points(DNNL_ARG_SRC, mask_1, {zps});
  reorder reorder_p = reorder(q_m, r_m, attr);

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

} // namespace AtenIpexTypeDPCPP
} // namespace at
