#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>

#include <oneDNN/oneDNN.h>
#include <quantized/QUtils.h>
#include <runtime/Utils.h>
#include "comm/ParamUtils.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

Tensor q_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  auto c = at::add(qa, qb, scale);
  return c;
}

Tensor q_add_relu(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  bool qa_is_opaque_u8 = is_opaque_u8(qa);
  bool qb_is_opaque_u8 = is_opaque_u8(qb);
  auto qa_ = to_plain_if_needed(qa);
  auto qb_ = to_plain_if_needed(qb);
  // For opaque u8 tensor, no need for s8 conversion, it is a s8 tensor inside.
  qa_ =
      (qa_.scalar_type() == kQUInt8 && (!qa_is_opaque_u8)) ? u8tos8(qa_) : qa_;
  qb_ =
      (qb_.scalar_type() == kQUInt8 && (!qb_is_opaque_u8)) ? u8tos8(qb_) : qb_;

  // Note: We currently only support symmetric quantization, and
  // the zp of qu8 tensor would has same zp (128) as PyTorch.
  // But all inside computation for u8 uses 0 as zp,
  // and dnn_scale_u8 = torch_scale_u8 / 2
  auto out = at::_empty_affine_quantized(
      qa_.sizes(),
      ScalarType::QUInt8,
      c10::nullopt,
      qa_.device(),
      c10::nullopt,
      scale,
      128,
      qa_.suggest_memory_format());

  float oscale = scale;
  float ascale = qa_.q_scale();
  float bscale = qb_.q_scale();

  auto func = [=](int8_t a, int8_t b) -> uint8_t {
    float fa = a * ascale;
    float fb = b * bscale;
    float fo = fa + fb;
    fo = fo >= 0.f ? fo : 0.f;
    return quantize_val<uint8_t>((float)(oscale / 2.f), 0.f, fo);
  };

  auto& dpcpp_queue = xpu::dpcpp::dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    int8_t* qa_ptr = (int8_t*)qa_.data_ptr();
    int8_t* qb_ptr = (int8_t*)qb_.data_ptr();
    uint8_t* o_ptr = (uint8_t*)out.data_ptr();
    cgh.parallel_for(
        cl::sycl::range<1>(qa_.numel()), [=](cl::sycl::item<1> item) {
          auto i = item.get_linear_id();
          o_ptr[i] = func(qa_ptr[i], qb_ptr[i]);
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return out;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("add", q_add);
  m.impl("add_relu", q_add_relu);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
