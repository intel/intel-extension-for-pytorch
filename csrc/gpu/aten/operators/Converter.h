#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <torch/all.h>
#include <torch/library.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

static inline float pack_bloat16_float(at::BFloat16 top, at::BFloat16 bot) {
  uint16_t* topp = reinterpret_cast<uint16_t*>(&top);
  uint16_t* botp = reinterpret_cast<uint16_t*>(&bot);
  uint32_t hi = static_cast<uint32_t>(*topp);
  uint32_t lo = static_cast<uint32_t>(*botp);
  uint32_t out = (hi << 16) + lo;
  float* outp = reinterpret_cast<float*>(&out);
  return *outp;
}

static inline std::tuple<at::BFloat16, at::BFloat16> unpack_float_bfloat16(
    float fp32_tensor) {
  uint32_t* float_32 = reinterpret_cast<uint32_t*>(&fp32_tensor);
  uint16_t hi = static_cast<uint16_t>(*(float_32) >> 16);
  uint16_t lo = static_cast<uint16_t>(*(float_32));
  at::BFloat16* hip = reinterpret_cast<at::BFloat16*>(&hi);
  at::BFloat16* lop = reinterpret_cast<at::BFloat16*>(&lo);
  return std::make_tuple(*hip, *lop);
}

Tensor cat_bfloat16_float_xpu(
    const at::Tensor top_half,
    const at::Tensor bottom_half) {
  TORCH_CHECK(
      top_half.scalar_type() == at::kBFloat16 &&
          bottom_half.scalar_type() == at::kBFloat16,
      "pack_bfloat16_float: expect both args to be at::BFloat16");

  // top half should be a true bfloat16
  Tensor fp32_tensor = at::empty_strided(
      top_half.sizes(),
      top_half.strides(),
      top_half.options().dtype(at::ScalarType::Float));

  TensorIterator iter = TensorIteratorConfig()
                            .add_output(fp32_tensor)
                            .add_input(top_half)
                            .add_input(bottom_half)
                            .check_all_same_dtype(false)
                            .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "cat_bfloat16_float_kernel",
      [&] {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t top_elem, scalar_t bot_elem) -> float {
              return pack_bloat16_float(top_elem, bot_elem);
            });
      });

  return fp32_tensor;
}

std::tuple<at::Tensor, at::Tensor> split_float_bfloat16_xpu(
    at::Tensor fp32_tensor) {
  TORCH_CHECK(
      fp32_tensor.scalar_type() == at::kFloat,
      "split_float_bfloat16_xpu: expect tensor to be float32 (at::kFloat)");

  // top half should be a true bfloat16
  Tensor top_half = at::empty_strided(
      fp32_tensor.sizes(),
      fp32_tensor.strides(),
      fp32_tensor.options().dtype(at::ScalarType::BFloat16));

  // bottom_half should be all tail tensor
  Tensor bottom_half = at::empty_strided(
      fp32_tensor.sizes(),
      fp32_tensor.strides(),
      fp32_tensor.options().dtype(at::ScalarType::BFloat16));

  TensorIterator iter = TensorIteratorConfig()
                            .add_output(top_half)
                            .add_output(bottom_half)
                            .add_input(fp32_tensor)
                            .check_all_same_dtype(false)
                            .build();

  IPEX_DISPATCH_FLOATING_TYPES(
      iter.input_dtype(), "split_float_bfloat16_kernel", [&] {
        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter,
            [=](scalar_t fp32_elem) -> std::tuple<at::BFloat16, at::BFloat16> {
              return unpack_float_bfloat16(fp32_elem);
            });
      });

  return std::tuple<at::Tensor, at::Tensor>{top_half, bottom_half};
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "split_float_bfloat16",
      at::AtenIpexTypeXPU::split_float_bfloat16_xpu,
      c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "cat_bfloat16_float",
      at::AtenIpexTypeXPU::cat_bfloat16_float_xpu,
      c10::DispatchKey::XPU);
}

} // namespace