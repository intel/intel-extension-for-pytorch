#if defined(USE_XETLA)
#include "XeInt4Dequantize.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

static inline void run_int4_dequantize(
    int n,
    int k,
    INT4_DEQUANTIZE_XETLA::WeightCompressType weight_type,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    Tensor* const output) {
  *output = output->defined() ? output->flatten(0, -2)
                              : at::empty({k, n}, weight_scl.options());

  auto launcher = INT4_DEQUANTIZE_XETLA()
                      .add_dequant_weight(*output)
                      .add_weight(weight)
                      .add_scl(weight_scl)
                      .add_group_size(group_size)
                      .add_weight_compress_type(weight_type);

  if (weight_zp.defined()) {
    launcher.add_zp(weight_zp);
  }

  launcher.check();
  launcher.run();
}

static inline void run_int4x8_dequantize(
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size,
    Tensor* const output) {
  auto weight_flat = weight.flatten(0, -2);
  int k = weight_flat.sizes()[1] * 8; // we only support int4x8 weight now.
  int n = weight_flat.sizes()[0];
  run_int4_dequantize(
      n,
      k,
      INT4_DEQUANTIZE_XETLA::WeightCompressType::int4x8,
      weight_flat,
      weight_scl,
      weight_zp,
      group_size,
      output);
}

static Tensor int4x8_dequantize(
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t group_size) {
  Tensor out;
  run_int4x8_dequantize(weight, weight_scl, weight_zp, group_size, &out);
  return out;
}

} // namespace AtenIpexTypeXPU
namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "int4x8_dequantize.xpu", at::AtenIpexTypeXPU::int4x8_dequantize);
}
} // namespace
} // namespace at
#endif
