#if defined(USE_XETLA)
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <sycl/sycl.hpp>

#include "XEGEMM_INT4_marlin.h"

#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

inline Tensor resize_as_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

static void mm_int4_out_marlin(
    Tensor& out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const c10::optional<Tensor>& weight_zp,
    const int64_t group_size) {
  using dtype_a = sycl::half;
  using dtype_b = uint32_t;
  using dtype_c = sycl::half;
  using dtype_zp = uint32_t;
  using dtype_scale = sycl::half;
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto m = input_flat.size(0);
  auto k = input_flat.size(1);
  auto n = weight_flat.size(1);
  if (out.defined())
    out = out.flatten(0, -2);
  else
    out = at::empty({m, n}, input.options());
  dtype_zp* weight_zp_ptr = nullptr;
  if (weight_zp.has_value()) {
    weight_zp_ptr = static_cast<dtype_zp*>(weight_zp->data_ptr());
  }

  launch_hgemm_wint4_marlin<dtype_a, dtype_b, dtype_c, dtype_zp, dtype_scale>(
      static_cast<dtype_c*>(out.data_ptr()),
      static_cast<dtype_a*>(input_flat.data_ptr()),
      static_cast<dtype_b*>(weight_flat.data_ptr()),
      weight_zp_ptr,
      static_cast<dtype_scale*>(weight_scl.data_ptr()),
      m,
      n,
      k);
  out = resize_as_mat1(input, out);
  return;
}

static void group_mm_int4_out_marlin(
    Tensor& out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const c10::optional<Tensor>& bias,
    const Tensor& rows_for_experts,
    const c10::optional<Tensor>& weight_zp,
    const int64_t group_size) {
  using dtype_a = sycl::half;
  using dtype_b = uint32_t;
  using dtype_c = sycl::half;
  using dtype_zp = uint32_t;
  using dtype_scale = sycl::half;
  auto total_m = input.size(0);
  auto k = input.size(1);
  auto experts_num = weight.size(0);
  auto n = weight.size(2);
  auto average_m = (total_m + experts_num - 1) / experts_num;
  if (out.defined())
    out = out.flatten(0, -2);
  else
    out = at::empty({total_m, n}, input.options());
  dtype_zp* weight_zp_ptr = nullptr;
  if (weight_zp.has_value()) {
    weight_zp_ptr = static_cast<dtype_zp*>(weight_zp->data_ptr());
  }
  dtype_a* bias_ptr = nullptr;
  if (bias.has_value()) {
    bias_ptr = static_cast<dtype_a*>(bias->data_ptr());
  }
  Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, input.options().dtype(at::kInt));

  launch_group_hgemm_wint4_marlin<
      dtype_a,
      dtype_b,
      dtype_c,
      dtype_zp,
      dtype_scale>(
      static_cast<dtype_c*>(out.data_ptr()),
      static_cast<dtype_a*>(input.data_ptr()),
      static_cast<dtype_b*>(weight.data_ptr()),
      weight_zp_ptr,
      static_cast<dtype_scale*>(weight_scl.data_ptr()),
      bias_ptr,
      static_cast<int*>(atomic_buffer.data_ptr()),
      reinterpret_cast<int*>(rows_for_experts.data_ptr()),
      experts_num,
      average_m,
      n,
      k);
  out = resize_as_mat1(input, out);
  return;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "mm_int4_out_marlin.xpu", at::AtenIpexTypeXPU::mm_int4_out_marlin);
  IPEX_OP_REGISTER(
      "group_mm_int4_out_marlin.xpu",
      at::AtenIpexTypeXPU::group_mm_int4_out_marlin);
}
} // namespace

#endif // USE_XETLA