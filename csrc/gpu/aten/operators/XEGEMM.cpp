#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <xetla/GEMM.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

#if defined(USE_XETLA)

namespace at {
namespace AtenIpexTypeXPU {

#define GEMM_XETLA_DISPATCH(F)                                          \
  {                                                                     \
    RECORD_FUNCTION("torch_ipex::" #F, c10::ArrayRef<c10::IValue>({})); \
    F(q,                                                                \
      reinterpret_cast<sycl::half*>(output.data_ptr<scalar_t>()),       \
      reinterpret_cast<sycl::half*>(a.data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(b.data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(bias.data_ptr<scalar_t>()),         \
      reinterpret_cast<sycl::half*>(res0.data_ptr<scalar_t>()),         \
      reinterpret_cast<sycl::half*>(res1.data_ptr<scalar_t>()),         \
      m,                                                                \
      n,                                                                \
      k);                                                               \
  }

static Tensor hgemm_bias_res_res(
    const Tensor& a,
    const Tensor& b,
    const Tensor& bias,
    const Tensor& res0,
    const Tensor& res1) {
  // a: m x k, b: k x n, bias: n, res0/1: m x n
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && bias.dim() == 1 && res0.dim() == 2 &&
      res1.dim() == 2);

  int m = a.sizes()[0];
  int n = b.sizes()[1];
  int k = a.sizes()[1];

  TORCH_CHECK(bias.sizes()[0] == n && bias.is_contiguous());
  auto output = at::empty({m, n}, a.options());

  TORCH_CHECK(res0.sizes() == output.sizes() && res1.sizes() == output.sizes());

  bool is_a_contiguous = a.is_contiguous();
  bool is_b_row_major = b.is_contiguous();
  bool is_b_col_major = b.transpose(0, 1).is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(
      a.scalar_type() == kHalf && b.scalar_type() == kHalf &&
      bias.scalar_type() == kHalf && res0.scalar_type() == kHalf &&
      res1.scalar_type() == kHalf);

  using namespace xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& q = dpcppGetCurrentQueue();
  GEMM_XETLA_DISPATCH(hgemm_bias_res_res_8x128_8x16x16_4);
  return output;
}

#undef GEMM_XETLA_DISPATCH

#define GEMM_QKV_XETLA_DISPATCH(F)                                      \
  {                                                                     \
    RECORD_FUNCTION("torch_ipex::" #F, c10::ArrayRef<c10::IValue>({})); \
    F(q,                                                                \
      reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),         \
      reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),         \
      reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),         \
      reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),        \
      reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),       \
      m,                                                                \
      n,                                                                \
      k);                                                               \
  }

static std::tuple<Tensor, Tensor, Tensor> hgemm_qkv(
    const Tensor& input,
    const Tensor& weight) {
  // input: m,k; weight: 3,k,n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3);
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = weight.sizes()[2];
  auto out0 = at::empty({m, n}, input.options());
  auto out1 = at::empty({m, n}, input.options());
  auto out2 = at::empty({m, n}, input.options());

  bool is_a_contiguous = input.is_contiguous();
  bool is_b_row_major = weight.is_contiguous();
  bool is_b_col_major = weight.transpose(1, 2).is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(input.scalar_type() == kHalf && weight.scalar_type() == kHalf);

  using namespace xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& q = dpcppGetCurrentQueue();
  GEMM_QKV_XETLA_DISPATCH(hgemm_qkv_8x128_8x16x32_4);
  return std::forward_as_tuple(out0, out1, out2);
}

#undef GEMM_QKV_XETLA_DISPATCH

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "hgemm_bias_res_res.xpu", at::AtenIpexTypeXPU::hgemm_bias_res_res);
  IPEX_OP_REGISTER("hgemm_qkv.xpu", at::AtenIpexTypeXPU::hgemm_qkv);
}
} // namespace

#endif
