#include "XEGEMM.h"
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

static Tensor mm_bias_resadd_resadd(
    const Tensor& a_,
    const Tensor& b_,
    const Tensor& bias_,
    const Tensor& res0_,
    const Tensor& res1_) {
  auto a = a_.flatten(0, -2);
  auto b = b_.flatten(0, -2);
  auto bias = bias_.flatten();
  auto res0 = res0_.flatten(0, -2);
  auto res1 = res1_.flatten(0, -2);
  // a: m x k, b: k x n, bias: n, res0/1: m x n
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && bias.dim() == 1 && res0.dim() == 2 &&
      res1.dim() == 2);
  int m = a.sizes()[0];
  int n = b.sizes()[1];
  int k = a.sizes()[1];
  auto output = at::empty({m, n}, a.options());

  auto policy = HGEMMXetla()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(bias, HGEMMXetla::EpilogueType::BIAS)
                    .add_epilogue(res0, HGEMMXetla::EpilogueType::RES_ADD)
                    .add_epilogue(res1, HGEMMXetla::EpilogueType::RES_ADD)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return output;
}

static Tensor mm_resmul(
    const Tensor& a_,
    const Tensor& b_,
    const Tensor& res_) {
  auto a = a_.flatten(0, -2);
  auto b = b_.flatten(0, -2);
  auto res = res_.flatten(0, -2);
  // a: m x k, b: k x n, res: m, n
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && res.dim() == 2);
  int m = a.sizes()[0];
  int n = b.sizes()[1];
  int k = a.sizes()[1];
  auto output = at::empty({m, n}, a.options());

  auto policy = HGEMMXetla()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(res, HGEMMXetla::EpilogueType::RES_MUL)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return output;
}

static Tensor mm_silu(const Tensor& a_, const Tensor& b_) {
  auto a = a_.flatten(0, -2);
  auto b = b_.flatten(0, -2);
  // a: m x k, b: k x n
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  int m = a.sizes()[0];
  int n = b.sizes()[1];
  int k = a.sizes()[1];
  auto output = at::empty({m, n}, a.options());

  auto policy = HGEMMXetla()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(Tensor(), HGEMMXetla::EpilogueType::SILU)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
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

static void mm_qkv_out(
    const Tensor& input_,
    const Tensor& weight,
    const Tensor& out0_,
    const Tensor& out1_,
    const Tensor& out2_) {
  auto input = input_.flatten(0, -2);
  auto out0 = out0_.flatten(0, -2);
  auto out1 = out1_.flatten(0, -2);
  auto out2 = out2_.flatten(0, -2);
  // input: m,k; weight: 3,k,n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3);
  TORCH_CHECK(out0.dim() == 2 && out1.dim() == 2 && out2.dim() == 2);
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = weight.sizes()[2];

  TORCH_CHECK(
      out0.sizes()[0] == m && out1.sizes()[0] == m && out2.sizes()[0] == m);
  TORCH_CHECK(
      out0.sizes()[1] == n && out1.sizes()[1] == n && out2.sizes()[1] == n);

  bool is_a_contiguous = input.is_contiguous();
  bool is_b_row_major = weight.is_contiguous();
  bool is_b_col_major = weight.transpose(1, 2).is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(input.scalar_type() == kHalf && weight.scalar_type() == kHalf);

  using namespace xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& q = dpcppGetCurrentQueue();
  if (m <= 32) {
    GEMM_QKV_XETLA_DISPATCH(hgemm_qkv_16x256_8x16x16_1);
  } else {
    GEMM_QKV_XETLA_DISPATCH(hgemm_qkv_256x256_32x64x32_1);
  }
}

#undef GEMM_QKV_XETLA_DISPATCH

static std::tuple<Tensor, Tensor, Tensor> mm_qkv(
    const Tensor& input,
    const Tensor& weight) {
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = weight.sizes()[2];
  auto out0 = at::empty({m, n}, input.options());
  auto out1 = at::empty({m, n}, input.options());
  auto out2 = at::empty({m, n}, input.options());
  mm_qkv_out(input, weight, out0, out1, out2);
  return std::forward_as_tuple(out0, out1, out2);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd_resadd);
  IPEX_OP_REGISTER("mm_resmul.xpu", at::AtenIpexTypeXPU::mm_resmul);
  IPEX_OP_REGISTER("mm_silu.xpu", at::AtenIpexTypeXPU::mm_silu);
  IPEX_OP_REGISTER("mm_qkv_out.xpu", at::AtenIpexTypeXPU::mm_qkv_out);
  IPEX_OP_REGISTER("mm_qkv.xpu", at::AtenIpexTypeXPU::mm_qkv);
}
} // namespace

#endif
