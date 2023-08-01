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
  return resize_as_mat1(a_, output);
}

static Tensor mm_bias_resadd(
    const Tensor& a_,
    const Tensor& b_,
    const Tensor& bias_,
    const Tensor& res_,
    const double res_scale) {
  auto a = a_.flatten(0, -2);
  auto b = b_.flatten(0, -2);
  auto bias = bias_.flatten();
  auto res = res_.flatten(0, -2);
  // a: m x k, b: k x n, bias: n, res: m x n
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && bias.dim() == 1 && res.dim() == 2);
  int m = a.sizes()[0];
  int n = b.sizes()[1];
  int k = a.sizes()[1];
  auto output = at::empty({m, n}, a.options());

  auto policy =
      HGEMMXetla()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMMXetla::EpilogueType::BIAS)
          .add_epilogue(
              res, HGEMMXetla::EpilogueType::SCALED_RES_ADD, res_scale)
          .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return resize_as_mat1(a_, output);
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
  return resize_as_mat1(a_, output);
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
  return resize_as_mat1(a_, output);
}

static void mm_qkv_out(
    const Tensor& input_,
    const Tensor& weight,
    const optional<Tensor>& bias_,
    const Tensor& out0_,
    const Tensor& out1_,
    const Tensor& out2_) {
  auto input = input_.flatten(0, -2);
  auto out0 = out0_.flatten(0, -2);
  auto out1 = out1_.flatten(0, -2);
  auto out2 = out2_.flatten(0, -2);
  // input: m,k; weight: 3,k,n, bias(opt): 3,n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3);
  TORCH_CHECK(out0.dim() == 2 && out1.dim() == 2 && out2.dim() == 2);
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = weight.sizes()[2];

  bool has_bias = bias_.has_value();
  if (has_bias) {
    auto bias = bias_.value();
    TORCH_CHECK(
        bias.dim() == 2 && bias.sizes()[0] == 3 && bias.sizes()[1] == n);
  }

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

  int selected_policy = select_gemm_config(m, n, k, is_b_row_major, 64);

  char str__[80];
  if (!has_bias) {
    sprintf(str__, "hgemm_qkv_%d(%d, %d, %d)", selected_policy, m, n, k);
    RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));
    hgemm_qkv_policies[selected_policy](
        q,
        reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
        m,
        n,
        k);
  } else {
    sprintf(str__, "hgemm_qkv_bias_%d(%d, %d, %d)", selected_policy, m, n, k);
    RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));
    hgemm_qkv_bias_policies[selected_policy](
        q,
        reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
        reinterpret_cast<sycl::half*>(bias_.value().data_ptr<scalar_t>()),
        m,
        n,
        k);
  }
}

static std::tuple<Tensor, Tensor, Tensor> mm_qkv(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_) {
  auto input_flat = input.flatten(0, -2);
  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[2];
  auto out0 = at::empty({m, n}, input.options());
  auto out1 = at::empty({m, n}, input.options());
  auto out2 = at::empty({m, n}, input.options());
  mm_qkv_out(input, weight, bias_, out0, out1, out2);
  auto sizes = input.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return std::forward_as_tuple(
      out0.view_symint(sizes),
      out1.view_symint(sizes),
      out2.view_symint(sizes));
}

Tensor matmul_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    c10::string_view approximate) {
  auto input_flat = input.flatten(0, -2);
  if (bias.has_value() && approximate == "tanh") {
    int m = input_flat.sizes()[0];
    int n = weight.sizes()[1];
    int k = input_flat.sizes()[1];
    auto bias_ = bias.value();
    auto output = at::empty({m, n}, input.options());
    auto policy = HGEMMXetla()
                      .add_matrix_c(output)
                      .add_matrix_a(input_flat)
                      .add_matrix_b(weight)
                      .add_epilogue(bias_, HGEMMXetla::EpilogueType::BIAS)
                      .add_epilogue(Tensor(), HGEMMXetla::EpilogueType::GELU)
                      .build();
    if (policy.fallback() == false) {
      policy.run();
      return resize_as_mat1(input, output);
    }
  }
  TORCH_CHECK(false);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd_resadd);
  IPEX_OP_REGISTER("mm_bias_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd);
  IPEX_OP_REGISTER("mm_resmul.xpu", at::AtenIpexTypeXPU::mm_resmul);
  IPEX_OP_REGISTER("mm_silu.xpu", at::AtenIpexTypeXPU::mm_silu);
  IPEX_OP_REGISTER("mm_qkv_out.xpu", at::AtenIpexTypeXPU::mm_qkv_out);
  IPEX_OP_REGISTER("mm_qkv.xpu", at::AtenIpexTypeXPU::mm_qkv);
  IPEX_OP_REGISTER("matmul_gelu.xpu", at::AtenIpexTypeXPU::matmul_gelu);
}
} // namespace

#endif
