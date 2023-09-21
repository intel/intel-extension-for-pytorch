#include "XEGEMM_INT4.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"
#include "xetla/GEMM_INT4.h"

#if defined(USE_XETLA)

namespace at {
namespace AtenIpexTypeXPU {

#define GEMM_QKV_WINT4_XETLA_DISPATCH(F)                                   \
  if (!has_bias) {                                                         \
    RECORD_FUNCTION(                                                       \
        "torch_ipex::hgemm_qkv_wint4" #F, c10::ArrayRef<c10::IValue>({})); \
    hgemm_qkv_wint4##F(                                                    \
        q,                                                                 \
        reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),          \
        reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),          \
        reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),          \
        reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),         \
        weight.data_ptr<uint8_t>(),                                        \
        weight_zp.data_ptr<uint8_t>(),                                     \
        reinterpret_cast<sycl::half*>(weight_scl.data_ptr<scalar_t>()),    \
        m,                                                                 \
        n,                                                                 \
        k);                                                                \
  } else {                                                                 \
    RECORD_FUNCTION(                                                       \
        "torch_ipex::hgemm_qkv_bias_wint4" #F,                             \
        c10::ArrayRef<c10::IValue>({}));                                   \
    hgemm_qkv_bias_wint4##F(                                               \
        q,                                                                 \
        reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),          \
        reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),          \
        reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),          \
        reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),         \
        weight.data_ptr<uint8_t>(),                                        \
        weight_zp.data_ptr<uint8_t>(),                                     \
        reinterpret_cast<sycl::half*>(weight_scl.data_ptr<scalar_t>()),    \
        reinterpret_cast<sycl::half*>(bias_.value().data_ptr<scalar_t>()), \
        m,                                                                 \
        n,                                                                 \
        k);                                                                \
  }

#define GEMM_QKV_WINT4_8M_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##8x256_8x16x32##_##gz##_2_)
#define GEMM_QKV_WINT4_8M_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##8x64_8x16x64##_##gz##_8_)
#define GEMM_QKV_WINT4_8M_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##8x64_8x16x64##_##gz##_8_)
#define GEMM_QKV_WINT4_8M_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##8x512_8x16x32##_##gz##_1_)

#define GEMM_QKV_WINT4_16M_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##16x256_16x16x32##_##gz##_2_)
#define GEMM_QKV_WINT4_16M_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##16x64_16x16x32##_##gz##_8_)
#define GEMM_QKV_WINT4_16M_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##16x64_16x16x32##_##gz##_8_)
#define GEMM_QKV_WINT4_16M_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##16x512_16x16x32##_##gz##_1_)

#define GEMM_QKV_WINT4_32M_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##32x256_32x16x32##_##gz##_2_)
#define GEMM_QKV_WINT4_32M_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##32x64_32x16x32##_##gz##_8_)
#define GEMM_QKV_WINT4_32M_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##32x64_32x16x32##_##gz##_8_)
#define GEMM_QKV_WINT4_32M_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##32x512_32x16x32##_##gz##_1_)

#define GEMM_QKV_WINT4_64M_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##64x256_64x16x32##_##gz##_2_)
#define GEMM_QKV_WINT4_64M_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##32x128_32x16x32##_##gz##_4_)
#define GEMM_QKV_WINT4_64M_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##32x128_32x16x32##_##gz##_4_)
#define GEMM_QKV_WINT4_64M_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##64x512_64x16x32##_##gz##_1_)

#define GEMM_QKV_WINT4_384M_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define GEMM_QKV_WINT4_384M_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define GEMM_QKV_WINT4_384M_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define GEMM_QKV_WINT4_384M_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##64x128_64x16x32##_##gz##_4_)

#define GEMM_QKV_WINT4_1024M_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##128x256_64x16x32##_##gz##_1_)
#define GEMM_QKV_WINT4_1024M_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##128x256_64x16x32##_##gz##_1_)
#define GEMM_QKV_WINT4_1024M_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##128x256_64x16x32##_##gz##_1_)
#define GEMM_QKV_WINT4_1024M_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##128x512_64x32x32##_##gz##_1_)

#define GEMM_QKV_WINT4_INFM_16384N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define GEMM_QKV_WINT4_INFM_4096N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define GEMM_QKV_WINT4_INFM_4096N_16384K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define GEMM_QKV_WINT4_INFM_50416N_4096K_IMPL(gz) \
  GEMM_QKV_WINT4_XETLA_DISPATCH(_##256x256_64x32x32##_##gz##_1_)

#define GEMM_QKV_WINT4_M_N_K_DISPATCH(m, n, k, gz) \
  GEMM_QKV_WINT4_##m##M_##n##N_##k##K_IMPL(gz)

#define GEMM_QKV_WINT4_N_K_DISPATCH(m, gz)             \
  if (n == 16384 && k == 4096) {                       \
    GEMM_QKV_WINT4_M_N_K_DISPATCH(m, 16384, 4096, gz); \
  } else if (n == 4096 && k == 4096) {                 \
    GEMM_QKV_WINT4_M_N_K_DISPATCH(m, 4096, 4096, gz);  \
  } else if (n == 4096 && k == 16384) {                \
    GEMM_QKV_WINT4_M_N_K_DISPATCH(m, 4096, 16384, gz); \
  } else if (n == 50416 && k == 4096) {                \
    GEMM_QKV_WINT4_M_N_K_DISPATCH(m, 50416, 4096, gz); \
  } else {                                             \
    TORCH_CHECK(false);                                \
  }

#define GEMM_QKV_WINT4_M_DISPATCH(m)     \
  if (calib_gz == k || calib_gz == -1) { \
    GEMM_QKV_WINT4_N_K_DISPATCH(m, 0);   \
  } else if (calib_gz == 128) {          \
    GEMM_QKV_WINT4_N_K_DISPATCH(m, 128); \
  } else {                               \
    TORCH_CHECK(false);                  \
  }

static void mm_qkv_out_wint4(
    const Tensor& input_,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const optional<Tensor>& bias_,
    const Tensor& out0_,
    const Tensor& out1_,
    const Tensor& out2_,
    int64_t calib_gz) {
  // TORCH_CHECK(calib_gz == 128);
  // std::cout << "this is mm_qkv_out_wint4 ....\n";
  RECORD_FUNCTION("mm_qkv_out_int4", {});
  auto input = input_.flatten(0, -2);
  auto out0 = out0_.flatten(0, -2);
  auto out1 = out1_.flatten(0, -2);
  auto out2 = out2_.flatten(0, -2);
  // input: m,k; weight: 3,k,n, bias(opt): 3,n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3);
  TORCH_CHECK(out0.dim() == 2 && out1.dim() == 2 && out2.dim() == 2);
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int n = weight.sizes()[2] * 2;

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

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(
      input.scalar_type() == kHalf &&
      (weight.scalar_type() == kQUInt8 || weight.scalar_type() == kByte));

  using namespace xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& q = dpcppGetCurrentQueue();

  if (m <= 8) {
    GEMM_QKV_WINT4_M_DISPATCH(8);
  } else if (m <= 16) {
    GEMM_QKV_WINT4_M_DISPATCH(16);
  } else if (m <= 32) {
    GEMM_QKV_WINT4_M_DISPATCH(32);
  } else if (m <= 64) {
    GEMM_QKV_WINT4_M_DISPATCH(64);
  } else if (m <= 384) {
    GEMM_QKV_WINT4_M_DISPATCH(384);
  } else if (m <= 1024) {
    GEMM_QKV_WINT4_M_DISPATCH(1024);
  } else {
    GEMM_QKV_WINT4_M_DISPATCH(INF);
  }
}

#undef GEMM_QKV_WINT4_XETLA_DISPATCH

static std::tuple<Tensor, Tensor, Tensor> mm_qkv_wint4(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[2] * 2;
  auto out0 = at::empty({m, n}, input.options());
  auto out1 = at::empty({m, n}, input.options());
  auto out2 = at::empty({m, n}, input.options());
  mm_qkv_out_wint4(
      input, weight, weight_scl, weight_zp, bias_, out0, out1, out2, calib_gz);
  auto sizes = input.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return std::forward_as_tuple(
      out0.view_symint(sizes),
      out1.view_symint(sizes),
      out2.view_symint(sizes));
}

static Tensor mm_bias_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);

  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[1] * 2;
  auto bias = bias_.flatten();
  auto output = at::empty({m, n}, input.options());

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(bias, HGEMMXetla_INT4::EpilogueType::BIAS)
                    .add_calib_gz(calib_gz)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);

  int m = input_flat.sizes()[0];
  int k = input_flat.sizes()[1];
  int n = weight.sizes()[1] * 2;
  auto output = at::empty({m, n}, input.options());

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_calib_gz(calib_gz)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_silu_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  // a: m x k, b: k x n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
                    .add_calib_gz(calib_gz)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_gelu_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& bias,
    int64_t calib_gz,
    c10::string_view approximate) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto bias_flat = bias.flatten();
  TORCH_CHECK(approximate == "tanh");
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS)
          .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::GELU)
          .add_calib_gz(calib_gz)
          .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_resadd_resadd_int4(
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& res0_,
    const Tensor& res1_,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  auto input = input_.flatten(0, -2);
  auto weight = weight_.flatten(0, -2);
  auto bias = bias_.flatten();
  auto res0 = res0_.flatten(0, -2);
  auto res1 = res1_.flatten(0, -2);
  // a: m x k, b: k x n, bias: n, res0/1: m x n
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 2 && bias.dim() == 1 &&
      res0.dim() == 2 && res1.dim() == 2);
  int m = input.sizes()[0];
  int n = weight.sizes()[1] * 2;
  int k = input.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input)
                    .add_matrix_wei(weight)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(bias, HGEMMXetla_INT4::EpilogueType::BIAS)
                    .add_epilogue(res0, HGEMMXetla_INT4::EpilogueType::RES_ADD)
                    .add_epilogue(res1, HGEMMXetla_INT4::EpilogueType::RES_ADD)
                    .add_calib_gz(calib_gz)
                    .build();
  TORCH_CHECK(policy.fallback() == false);
  policy.run();
  return resize_as_mat1(input_, output);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "mm_qkv_out_int4.xpu", at::AtenIpexTypeXPU::mm_qkv_out_wint4);
  IPEX_OP_REGISTER("mm_qkv_int4.xpu", at::AtenIpexTypeXPU::mm_qkv_wint4);
  IPEX_OP_REGISTER("mm_int4.xpu", at::AtenIpexTypeXPU::mm_int4);
  IPEX_OP_REGISTER("mm_bias_int4.xpu", at::AtenIpexTypeXPU::mm_bias_int4);
  IPEX_OP_REGISTER("mm_silu_int4.xpu", at::AtenIpexTypeXPU::mm_silu_int4);
  IPEX_OP_REGISTER(
      "mm_bias_gelu_int4.xpu", at::AtenIpexTypeXPU::mm_bias_gelu_int4);
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd_int4.xpu",
      at::AtenIpexTypeXPU::mm_bias_resadd_resadd_int4);
}
} // namespace
#endif
