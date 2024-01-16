#include "XEGEMM_INT4.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

#if defined(USE_XETLA)

namespace at {
namespace AtenIpexTypeXPU {

#define GEMM_QKV_WINT4_XETLA_DISPATCH(has_bias, arch)                         \
  if (!has_bias) {                                                            \
    auto policy =                                                             \
        HGEMMXetla_INT4()                                                     \
            .add_matrix_out(out0)                                             \
            .add_matrix_out(out1)                                             \
            .add_matrix_out(out2)                                             \
            .add_matrix_inp(input)                                            \
            .add_matrix_wei(weight)                                           \
            .add_matrix_scl(weight_scl)                                       \
            .add_matrix_zp(weight_zp)                                         \
            .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SPLIT3)    \
            .add_calib_gz(calib_gz)                                           \
            .add_arch(arch)                                                   \
            .build();                                                         \
    TORCH_CHECK(policy.fallback() == false, "qkv: invalid gemm shape");       \
    policy.run();                                                             \
  } else {                                                                    \
    auto policy =                                                             \
        HGEMMXetla_INT4()                                                     \
            .add_matrix_out(out0)                                             \
            .add_matrix_out(out1)                                             \
            .add_matrix_out(out2)                                             \
            .add_matrix_inp(input)                                            \
            .add_matrix_wei(weight)                                           \
            .add_matrix_scl(weight_scl)                                       \
            .add_matrix_zp(weight_zp)                                         \
            .add_epilogue(bias_.value(), HGEMMXetla_INT4::EpilogueType::BIAS) \
            .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SPLIT3)    \
            .add_calib_gz(calib_gz)                                           \
            .add_arch(arch)                                                   \
            .build();                                                         \
    TORCH_CHECK(policy.fallback() == false, "qkv bias: invalid gemm shape");  \
    policy.run();                                                             \
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
      (weight.scalar_type() == kQUInt8 || weight.scalar_type() == kByte ||
       weight.scalar_type() == kChar));

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int8_t fp64_valid =
      static_cast<int8_t>(Settings::I().has_2d_block_array(curDevID));
  GEMM_QKV_WINT4_XETLA_DISPATCH(has_bias, fp64_valid);
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
    int64_t calib_gz,
    int8_t arch) {
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
                    .add_arch(arch)
                    .build();
  TORCH_CHECK(policy.fallback() == false, "mm bias int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    int8_t arch) {
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
                    .add_arch(arch)
                    .build();
  TORCH_CHECK(policy.fallback() == false, "mm int4: invalid gemm shape");
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
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int8_t fp64_valid =
      static_cast<int8_t>(Settings::I().has_2d_block_array(curDevID));
  auto policy = HGEMMXetla_INT4()
                    .add_matrix_out(output)
                    .add_matrix_inp(input_flat)
                    .add_matrix_wei(weight_flat)
                    .add_matrix_scl(weight_scl)
                    .add_matrix_zp(weight_zp)
                    .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
                    .add_calib_gz(calib_gz)
                    .add_arch(fp64_valid)
                    .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_resmul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& res,
    int64_t calib_gz) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
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
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL)
          .add_calib_gz(calib_gz)
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm resmul int4: invalid gemm shape");
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
  TORCH_CHECK(
      policy.fallback() == false, "mm bias gelu int4: invalid gemm shape");
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
  TORCH_CHECK(
      policy.fallback() == false,
      "mm bias resadd resadd int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input_, output);
}

static Tensor mm_low_bits(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const Tensor& bias,
    bool has_bias,
    const std::string& compute_dtype,
    const std::string& weight_dtype,
    int64_t calib_gz) {
  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int64_t fp64_valid =
      static_cast<int64_t>(Settings::I().has_2d_block_array(curDevID));
  return has_bias
      ? mm_bias_int4(
            input, weight, bias, weight_scl, weight_zp, calib_gz, fp64_valid)
      : mm_int4(input, weight, weight_scl, weight_zp, calib_gz, fp64_valid);
}

static Tensor mm_silu_mul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int8_t fp64_valid =
      static_cast<int8_t>(Settings::I().has_2d_block_array(curDevID));
  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL)
          .add_calib_gz(calib_gz)
          .add_arch(fp64_valid)
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_silu_mul_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  auto bias_flat = bias.flatten();
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int8_t fp64_valid =
      static_cast<int8_t>(Settings::I().has_2d_block_array(curDevID));
  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS)
          .add_epilogue(Tensor(), HGEMMXetla_INT4::EpilogueType::SILU)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_MUL)
          .add_calib_gz(calib_gz)
          .add_arch(fp64_valid)
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_add_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int8_t fp64_valid =
      static_cast<int8_t>(Settings::I().has_2d_block_array(curDevID));
  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD)
          .add_calib_gz(calib_gz)
          .add_arch(fp64_valid)
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
}

static Tensor mm_bias_add_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz,
    const Tensor& res) {
  auto input_flat = input.flatten(0, -2);
  auto weight_flat = weight.flatten(0, -2);
  auto res_flat = res.flatten(0, -2);
  auto bias_flat = bias.flatten();
  // a: m x k, b: k x n
  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  int m = input_flat.sizes()[0];
  int n = weight_flat.sizes()[1] * 2;
  int k = input_flat.sizes()[1];
  auto output = at::empty({m, n}, input.options());

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
  int8_t fp64_valid =
      static_cast<int8_t>(Settings::I().has_2d_block_array(curDevID));
  auto policy =
      HGEMMXetla_INT4()
          .add_matrix_out(output)
          .add_matrix_inp(input_flat)
          .add_matrix_wei(weight_flat)
          .add_matrix_scl(weight_scl)
          .add_matrix_zp(weight_zp)
          .add_epilogue(bias_flat, HGEMMXetla_INT4::EpilogueType::BIAS)
          .add_epilogue(res_flat, HGEMMXetla_INT4::EpilogueType::RES_ADD)
          .add_calib_gz(calib_gz)
          .add_arch(fp64_valid)
          .build();
  TORCH_CHECK(policy.fallback() == false, "mm silu int4: invalid gemm shape");
  policy.run();
  return resize_as_mat1(input, output);
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
  IPEX_OP_REGISTER("mm_resmul_int4.xpu", at::AtenIpexTypeXPU::mm_resmul_int4);
  IPEX_OP_REGISTER(
      "mm_bias_gelu_int4.xpu", at::AtenIpexTypeXPU::mm_bias_gelu_int4);
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd_int4.xpu",
      at::AtenIpexTypeXPU::mm_bias_resadd_resadd_int4);
  IPEX_OP_REGISTER("mm_low_bits.xpu", at::AtenIpexTypeXPU::mm_low_bits);
  IPEX_OP_REGISTER(
      "mm_silu_mul_int4.xpu", at::AtenIpexTypeXPU::mm_silu_mul_int4);
  IPEX_OP_REGISTER(
      "mm_bias_silu_mul_int4.xpu", at::AtenIpexTypeXPU::mm_bias_silu_mul_int4);
  IPEX_OP_REGISTER("mm_add_int4.xpu", at::AtenIpexTypeXPU::mm_add_int4);
  IPEX_OP_REGISTER(
      "mm_bias_add_int4.xpu", at::AtenIpexTypeXPU::mm_bias_add_int4);
}
} // namespace
#endif
