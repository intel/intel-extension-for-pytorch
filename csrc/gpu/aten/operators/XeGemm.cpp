#include "XeGemm.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <iostream>
#include "Blas.h"
#include "Linear.h"
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC) // XeGemm only supports PVC
#include "xetla/hgemm.h"
#endif

namespace at {
namespace AtenIpexTypeXPU {

using namespace torch_ipex::xpu::xetla;

inline bool fp64_valid() {
  DeviceId curDevID = at::xpu::current_device();
  return Settings::I().has_2d_block_array(curDevID);
}

#ifdef USE_PTI
#define RECORD_ONEDNN_FUNCTION_IMPL(F)                     \
  char str__[100];                                         \
  sprintf(str__, "onednn_%s(%d, %d, %d)", "" #F, m, n, k); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define RECORD_XETLA_FUNCTION_IMPL(F)                     \
  char str__[100];                                        \
  sprintf(str__, "xetla_%s(%d, %d, %d)", "" #F, m, n, k); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define RECORD_XETLA_FUNCTION_IMPLG(F)                                 \
  char str__[100];                                                     \
  sprintf(str__, "xetla_%s(%d, %d, %d, g=%d)", "" #F, m, n, k, group); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define RECORD_FUNC_IMPL(F)                         \
  char str__[100];                                  \
  sprintf(str__, "%s(%d, %d, %d)", "" #F, m, n, k); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define RECORD_FUNC_IMPLG(F)                                     \
  char str__[100];                                               \
  sprintf(str__, "%s(%d, %d, %d, g=%d)", "" #F, m, n, k, group); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));
#else
#define RECORD_ONEDNN_FUNCTION_IMPL(F)
#define RECORD_XETLA_FUNCTION_IMPL(F)
#define RECORD_XETLA_FUNCTION_IMPLG(F)
#define RECORD_FUNC_IMPL(F)
#define RECORD_FUNC_IMPLG(F)
#endif

static void mm_common_out(const Tensor& a, const Tensor& b, Tensor& out) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy = HGEMM_XETLA() //
                    .add_matrix_c(out)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_common)
    bool is_fused;
    Attr attr;
    impl::matmul_fusion_variants(out, a, b, true, attr, is_fused = false);
  }
}

static Tensor mm_common(const Tensor& a, const Tensor& b) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  auto out = at::empty({m, n}, a.options());
  mm_common_out(a, b, out);
  return matmul_resize(a, out);
}

static Tensor mm_resadd(
    const Tensor& a,
    const Tensor& b,
    const Tensor& res,
    const double res_factor) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b, res);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(res, HGEMM_XETLA::EpilogueType::RES_ADD, res_factor)
          .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_resadd)
    bool is_fused;
    Attr attr;
    attr.append_scale_binary(attr.kind_with_binary_add, res, float(res_factor));

    output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
    if (!is_fused) {
      output += at::mul(res, Scalar(res_factor));
    }
  }
  return matmul_resize(a, output);
}

static Tensor mm_resadd_resadd(
    const Tensor& a,
    const Tensor& b,
    const Tensor& res0,
    const double res0_factor,
    const Tensor& res1,
    const double res1_factor) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b, res0, res1);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(res0, HGEMM_XETLA::EpilogueType::RES_ADD, res0_factor)
          .add_epilogue(res1, HGEMM_XETLA::EpilogueType::RES_ADD, res1_factor)
          .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_resadd_resadd)
    bool is_fused;
    Attr attr;
    attr.append_scale_binary(
        attr.kind_with_binary_add, res0, float(res0_factor));
    attr.append_scale_binary(
        attr.kind_with_binary_add, res1, float(res1_factor));
    output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
    if (!is_fused) {
      output += at::mul(res0, Scalar(res0_factor)) +
          at::mul(res1, Scalar(res1_factor));
    }
  }
  return matmul_resize(a, output);
}

static Tensor mm_bias(
    const Tensor& a,
    const Tensor& b,
    const Tensor& bias,
    const double bias_factor) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b, bias);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
          .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_bias)
    bool is_fused;
    Attr attr;
    attr.append_scale_binary(
        attr.kind_with_binary_add, bias, float(bias_factor));
    output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
    if (!is_fused) {
      output += at::mul(bias, Scalar(bias_factor));
    }
  }
  return matmul_resize(a, output);
}

static Tensor mm_bias_resadd(
    const Tensor& a,
    const Tensor& b,
    const Tensor& bias,
    const double bias_factor,
    const Tensor& res,
    const double res_factor) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b, bias, res);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
          .add_epilogue(res, HGEMM_XETLA::EpilogueType::RES_ADD, res_factor)
          .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_bias_resadd)
    bool is_fused;
    Attr attr;
    attr.append_scale_binary(
        attr.kind_with_binary_add, bias, float(bias_factor));
    attr.append_scale_binary(attr.kind_with_binary_add, res, float(res_factor));
    output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
    if (!is_fused) {
      output +=
          at::mul(bias, Scalar(bias_factor)) + at::mul(res, Scalar(res_factor));
    }
  }
  return matmul_resize(a, output);
}

static Tensor mm_bias_resadd_resadd(
    const Tensor& a,
    const Tensor& b,
    const Tensor& bias,
    const double bias_factor,
    const Tensor& res0,
    const double res0_factor,
    const Tensor& res1,
    const double res1_factor) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng = choose_compute_eng(
      torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b, bias, res0, res1);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
          .add_epilogue(res0, HGEMM_XETLA::EpilogueType::RES_ADD, res0_factor)
          .add_epilogue(res1, HGEMM_XETLA::EpilogueType::RES_ADD, res1_factor)
          .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_bias_resadd_resadd)
    bool is_fused;
    Attr attr;
    attr.append_scale_binary(
        attr.kind_with_binary_add, bias, float(bias_factor));
    attr.append_scale_binary(
        attr.kind_with_binary_add, res0, float(res0_factor));
    attr.append_scale_binary(
        attr.kind_with_binary_add, res1, float(res1_factor));
    output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
    if (!is_fused) {
      output += at::mul(bias, Scalar(bias_factor)) +
          at::mul(res0, Scalar(res0_factor)) +
          at::mul(res1, Scalar(res1_factor));
    }
  }
  return matmul_resize(a, output);
}

static Tensor mm_resmul(const Tensor& a, const Tensor& b, const Tensor& res) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b, res);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy = HGEMM_XETLA()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(res, HGEMM_XETLA::EpilogueType::RES_MUL)
                    .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_resmul)
    bool is_fused;
    Attr attr;
    attr.append_post_binary(attr.kind_with_binary_mul, res);

    output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
    if (!is_fused) {
      output = output * res.flatten(0, -2);
    }
  }
  return matmul_resize(a, output);
}

static Tensor mm_silu(const Tensor& a, const Tensor& b) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  auto policy = HGEMM_XETLA()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(Tensor(), HGEMM_XETLA::EpilogueType::SILU)
                    .build();
  GemmStatus status = GemmStatus::kError;
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (xetla_valid && policy.valid()) {
    status = policy.run();
  }
#endif
  if (status != GemmStatus::kSuccess) {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_silu)
    auto result = matmul_silu(a, b);
    return matmul_resize(a, result);
  }
  return matmul_resize(a, output);
}

Tensor matmul_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const double bias_factor) {
  int m = input.flatten(0, -2).sizes()[0];
  int n = weight.sizes()[1];
  int k = weight.sizes()[0];
  auto output = at::empty({m, n}, input.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, input);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  GemmStatus status = GemmStatus::kError;
  if (bias.has_value()) {
    auto policy =
        HGEMM_XETLA()
            .add_matrix_c(output)
            .add_matrix_a(input)
            .add_matrix_b(weight)
            .add_epilogue(
                bias.value(), HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
            .add_epilogue(Tensor(), HGEMM_XETLA::EpilogueType::RELU)
            .build();
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
    if (xetla_valid && policy.valid()) {
      status = policy.run();
      if (status == GemmStatus::kSuccess) {
        return matmul_resize(input, output);
      }
    }
#endif
  }

  RECORD_ONEDNN_FUNCTION_IMPL(matmul_relu)
  auto weight_ = weight.transpose(0, 1);
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(1.f, 0.f, 0.f, attr.kind_with_relu);
    return attr;
  };
  auto input_flatten = input.flatten(0, -2);
  linear_wrapper.call(output, input_flatten, weight_, bias, post_op);

  return matmul_resize(input, output);
}

Tensor matmul_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const double bias_factor,
    c10::string_view approximate) {
  int m = input.flatten(0, -2).sizes()[0];
  int n = weight.sizes()[1];
  int k = weight.sizes()[0];
  auto output = at::empty({m, n}, input.options());
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, input);
  bool compute_eng_valid = (real_eng == torch_ipex::xpu::COMPUTE_ENG::XETLA);
  bool xetla_valid = fp64_valid() && compute_eng_valid;
  if (bias.has_value() && approximate == "tanh") {
    auto policy =
        HGEMM_XETLA()
            .add_matrix_c(output)
            .add_matrix_a(input)
            .add_matrix_b(weight)
            .add_epilogue(
                bias.value(), HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
            .add_epilogue(Tensor(), HGEMM_XETLA::EpilogueType::GELU)
            .build();
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
    if (xetla_valid && policy.valid()) {
      auto status = policy.run();
      if (status == GemmStatus::kSuccess)
        return matmul_resize(input, output);
    }
#endif
  }
  RECORD_ONEDNN_FUNCTION_IMPL(matmul_gelu)
  auto weight_ = weight.transpose(0, 1);
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    algorithm algo;
    if (approximate == "none") {
      algo = attr.kind_with_gelu_erf;
    } else if (approximate == "tanh") {
      algo = attr.kind_with_gelu_tanh;
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
    }
    attr.append_post_eltwise(1.0f, 0.0f, 0.0f, algo);
    return attr;
  };

  auto input_flatten = input.flatten(0, -2);
  auto split_ms = hgemm_split_m(m, n);

  if (bias.has_value() && bias_factor != 1.0f) {
    auto bias_ = bias.value() * Scalar(bias_factor);
    for (auto data : split_ms) {
      auto newo = output.narrow(0, std::get<0>(data), std::get<1>(data));
      auto newa = input_flatten.narrow(0, std::get<0>(data), std::get<1>(data));
      linear_wrapper.call(newo, newa, weight_, bias_, post_op);
    }
  } else {
    for (auto data : split_ms) {
      auto newo = output.narrow(0, std::get<0>(data), std::get<1>(data));
      auto newa = input_flatten.narrow(0, std::get<0>(data), std::get<1>(data));
      linear_wrapper.call(newo, newa, weight_, bias, post_op);
    }
  }

  return matmul_resize(input, output);
}

static size_t get_acc_size(uint32_t matrix_m, uint32_t matrix_n) {
  return matrix_m * matrix_n;
};

static uint32_t find_ks_coop_num_y(uint32_t slm_kslicing, uint32_t sg_m) {
  uint32_t ks_coop_num_y = sg_m;
  while (slm_kslicing % sg_m != 0) {
    ks_coop_num_y = slm_kslicing % sg_m;
    slm_kslicing = sg_m;
    sg_m = ks_coop_num_y;
  }
  return ks_coop_num_y;
}

static size_t get_cnt_size(
    uint32_t matrix_m,
    uint32_t matrix_n,
    uint32_t wg_m,
    uint32_t wg_n,
    uint32_t sg_m,
    uint32_t sg_n,
    uint32_t slm_kslicing) {
  // return matrix_m * matrix_n;
  size_t group_range_m = (matrix_m + wg_m - 1) / wg_m;
  size_t group_range_n = (matrix_n + wg_n - 1) / wg_n;

  uint32_t wg_size_x = (wg_m + sg_m - 1) / sg_m;
  uint32_t wg_size_y = (wg_n + sg_n - 1) / sg_n;
  // uint32_t ks_coop_num_y = gcd<slm_kslicing, sg_m>::value;
  uint32_t ks_coop_num_y = find_ks_coop_num_y(slm_kslicing, sg_m);
  uint32_t coop_remain_num_x = slm_kslicing / ks_coop_num_y;
  bool has_redundant_wg = (coop_remain_num_x * 16) > sg_n;
  uint32_t tile_size_y = sg_m / ks_coop_num_y;
  uint32_t tile_size_x = has_redundant_wg ? 16 : sg_n / coop_remain_num_x;
  uint32_t ks_coop_num_x = sg_n / tile_size_x;

  uint32_t counter_size = 8;
  return group_range_m * group_range_n * wg_size_x * wg_size_y * ks_coop_num_y *
      ks_coop_num_x * counter_size;
}

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
static void get_acc_and_cnt_tensor(
    const Tensor& input_,
    uint32_t m_,
    uint32_t n_,
    uint32_t k_,
    bool is_b_row_major_,
    Tensor& acc_tensor_,
    Tensor& cnt_tensor_) {
  int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
  if (policy_id == -1) {
    acc_tensor_ = at::AtenIpexTypeXPU::empty(
        {0}, input_.options().dtype(at::kFloat), c10::nullopt);
    cnt_tensor_ = at::AtenIpexTypeXPU::empty(
        {0}, input_.options().dtype(at::kByte), c10::nullopt);
    return;
  }

  auto policy_config = hgemm_policy_traits[policy_id];
  int wg_m = policy_config.wg_m_;
  int wg_n = policy_config.wg_n_;
  int sg_m = policy_config.sg_m_;
  int sg_n = policy_config.sg_n_;
  int slm_ks = policy_config.slm_ks_;
  size_t acc_size = get_acc_size(m_, n_);
  size_t cnt_size = get_cnt_size(m_, n_, wg_m, wg_n, sg_m, sg_n, slm_ks);

  acc_tensor_ = at::AtenIpexTypeXPU::empty(
      {acc_size}, input_.options().dtype(at::kFloat), c10::nullopt);
  cnt_tensor_ = at::AtenIpexTypeXPU::empty(
      {cnt_size}, input_.options().dtype(at::kByte), c10::nullopt);
}
#endif

static void mm_qkv_out(
    const Tensor& input_,
    const Tensor& weight,
    const optional<Tensor>& bias_,
    Tensor& out0_,
    Tensor& out1_,
    Tensor& out2_) {
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

  using namespace torch_ipex::xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& q = dpcppGetCurrentQueue();

  DeviceId curDevID = at::xpu::current_device();
  bool fp64_valid = Settings::I().has_2d_block_array(curDevID);
  bool out0_valid =
      reinterpret_cast<uint64_t>(out0.data_ptr<scalar_t>()) % 8 == 0;
  bool out1_valid =
      reinterpret_cast<uint64_t>(out1.data_ptr<scalar_t>()) % 8 == 0;
  bool out2_valid =
      reinterpret_cast<uint64_t>(out2.data_ptr<scalar_t>()) % 8 == 0;
  bool input_valid =
      reinterpret_cast<uint64_t>(input.data_ptr<scalar_t>()) % 8 == 0;
  bool weight_valid =
      reinterpret_cast<uint64_t>(weight.data_ptr<scalar_t>()) % 8 == 0;
  bool bias_valid = true;
  if (has_bias) {
    bias_valid =
        reinterpret_cast<uint64_t>(bias_.value().data_ptr<scalar_t>()) % 8 == 0;
  }
  bool shape_valid = k % 4 == 0 && n % 4 == 0;
  bool xetla_valid = fp64_valid && out0_valid && out1_valid && out2_valid &&
      input_valid && weight_valid && bias_valid && shape_valid;

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (dpcppGetDeviceHasXMX() && xetla_valid) {
    if (!has_bias) {
      Tensor acc_tensor_, cnt_tensor_;
      get_acc_and_cnt_tensor(
          input_, m, n, k, is_b_row_major, acc_tensor_, cnt_tensor_);

      int policy_id = hgemm_qkv_find_policy_id(m, n, k, is_b_row_major);
      if (policy_id >= 0) {
        RECORD_XETLA_FUNCTION_IMPL(hgemm_qkv);
        auto cgfs = hgemm_qkv(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
            reinterpret_cast<float*>(acc_tensor_.data_ptr<float>()),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m,
            n,
            k,
            is_b_row_major);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return;
      }
    } else {
      Tensor acc_tensor_, cnt_tensor_;
      get_acc_and_cnt_tensor(
          input_, m, n, k, is_b_row_major, acc_tensor_, cnt_tensor_);

      int policy_id = hgemm_qkv_find_policy_id(m, n, k, is_b_row_major);
      if (policy_id >= 0) {
        RECORD_XETLA_FUNCTION_IMPL(hgemm_qkv_bias);
        auto cgfs = hgemm_qkv_bias(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(bias_.value().data_ptr<scalar_t>()),
            reinterpret_cast<float*>(acc_tensor_.data_ptr<float>()),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m,
            n,
            k,
            is_b_row_major);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return;
      }
    }
  }
#endif

  auto wq = weight[0];
  auto wk = weight[1];
  auto wv = weight[2];
  if (!has_bias) {
    RECORD_FUNC_IMPL(hgemm_mm_out);
    at::AtenIpexTypeXPU::mm_out(input, wq, out0);
    at::AtenIpexTypeXPU::mm_out(input, wk, out1);
    at::AtenIpexTypeXPU::mm_out(input, wv, out2);
  } else {
    RECORD_FUNC_IMPL(hgemm_addmm_out);
    at::AtenIpexTypeXPU::addmm_out(
        bias_.value()[0], input, wq, at::Scalar(1), at::Scalar(1), out0);
    at::AtenIpexTypeXPU::addmm_out(
        bias_.value()[1], input, wk, at::Scalar(1), at::Scalar(1), out1);
    at::AtenIpexTypeXPU::addmm_out(
        bias_.value()[2], input, wv, at::Scalar(1), at::Scalar(1), out2);
  }
}

static void mm_qkv_group_out(
    const Tensor& input_,
    const Tensor& weight,
    const optional<Tensor>& bias_,
    Tensor& out0_,
    Tensor& out1_,
    Tensor& out2_) {
  auto input = input_.flatten(0, -2);
  auto out0 = out0_.flatten(0, -2);
  auto out1 = out1_.flatten(0, -2);
  auto out2 = out2_.flatten(0, -2);
  // input: [bs * seq_len, hidden_size]
  // weight: [hidden_size, num_kv_head, num_head//num_kv_head + 2, head_dim]
  // bias: [num_kv_head, num_head//num_kv_head + 2, head_dim]
  // out0: [bs * seq_len, num_kv_head * num_head//num_kv_head * head_dim]
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 4);
  TORCH_CHECK(out0.dim() == 2 && out1.dim() == 2 && out2.dim() == 2);
  int m = input.sizes()[0];
  int k = input.sizes()[1];
  int num_kv_head = weight.sizes()[1];
  int group = weight.sizes()[2];
  int head_dim = weight.sizes()[3];
  int n = num_kv_head * head_dim;

  bool has_bias = bias_.has_value();
  if (has_bias) {
    auto bias = bias_.value();
    TORCH_CHECK(
        bias.dim() == 3 && bias.sizes()[0] == num_kv_head &&
        bias.sizes()[1] == group && bias.sizes()[2] == head_dim);
  }

  TORCH_CHECK(
      out0.sizes()[0] == m && out0.sizes()[1] == n * (group - 2) &&
      out1.sizes()[0] == m && out1.sizes()[1] == n && out2.sizes()[0] == m &&
      out2.sizes()[1] == n);

  bool is_a_contiguous = input.is_contiguous();
  bool is_b_row_major = weight.is_contiguous();
  bool is_b_col_major = weight.transpose(1, 2).is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(input.scalar_type() == kHalf && weight.scalar_type() == kHalf);

  using namespace torch_ipex::xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& queue = dpcppGetCurrentQueue();

  DeviceId curDevID = at::xpu::current_device();
  bool fp64_valid = Settings::I().has_2d_block_array(curDevID);
  bool out0_valid =
      reinterpret_cast<uint64_t>(out0.data_ptr<scalar_t>()) % 8 == 0;
  bool out1_valid =
      reinterpret_cast<uint64_t>(out1.data_ptr<scalar_t>()) % 8 == 0;
  bool out2_valid =
      reinterpret_cast<uint64_t>(out2.data_ptr<scalar_t>()) % 8 == 0;
  bool input_valid =
      reinterpret_cast<uint64_t>(input.data_ptr<scalar_t>()) % 8 == 0;
  bool weight_valid =
      reinterpret_cast<uint64_t>(weight.data_ptr<scalar_t>()) % 8 == 0;
  bool bias_valid = true;
  if (has_bias) {
    bias_valid =
        reinterpret_cast<uint64_t>(bias_.value().data_ptr<scalar_t>()) % 8 == 0;
  }
  bool shape_valid = k % 4 == 0 && n % 4 == 0;
  bool xetla_valid = fp64_valid && out0_valid && out1_valid && out2_valid &&
      input_valid && weight_valid && bias_valid && shape_valid;

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (dpcppGetDeviceHasXMX() && xetla_valid) {
    Tensor acc_tensor_, cnt_tensor_;
    get_acc_and_cnt_tensor(
        input_, m, n, k, is_b_row_major, acc_tensor_, cnt_tensor_);
    if (!has_bias) {
      int policy_id = hgemm_qkv_find_policy_id(m, head_dim, k, is_b_row_major);
      if (policy_id >= 0) {
        RECORD_XETLA_FUNCTION_IMPLG(hgemm_qkv_group);
        auto cgfs = hgemm_qkv_group(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
            reinterpret_cast<float*>(acc_tensor_.data_ptr<float>()),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m,
            n,
            k,
            num_kv_head,
            group,
            head_dim,
            is_b_row_major);
        DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
        return;
      }
    } else {
      int policy_id = hgemm_qkv_find_policy_id(m, head_dim, k, is_b_row_major);
      if (policy_id >= 0) {
        RECORD_XETLA_FUNCTION_IMPLG(hgemm_qkv_group_bias);
        auto cgfs = hgemm_qkv_group_bias(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(bias_.value().data_ptr<scalar_t>()),
            reinterpret_cast<float*>(acc_tensor_.data_ptr<float>()),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m,
            n,
            k,
            num_kv_head,
            group,
            head_dim,
            is_b_row_major);
        DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
        return;
      }
    }
  }
#endif

  using namespace at::indexing;
  out0 = out0.view({m, num_kv_head, group - 2, head_dim});
  out1 = out1.view({m, num_kv_head, head_dim});
  out2 = out2.view({m, num_kv_head, head_dim});
  if (!has_bias) {
    RECORD_FUNC_IMPLG(hgemm_qkv_group_mm_common);
    auto out =
        mm_common(input, weight.view({k, num_kv_head * group * head_dim}));
    out = out.view({m, num_kv_head, group, head_dim});
    out0.index_put_(
        {"..."},
        out.index({Slice(), Slice(), Slice(None, group - 2), Slice()}));
    out1.index_put_({"..."}, out.index({Slice(), Slice(), group - 2, Slice()}));
    out2.index_put_({"..."}, out.index({Slice(), Slice(), group - 1, Slice()}));
  } else {
    RECORD_FUNC_IMPLG(hgemm_qkv_group_mm_bias);
    auto out = mm_bias(
        input,
        weight.view({k, num_kv_head * group * head_dim}),
        bias_.value().view({1, num_kv_head * group * head_dim}),
        1.0);
    out = out.view({m, num_kv_head, group, head_dim});
    out0.index_put_(
        {"..."},
        out.index({Slice(), Slice(), Slice(None, group - 2), Slice()}));
    out1.index_put_({"..."}, out.index({Slice(), Slice(), group - 2, Slice()}));
    out2.index_put_({"..."}, out.index({Slice(), Slice(), group - 1, Slice()}));
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

#undef RECORD_ONEDNN_FUNCTION_IMPL
#undef RECORD_XETLA_FUNCTION_IMPL
#undef RECORD_XETLA_FUNCTION_IMPLG
#undef RECORD_FUNC_IMPL
#undef RECORD_FUNC_IMPLG

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("mm_.xpu", at::AtenIpexTypeXPU::mm_common);
  IPEX_OP_REGISTER("mm_common.xpu", at::AtenIpexTypeXPU::mm_common);
  IPEX_OP_REGISTER("mm_common_out.xpu", at::AtenIpexTypeXPU::mm_common_out);
  IPEX_OP_REGISTER("mm_resadd.xpu", at::AtenIpexTypeXPU::mm_resadd);
  IPEX_OP_REGISTER(
      "mm_resadd_resadd.xpu", at::AtenIpexTypeXPU::mm_resadd_resadd);
  IPEX_OP_REGISTER("mm_bias.xpu", at::AtenIpexTypeXPU::mm_bias);
  IPEX_OP_REGISTER("mm_bias_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd);
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd_resadd);

  IPEX_OP_REGISTER("mm_resmul.xpu", at::AtenIpexTypeXPU::mm_resmul);
  IPEX_OP_REGISTER("mm_silu.xpu", at::AtenIpexTypeXPU::mm_silu);
  IPEX_OP_REGISTER("matmul_relu.xpu", at::AtenIpexTypeXPU::matmul_relu);
  IPEX_OP_REGISTER("matmul_gelu.xpu", at::AtenIpexTypeXPU::matmul_gelu);

  IPEX_OP_REGISTER("mm_qkv_out.xpu", at::AtenIpexTypeXPU::mm_qkv_out);
  IPEX_OP_REGISTER(
      "mm_qkv_group_out.xpu", at::AtenIpexTypeXPU::mm_qkv_group_out);
  IPEX_OP_REGISTER("mm_qkv.xpu", at::AtenIpexTypeXPU::mm_qkv);
}
} // namespace
