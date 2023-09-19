#include "XEGEMM.h"
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <xetla/hgemm.h>
#include "Linear.h"
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

#if defined(USE_XETLA)

namespace at {
namespace AtenIpexTypeXPU {

#define RECORD_ONEDNN_FUNCTION_IMPL(F)                     \
  char str__[100];                                         \
  sprintf(str__, "onednn_%s(%d, %d, %d)", "" #F, m, n, k); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

static Tensor mm_common(const Tensor& a, const Tensor& b) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  auto policy = HGEMM_XETLA()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_common)
    xpu::oneDNN::matmul(output, af, b, at::Tensor(), true, Attr());
  }
  return matmul_resize(a, output);
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
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(res, HGEMM_XETLA::EpilogueType::RES_ADD, res_factor)
          .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_resadd)
    auto att = Attr();
    Tensor binary =
        (res.dim() == 1 ? res.unsqueeze(0) : res) * Scalar(res_factor);
    att.append_post_binary(att.kind_with_binary_add, binary);
    xpu::oneDNN::matmul(output, af, b, at::Tensor(), true, att);
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
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(res0, HGEMM_XETLA::EpilogueType::RES_ADD, res0_factor)
          .add_epilogue(res1, HGEMM_XETLA::EpilogueType::RES_ADD, res1_factor)
          .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_resadd_resadd)
    auto att = Attr();
    Tensor binary0 =
        (res0.dim() == 1 ? res0.unsqueeze(0) : res0) * Scalar(res0_factor);
    att.append_post_binary(att.kind_with_binary_add, binary0);
    Tensor binary1 =
        (res1.dim() == 1 ? res1.unsqueeze(0) : res1) * Scalar(res1_factor);
    att.append_post_binary(att.kind_with_binary_add, binary1);
    xpu::oneDNN::matmul(output, af, b, at::Tensor(), true, att);
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
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
          .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_bias)
    xpu::oneDNN::matmul(
        output, af, b, bias * Scalar(bias_factor), true, Attr());
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
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
          .add_epilogue(res, HGEMM_XETLA::EpilogueType::RES_ADD, res_factor)
          .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_bias_resadd)
    auto att = Attr();
    Tensor binary =
        (res.dim() == 1 ? res.unsqueeze(0) : res) * Scalar(res_factor);
    att.append_post_binary(att.kind_with_binary_add, binary);
    xpu::oneDNN::matmul(output, af, b, bias * Scalar(bias_factor), true, att);
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
  auto policy =
      HGEMM_XETLA()
          .add_matrix_c(output)
          .add_matrix_a(a)
          .add_matrix_b(b)
          .add_epilogue(bias, HGEMM_XETLA::EpilogueType::BIAS, bias_factor)
          .add_epilogue(res0, HGEMM_XETLA::EpilogueType::RES_ADD, res0_factor)
          .add_epilogue(res1, HGEMM_XETLA::EpilogueType::RES_ADD, res1_factor)
          .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_bias_resadd_resadd)
    auto att = Attr();
    Tensor binary0 =
        (res0.dim() == 1 ? res0.unsqueeze(0) : res0) * Scalar(res0_factor);
    att.append_post_binary(att.kind_with_binary_add, binary0);
    Tensor binary1 =
        (res1.dim() == 1 ? res1.unsqueeze(0) : res1) * Scalar(res1_factor);
    att.append_post_binary(att.kind_with_binary_add, binary1);
    auto split_ms = hgemm_split_m(m, n);
    auto bias_ = bias * Scalar(bias_factor);
    for (auto data : split_ms) {
      auto newo = output.narrow(0, std::get<0>(data), std::get<1>(data));
      auto newa = af.narrow(0, std::get<0>(data), std::get<1>(data));
      xpu::oneDNN::matmul(newo, newa, b, bias_, true, att);
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
  auto policy = HGEMM_XETLA()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(res, HGEMM_XETLA::EpilogueType::RES_MUL)
                    .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_resmul)
    xpu::oneDNN::matmul(output, af, b, at::Tensor(), true, Attr());
    output = output * res.flatten(0, -2);
  }
  return matmul_resize(a, output);
}

static Tensor mm_silu(const Tensor& a, const Tensor& b) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());
  auto policy = HGEMM_XETLA()
                    .add_matrix_c(output)
                    .add_matrix_a(a)
                    .add_matrix_b(b)
                    .add_epilogue(Tensor(), HGEMM_XETLA::EpilogueType::SILU)
                    .build();
  if (policy.valid()) {
    policy.run();
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(mm_silu)
    xpu::oneDNN::matmul(output, af, b, at::Tensor(), true, Attr());
    at::AtenIpexTypeXPU::silu_out(output, output);
  }
  return matmul_resize(a, output);
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
    if (policy.valid()) {
      policy.run();
      return matmul_resize(input, output);
    }
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

  using namespace xpu::xetla;
  using scalar_t =
      decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
  auto& q = dpcppGetCurrentQueue();

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
  bool use_xetla = out0_valid && out1_valid && out2_valid && input_valid &&
      weight_valid && bias_valid && shape_valid;

  if (!use_xetla) {
    auto wq = weight[0];
    auto wk = weight[1];
    auto wv = weight[2];
    if (!has_bias) {
      at::AtenIpexTypeXPU::mm_out(input, wq, out0);
      at::AtenIpexTypeXPU::mm_out(input, wk, out1);
      at::AtenIpexTypeXPU::mm_out(input, wv, out2);
    } else {
      at::AtenIpexTypeXPU::addmm_out(
          bias_.value()[0], input, wq, at::Scalar(1), at::Scalar(1), out0_);
      at::AtenIpexTypeXPU::addmm_out(
          bias_.value()[1], input, wk, at::Scalar(1), at::Scalar(1), out1_);
      at::AtenIpexTypeXPU::addmm_out(
          bias_.value()[2], input, wv, at::Scalar(1), at::Scalar(1), out2_);
    }
  } else {
    char str__[100];
    if (!has_bias) {
      sprintf(str__, "hgemm_qkv(%d, %d, %d)", m, n, k);
      RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));
      hgemm_qkv(
          q,
          reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
          m,
          n,
          k,
          is_b_row_major);
    } else {
      sprintf(str__, "hgemm_qkv_bias(%d, %d, %d)", m, n, k);
      RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));
      hgemm_qkv_bias(
          q,
          reinterpret_cast<sycl::half*>(out0.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(out1.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(out2.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(input.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(weight.data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(bias_.value().data_ptr<scalar_t>()),
          m,
          n,
          k,
          is_b_row_major);
    }
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

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("mm_.xpu", at::AtenIpexTypeXPU::mm_common);
  IPEX_OP_REGISTER("mm_common.xpu", at::AtenIpexTypeXPU::mm_common);
  IPEX_OP_REGISTER("mm_resadd.xpu", at::AtenIpexTypeXPU::mm_resadd);
  IPEX_OP_REGISTER(
      "mm_resadd_resadd.xpu", at::AtenIpexTypeXPU::mm_resadd_resadd);
  IPEX_OP_REGISTER("mm_bias.xpu", at::AtenIpexTypeXPU::mm_bias);
  IPEX_OP_REGISTER("mm_bias_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd);
  IPEX_OP_REGISTER(
      "mm_bias_resadd_resadd.xpu", at::AtenIpexTypeXPU::mm_bias_resadd_resadd);

  IPEX_OP_REGISTER("mm_resmul.xpu", at::AtenIpexTypeXPU::mm_resmul);
  IPEX_OP_REGISTER("mm_silu.xpu", at::AtenIpexTypeXPU::mm_silu);
  IPEX_OP_REGISTER("matmul_gelu.xpu", at::AtenIpexTypeXPU::matmul_gelu);

  IPEX_OP_REGISTER("mm_qkv_out.xpu", at::AtenIpexTypeXPU::mm_qkv_out);
  IPEX_OP_REGISTER("mm_qkv.xpu", at::AtenIpexTypeXPU::mm_qkv);
}
} // namespace

#endif
