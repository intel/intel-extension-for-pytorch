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
#include <ATen/autocast_mode.h>

namespace at {
namespace AtenIpexTypeXPU {

using namespace torch_ipex::xpu::xetla;
using autocast::cached_cast;

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

static void mm_common_out(const Tensor& a, const Tensor& b, Tensor& out) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<0> hgemm_common(arch_tag);
    auto policy = hgemm_common.add_operands(out, a, b).build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess)
        return;
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_common_out)
  bool is_fused;
  Attr attr;
  impl::matmul_fusion_variants(out, a, b, true, attr, is_fused = false);
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b, res)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<1> hgemm_res(arch_tag);
    auto policy = hgemm_res.add_operands(output, a, b)
                      .add_epilogue(res, EpilogueType::RES_ADD, res_factor)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_resadd)
  bool is_fused;
  Attr attr;
  attr.append_scale_binary(attr.kind_with_binary_add, res, float(res_factor));

  output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
  if (!is_fused) {
    output += at::mul(res, Scalar(res_factor));
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b, res0)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<2> hgemm_res_res(arch_tag);
    auto policy = hgemm_res_res.add_operands(output, a, b)
                      .add_epilogue(res0, EpilogueType::RES_ADD, res0_factor)
                      .add_epilogue(res1, EpilogueType::RES_ADD, res1_factor)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_resadd_resadd)
  bool is_fused;
  Attr attr;
  attr.append_scale_binary(attr.kind_with_binary_add, res0, float(res0_factor));
  attr.append_scale_binary(attr.kind_with_binary_add, res1, float(res1_factor));
  output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
  if (!is_fused) {
    output +=
        at::mul(res0, Scalar(res0_factor)) + at::mul(res1, Scalar(res1_factor));
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b, bias)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<1> hgemm_bias(arch_tag);
    auto policy = hgemm_bias.add_operands(output, a, b)
                      .add_epilogue(bias, EpilogueType::BIAS, bias_factor)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_bias)
  bool is_fused;
  Attr attr;
  attr.append_scale_binary(attr.kind_with_binary_add, bias, float(bias_factor));
  output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
  if (!is_fused) {
    output += at::mul(bias, Scalar(bias_factor));
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b, res)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<2> hgemm_bias_res(arch_tag);
    auto policy = hgemm_bias_res.add_operands(output, a, b)
                      .add_epilogue(bias, EpilogueType::BIAS, bias_factor)
                      .add_epilogue(res, EpilogueType::RES_ADD, res_factor)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_bias_resadd)
  bool is_fused;
  Attr attr;
  attr.append_scale_binary(attr.kind_with_binary_add, bias, float(bias_factor));
  attr.append_scale_binary(attr.kind_with_binary_add, res, float(res_factor));
  output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
  if (!is_fused) {
    output +=
        at::mul(bias, Scalar(bias_factor)) + at::mul(res, Scalar(res_factor));
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b, res0)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<3> hgemm_bias_res_res(arch_tag);
    auto policy = hgemm_bias_res_res.add_operands(output, a, b)
                      .add_epilogue(bias, EpilogueType::BIAS, bias_factor)
                      .add_epilogue(res0, EpilogueType::RES_ADD, res0_factor)
                      .add_epilogue(res1, EpilogueType::RES_ADD, res1_factor)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_bias_resadd_resadd)
  bool is_fused;
  Attr attr;
  attr.append_scale_binary(attr.kind_with_binary_add, bias, float(bias_factor));
  attr.append_scale_binary(attr.kind_with_binary_add, res0, float(res0_factor));
  attr.append_scale_binary(attr.kind_with_binary_add, res1, float(res1_factor));
  output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
  if (!is_fused) {
    output += at::mul(bias, Scalar(bias_factor)) +
        at::mul(res0, Scalar(res0_factor)) + at::mul(res1, Scalar(res1_factor));
  }
  return matmul_resize(a, output);
}

static Tensor mm_resmul(const Tensor& a, const Tensor& b, const Tensor& res) {
  auto af = a.flatten(0, -2);
  int m = af.sizes()[0];
  int n = b.sizes()[1];
  int k = b.sizes()[0];
  auto output = at::empty({m, n}, a.options());

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b, res)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<1> hgemm_res(arch_tag);
    auto policy = hgemm_res.add_operands(output, a, b)
                      .add_epilogue(res, EpilogueType::RES_MUL)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_resmul)
  bool is_fused;
  Attr attr;
  attr.append_post_binary(attr.kind_with_binary_mul, res);

  output = impl::matmul_fusion_variants(output, a, b, true, attr, is_fused);
  if (!is_fused) {
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(a, b)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<1> hgemm_silu(arch_tag);
    auto policy = hgemm_silu.add_operands(output, a, b)
                      .add_epilogue(Tensor(), EpilogueType::SILU)
                      .build();
    if (policy.valid()) {
      if (policy.run() == GemmStatus::kSuccess) {
        return matmul_resize(a, output);
      }
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_silu)
  output = matmul_silu(a, b);
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (bias.has_value()) {
    if (hgemm_xetla_valid(input, weight)) {
      gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
      HGEMM_XETLA<2> hgemm_bias_relu(arch_tag);
      auto policy =
          hgemm_bias_relu.add_operands(output, input, weight)
              .add_epilogue(bias.value(), EpilogueType::BIAS, bias_factor)
              .add_epilogue(Tensor(), EpilogueType::RELU)
              .build();
      if (policy.valid()) {
        if (policy.run() == GemmStatus::kSuccess) {
          return matmul_resize(input, output);
        }
      }
    }
  }
#endif

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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (bias.has_value() && approximate == "tanh") {
    if (hgemm_xetla_valid(input, weight)) {
      gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
      HGEMM_XETLA<2> hgemm_bias_gelu(arch_tag);
      auto policy =
          hgemm_bias_gelu.add_operands(output, input, weight)
              .add_epilogue(bias.value(), EpilogueType::BIAS, bias_factor)
              .add_epilogue(Tensor(), EpilogueType::GELU)
              .build();
      if (policy.valid()) {
        if (policy.run() == GemmStatus::kSuccess) {
          return matmul_resize(input, output);
        }
      }
    }
  }
#endif

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
    RECORD_ONEDNN_FUNCTION_IMPL(matmul_bias_gelu)
    for (auto data : split_ms) {
      auto newo = output.narrow(0, std::get<0>(data), std::get<1>(data));
      auto newa = input_flatten.narrow(0, std::get<0>(data), std::get<1>(data));
      linear_wrapper.call(newo, newa, weight_, bias_, post_op);
    }
  } else {
    RECORD_ONEDNN_FUNCTION_IMPL(matmul_gelu)
    for (auto data : split_ms) {
      auto newo = output.narrow(0, std::get<0>(data), std::get<1>(data));
      auto newa = input_flatten.narrow(0, std::get<0>(data), std::get<1>(data));
      linear_wrapper.call(newo, newa, weight_, bias, post_op);
    }
  }
  return matmul_resize(input, output);
}

static inline bool check_qkv_align(
    const Tensor& out0,
    const Tensor& out1,
    const Tensor& out2,
    const Tensor& input,
    const Tensor& weight) {
  return (
      ptr_align64(out0.data_ptr()) && ptr_align64(out1.data_ptr()) &&
      ptr_align64(out2.data_ptr()) && ptr_align64(input.data_ptr()) &&
      ptr_align64(weight.data_ptr()));
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  bool is_a_contiguous = input.is_contiguous();
  bool is_b_row_major = weight.is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(input.scalar_type() == kHalf && weight.scalar_type() == kHalf);

  using namespace torch_ipex::xpu::xetla;

  gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
  bool align_valid = check_qkv_align(out0, out1, out2, input, weight);
  bool bias_valid = true;
  if (has_bias) {
    bias_valid = ptr_align64(bias_.value().data_ptr());
  }
  bool shape_valid = ((k & 0x3) == 0) && ((n & 0x3) == 0);
  bool xetla_valid = align_valid && bias_valid && shape_valid;
  if (xetla_valid) {
    int policy_id = hgemm_qkv_find_policy_id(m, n, k, is_b_row_major, arch_tag);
    if (policy_id >= 0) {
      auto& q = dpcppGetCurrentQueue();
      auto input_ptr = c10::MaybeOwned<Tensor>::borrowed(input_);
#if 0
    Tensor acc_tensor_, cnt_tensor_;
    float* acc_ptr = get_acc_and_cnt_tensor(
        input_ptr, m, n, k, policy_id, acc_tensor_, cnt_tensor_);
    uint32_t* cnt_ptr = (acc_ptr == nullptr)
        ? (uint32_t*)nullptr
        : reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr());
#endif
      float* acc_ptr = (float*)nullptr;
      uint32_t* cnt_ptr = (uint32_t*)nullptr;
      if (!has_bias) {
        RECORD_XETLA_FUNCTION_IMPL(hgemm_qkv);
        auto cgfs = hgemm_qkv(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr()),
            reinterpret_cast<sycl::half*>(out1.data_ptr()),
            reinterpret_cast<sycl::half*>(out2.data_ptr()),
            reinterpret_cast<sycl::half*>(input.data_ptr()),
            reinterpret_cast<sycl::half*>(weight.data_ptr()),
            acc_ptr,
            cnt_ptr,
            m,
            n,
            k,
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return;
      } else {
        RECORD_XETLA_FUNCTION_IMPL(hgemm_qkv_bias);
        auto cgfs = hgemm_qkv_bias(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr()),
            reinterpret_cast<sycl::half*>(out1.data_ptr()),
            reinterpret_cast<sycl::half*>(out2.data_ptr()),
            reinterpret_cast<sycl::half*>(input.data_ptr()),
            reinterpret_cast<sycl::half*>(weight.data_ptr()),
            reinterpret_cast<sycl::half*>(bias_.value().data_ptr()),
            acc_ptr,
            cnt_ptr,
            m,
            n,
            k,
            arch_tag);
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
    RECORD_FUNC_IMPL(qkv_mm_out3);
    at::AtenIpexTypeXPU::mm_out(input, wq, out0);
    at::AtenIpexTypeXPU::mm_out(input, wk, out1);
    at::AtenIpexTypeXPU::mm_out(input, wv, out2);
  } else {
    RECORD_FUNC_IMPL(qkv_addmm_out3);
    at::AtenIpexTypeXPU::addmm_out(
        bias_.value()[0], input, wq, at::Scalar(1), at::Scalar(1), out0);
    at::AtenIpexTypeXPU::addmm_out(
        bias_.value()[1], input, wk, at::Scalar(1), at::Scalar(1), out1);
    at::AtenIpexTypeXPU::addmm_out(
        bias_.value()[2], input, wv, at::Scalar(1), at::Scalar(1), out2);
  }
}

static void mm_qkv_out_autocast(
    const Tensor& input_,
    const Tensor& weight_,
    const optional<Tensor>& bias_,
    Tensor& out0_,
    Tensor& out1_,
    Tensor& out2_) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  auto to_type = kHalf;
  if (input_.scalar_type() == to_type && weight_.scalar_type() == to_type) {
    mm_qkv_out(input_, weight_, bias_, out0_, out1_, out2_);
  } else {
    auto casted_input = cached_cast(to_type, input_, c10::DeviceType::XPU);
    auto out0 = at::empty_like(out0_, to_type, at::MemoryFormat::Contiguous);
    auto out1 = at::empty_like(out1_, to_type, at::MemoryFormat::Contiguous);
    auto out2 = at::empty_like(out2_, to_type, at::MemoryFormat::Contiguous);
    mm_qkv_out(
        casted_input,
        cached_cast(to_type, weight_, c10::DeviceType::XPU),
        cached_cast(to_type, bias_, c10::DeviceType::XPU),
        out0,
        out1,
        out2);
    out0_.copy_(out0);
    out1_.copy_(out1);
    out2_.copy_(out2);
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  bool is_a_contiguous = input.is_contiguous();
  bool is_b_row_major = weight.is_contiguous();

  TORCH_CHECK(is_a_contiguous && is_b_row_major);
  TORCH_CHECK(input.scalar_type() == kHalf && weight.scalar_type() == kHalf);

  using namespace torch_ipex::xpu::xetla;

  gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
  bool align_valid = check_qkv_align(out0, out1, out2, input, weight);
  bool bias_valid = true;
  if (has_bias) {
    bias_valid = ptr_align64(bias_.value().data_ptr());
  }
  bool shape_valid = ((k & 0x3) == 0) && ((n & 0x3) == 0);
  bool xetla_valid = align_valid && bias_valid && shape_valid;
  if (xetla_valid) {
    int policy_id =
        hgemm_qkv_find_policy_id(m, head_dim, k, is_b_row_major, arch_tag);
    if (policy_id >= 0) {
      auto& queue = dpcppGetCurrentQueue();
      auto input_ptr = c10::MaybeOwned<Tensor>::borrowed(input_);
#if 0
    Tensor acc_tensor_, cnt_tensor_;
    float* acc_ptr = get_acc_and_cnt_tensor(
        input_ptr, m, n, k, policy_id, acc_tensor_, cnt_tensor_);
    uint32_t* cnt_ptr = (acc_ptr == nullptr)
        ? (uint32_t*)nullptr
        : reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr());
#endif
      float* acc_ptr = (float*)nullptr;
      uint32_t* cnt_ptr = (uint32_t*)nullptr;
      if (!has_bias) {
        RECORD_XETLA_FUNCTION_IMPLG(hgemm_qkv_group);
        auto cgfs = hgemm_qkv_group(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr()),
            reinterpret_cast<sycl::half*>(out1.data_ptr()),
            reinterpret_cast<sycl::half*>(out2.data_ptr()),
            reinterpret_cast<sycl::half*>(input.data_ptr()),
            reinterpret_cast<sycl::half*>(weight.data_ptr()),
            acc_ptr,
            cnt_ptr,
            m,
            n,
            k,
            num_kv_head,
            group,
            head_dim,
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
        return;
      } else {
        RECORD_XETLA_FUNCTION_IMPLG(hgemm_qkv_group_bias);
        auto cgfs = hgemm_qkv_group_bias(
            policy_id,
            reinterpret_cast<sycl::half*>(out0.data_ptr()),
            reinterpret_cast<sycl::half*>(out1.data_ptr()),
            reinterpret_cast<sycl::half*>(out2.data_ptr()),
            reinterpret_cast<sycl::half*>(input.data_ptr()),
            reinterpret_cast<sycl::half*>(weight.data_ptr()),
            reinterpret_cast<sycl::half*>(bias_.value().data_ptr()),
            acc_ptr,
            cnt_ptr,
            m,
            n,
            k,
            num_kv_head,
            group,
            head_dim,
            arch_tag);
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
    RECORD_FUNC_IMPLG(qkv_group_mm_common);
    auto out =
        mm_common(input, weight.view({k, num_kv_head * group * head_dim}));
    out = out.view({m, num_kv_head, group, head_dim});
    out0.index_put_(
        {"..."},
        out.index({Slice(), Slice(), Slice(None, group - 2), Slice()}));
    out1.index_put_({"..."}, out.index({Slice(), Slice(), group - 2, Slice()}));
    out2.index_put_({"..."}, out.index({Slice(), Slice(), group - 1, Slice()}));
  } else {
    RECORD_FUNC_IMPLG(qkv_group_mm_bias);
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
  IPEX_OP_REGISTER_DISPATCH(
      "mm_qkv_out.xpu",
      at::AtenIpexTypeXPU::mm_qkv_out_autocast,
      c10::DispatchKey::AutocastXPU);

  IPEX_OP_REGISTER(
      "mm_qkv_group_out.xpu", at::AtenIpexTypeXPU::mm_qkv_group_out);
  IPEX_OP_REGISTER("mm_qkv.xpu", at::AtenIpexTypeXPU::mm_qkv);
}
} // namespace
