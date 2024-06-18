
#ifdef USE_LIBXSMM
#include "tpp/kernels/TPPGEMMKrnl.h"
#include <ATen/record_function.h>
#include <aten/TPPGEMM.h>
#include <torch/all.h>
#include <cstdint>
#include <iostream>
#include <vector>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor tpp_linear_bias_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_bias<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_bias<at::BFloat16>(t_in, t_wt, t_bias, t_out);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }

  return t_out;
}

at::Tensor tpp_linear_nobias_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_wt) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_no_bias<float>(t_in, t_wt, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_no_bias<at::BFloat16>(t_in, t_wt, t_out);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_linear_gelu_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    const c10::string_view& algorithm) {
  AT_ASSERT(
      algorithm == "none" || algorithm == "tanh",
      "tpp_linear_gelu: Invalid gelu algorithm %s\n",
      algorithm);

  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    if (algorithm == "none") {
      torch_ipex::tpp::tpp_linear_gelu<float>(t_in, t_wt, t_bias, t_out);
    } else { // tanh
      torch_ipex::tpp::tpp_linear_gelu_tanh<float>(t_in, t_wt, t_bias, t_out);
    }
  } else if (dt == at::kBFloat16) {
    if (algorithm == "none") {
      torch_ipex::tpp::tpp_linear_gelu<at::BFloat16>(t_in, t_wt, t_bias, t_out);
    } else { // tanh
      torch_ipex::tpp::tpp_linear_gelu_tanh<at::BFloat16>(
          t_in, t_wt, t_bias, t_out);
    }
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_fused_gate_up_proj_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_wt_gate,
    const at::Tensor& t_bias_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_bias_up) {
  auto sizes = t_in.sizes().vec();
  AT_ASSERT(
      t_wt_gate.sizes() == t_wt_up.sizes(),
      "Expect t_wt_gate.sizes() == t_wt_up.sizes()");
  auto wt_sizes = t_wt_gate.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt_gate.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_fused_gate_up_proj<float>(
        t_in, t_wt_gate, t_bias_gate, t_wt_up, t_bias_up, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_fused_gate_up_proj<at::BFloat16>(
        t_in, t_wt_gate, t_bias_gate, t_wt_up, t_bias_up, t_out);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_linear_silu_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_silu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_silu<at::BFloat16>(t_in, t_wt, t_bias, t_out);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_linear_relu_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_relu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_relu<at::BFloat16>(t_in, t_wt, t_bias, t_out);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_linear_add_add_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_in2,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    double scale) {
  auto t_out = at::empty_like(t_in1);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_add_add<float>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_add_add<at::BFloat16>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_linear_add_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    double scale) {
  auto t_out = at::empty_like(t_in1);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_add<float>(
        t_in, t_in1, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_add<at::BFloat16>(
        t_in, t_in1, t_wt, t_bias, t_out, scale);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

at::Tensor tpp_linear_mul_kernel_impl(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias) {
  auto t_out = at::empty_like(t_in1);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_mul<float>(t_in, t_in1, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_mul<at::BFloat16>(
        t_in, t_in1, t_wt, t_bias, t_out);
  } else {
    AT_ASSERT(
        0,
        "TPP does not support current weight dtype %s:%d\n",
        __FILE__,
        __LINE__);
  }
  return t_out;
}

void tpp_gelu_tanh_bf16_kernel_impl(
    at::BFloat16* in,
    at::BFloat16* out,
    int M,
    int N,
    int ldi,
    int ldo) {
#ifdef CPU_CAPABILITY_AVX512
  const __m512 c1 = _mm512_set1_ps((float)0.7978846);
  const __m512 c2 = _mm512_set1_ps((float)0.0356814);
  const __m512 c_half = _mm512_set1_ps((float)0.5);
  for (int j = 0; j < M; j++) {
    int i;
    for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
      auto vin = torch_ipex::tpp::_mm512_loadu_ps_auto(&in[j * ldi + i]);
      __m512 x_half = _mm512_mul_ps(vin, c_half);
      __m512 x_sq = _mm512_mul_ps(vin, vin);
      __m512 poly_x1 = _mm512_mul_ps(vin, _mm512_fmadd_ps(x_sq, c2, c1));
      __m512 tanh_poly_x = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX3(poly_x1);
      __m512 vout = _mm512_fmadd_ps(tanh_poly_x, x_half, x_half);
      torch_ipex::tpp::_mm512_storeu_ps_auto(&out[j * ldo + i], vout);
    }
    if (i < N) {
      int rem = N - i;
      __mmask16 mask = (1 << rem) - 1;
      auto vin =
          torch_ipex::tpp::_mm512_maskz_loadu_ps_auto(mask, &in[j * ldi + i]);
      __m512 x_half = _mm512_mul_ps(vin, c_half);
      __m512 x_sq = _mm512_mul_ps(vin, vin);
      __m512 poly_x1 = _mm512_mul_ps(vin, _mm512_fmadd_ps(x_sq, c2, c1));
      __m512 tanh_poly_x = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX3(poly_x1);
      __m512 vout = _mm512_fmadd_ps(tanh_poly_x, x_half, x_half);
      torch_ipex::tpp::_mm512_mask_storeu_ps_auto(
          &out[j * ldo + i], mask, vout);
    }
  }
#else
  for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      float x = in[j * ldi + i];
      out[j * ldo + i] =
          ((tanh(sqrt(2 / M_PI) * (x + 0.044715 * std::pow(x, 3)))) + 1) * x *
          0.5;
    }
  }
#endif
}

} // namespace

IPEX_REGISTER_DISPATCH(
    tpp_linear_nobias_kernel_stub,
    &tpp_linear_nobias_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_bias_kernel_stub,
    &tpp_linear_bias_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_gelu_kernel_stub,
    &tpp_linear_gelu_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_fused_gate_up_proj_kernel_stub,
    &tpp_fused_gate_up_proj_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_relu_kernel_stub,
    &tpp_linear_relu_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_silu_kernel_stub,
    &tpp_linear_silu_kernel_impl);
IPEX_REGISTER_DISPATCH(tpp_linear_mul_kernel_stub, &tpp_linear_mul_kernel_impl);
IPEX_REGISTER_DISPATCH(tpp_linear_add_kernel_stub, &tpp_linear_add_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_add_add_kernel_stub,
    &tpp_linear_add_add_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_gelu_tanh_bf16_kernel_stub,
    &tpp_gelu_tanh_bf16_kernel_impl);
} // namespace cpu
} // namespace torch_ipex
#endif