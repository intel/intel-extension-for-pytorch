
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
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
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

at::Tensor tpp_linear_nobias_kernel_impl(at::Tensor& t_in, at::Tensor& t_wt) {
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
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::tpp_linear_gelu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::tpp_linear_gelu<at::BFloat16>(t_in, t_wt, t_bias, t_out);
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
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
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
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
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
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
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
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
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
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
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

} // namespace

REGISTER_DISPATCH(
    tpp_linear_nobias_kernel_stub,
    &tpp_linear_nobias_kernel_impl);
REGISTER_DISPATCH(tpp_linear_bias_kernel_stub, &tpp_linear_bias_kernel_impl);
REGISTER_DISPATCH(tpp_linear_gelu_kernel_stub, &tpp_linear_gelu_kernel_impl);
REGISTER_DISPATCH(tpp_linear_relu_kernel_stub, &tpp_linear_relu_kernel_impl);
REGISTER_DISPATCH(tpp_linear_silu_kernel_stub, &tpp_linear_silu_kernel_impl);
REGISTER_DISPATCH(tpp_linear_mul_kernel_stub, &tpp_linear_mul_kernel_impl);
REGISTER_DISPATCH(tpp_linear_add_kernel_stub, &tpp_linear_add_kernel_impl);
REGISTER_DISPATCH(
    tpp_linear_add_add_kernel_stub,
    &tpp_linear_add_add_kernel_impl);
} // namespace cpu
} // namespace torch_ipex
#endif