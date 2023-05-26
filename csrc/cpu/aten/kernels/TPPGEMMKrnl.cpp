
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

at::Tensor fc_plain_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);
  // std::cout << "YYY " << t_out.dtype() << "  " << t_in.dtype() << std::endl;
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::fc_plain<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::fc_plain<at::BFloat16>(t_in, t_wt, t_bias, t_out);
  } else {
    AT_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }

  return t_out;
}

at::Tensor fc_out_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    double scale) {
  auto t_out = at::empty_like(t_in1);
  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::fc_out<float>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::fc_out<at::BFloat16>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else {
    AT_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

at::Tensor fc_in_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::fc_in<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::fc_in<at::BFloat16>(t_in, t_wt, t_bias, t_out);
  } else {
    AT_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

at::Tensor qkv_kernel_impl(at::Tensor& t_in, at::Tensor& t_wt) {
  auto sizes = t_in.sizes().vec();
  auto wt_sizes = t_wt.sizes();
  sizes[2] = wt_sizes[0] * wt_sizes[3];

  auto t_out = t_in.new_empty(sizes);

  auto dt = t_wt.dtype();
  if (dt == at::kFloat) {
    torch_ipex::tpp::qkv_gemm<float>(t_in, t_wt, t_out);
  } else if (dt == at::kBFloat16) {
    torch_ipex::tpp::qkv_gemm<at::BFloat16>(t_in, t_wt, t_out);
  } else {
    AT_ASSERT(0, "Should not come here %s:%d\n", __FILE__, __LINE__);
  }
  return t_out;
}

} // namespace

REGISTER_DISPATCH(fc_plain_kernel_stub, &fc_plain_kernel_impl);
REGISTER_DISPATCH(fc_in_kernel_stub, &fc_in_kernel_impl);
REGISTER_DISPATCH(fc_out_kernel_stub, &fc_out_kernel_impl);
REGISTER_DISPATCH(qkv_kernel_stub, &qkv_kernel_impl);
} // namespace cpu
} // namespace torch_ipex
