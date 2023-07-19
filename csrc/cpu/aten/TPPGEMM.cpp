#include "TPPGEMM.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "tpp/xsmm_functors.h"
namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(tpp_linear_nobias_kernel_stub);
DEFINE_DISPATCH(tpp_linear_bias_kernel_stub);
DEFINE_DISPATCH(tpp_linear_gelu_kernel_stub);
DEFINE_DISPATCH(tpp_linear_silu_kernel_stub);
DEFINE_DISPATCH(tpp_linear_relu_kernel_stub);
DEFINE_DISPATCH(tpp_linear_add_kernel_stub);
DEFINE_DISPATCH(tpp_linear_mul_kernel_stub);
DEFINE_DISPATCH(tpp_linear_add_add_kernel_stub);

at::Tensor tpp_linear_nobias_forward_cpu(at::Tensor& t_in, at::Tensor& t_wt) {
  return tpp_linear_nobias_kernel_stub(kCPU, t_in, t_wt);
}

at::Tensor tpp_linear_bias_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return tpp_linear_bias_kernel_stub(kCPU, t_in, t_wt, t_bias);
}

at::Tensor tpp_linear_gelu_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return tpp_linear_gelu_kernel_stub(kCPU, t_in, t_wt, t_bias);
}

at::Tensor tpp_linear_silu_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return tpp_linear_silu_kernel_stub(kCPU, t_in, t_wt, t_bias);
}

at::Tensor tpp_linear_relu_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return tpp_linear_relu_kernel_stub(kCPU, t_in, t_wt, t_bias);
}

at::Tensor tpp_linear_add_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    double scale) {
  return tpp_linear_add_kernel_stub(kCPU, t_in, t_in1, t_wt, t_bias, scale);
}

at::Tensor tpp_linear_mul_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return tpp_linear_mul_kernel_stub(kCPU, t_in, t_in1, t_wt, t_bias);
}

at::Tensor tpp_linear_add_add_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    double scale) {
  return tpp_linear_add_add_kernel_stub(
      kCPU, t_in, t_in1, t_in2, t_wt, t_bias, scale);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("tpp_linear(Tensor (a!)t_in, Tensor (a!)t_wt)-> Tensor out");
  m.impl(
      "tpp_linear",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_nobias_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_bias(Tensor (a!)t_in, Tensor (a!)t_wt, Tensor (a!)t_bias)-> Tensor out");
  m.impl(
      "tpp_linear_bias",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_bias_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_gelu(Tensor (a!)t_in, Tensor (a!)t_wt, Tensor (a!)t_bias)-> Tensor out");
  m.impl(
      "tpp_linear_gelu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_gelu_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_add_add(Tensor (a!)t_in, Tensor (a!)t_in1, Tensor (a!)t_in2, Tensor (a!)t_wt, Tensor (a!)t_bias, float scale )-> Tensor out");
  m.impl(
      "tpp_linear_add_add",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_add_add_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_relu(Tensor (a!)t_in, Tensor (a!)t_wt, Tensor (a!)t_bias)-> Tensor out");
  m.impl(
      "tpp_linear_relu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_relu_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_silu(Tensor (a!)t_in, Tensor (a!)t_wt, Tensor (a!)t_bias)-> Tensor out");
  m.impl(
      "tpp_linear_silu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_silu_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_add(Tensor (a!)t_in, Tensor (a!)t_in1, Tensor (a!)t_wt, Tensor (a!)t_bias, float scale )-> Tensor out");
  m.impl(
      "tpp_linear_add",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_add_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "tpp_linear_mul(Tensor (a!)t_in, Tensor (a!)t_in1, Tensor (a!)t_wt, Tensor (a!)t_bias )-> Tensor out");
  m.impl(
      "tpp_linear_mul",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::tpp_linear_mul_forward_cpu);
}

} // namespace