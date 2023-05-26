#include "TPPGEMM.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "tpp/xsmm_functors.h"
namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(fc_in_kernel_stub);
DEFINE_DISPATCH(fc_out_kernel_stub);
DEFINE_DISPATCH(fc_plain_kernel_stub);
DEFINE_DISPATCH(qkv_kernel_stub);


at::Tensor qkv_gemm_forward_cpu(at::Tensor& t_in, at::Tensor& t_wt) {
  return qkv_kernel_stub(kCPU, t_in, t_wt);
}

at::Tensor fc_in_gemm_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return fc_in_kernel_stub(kCPU, t_in, t_wt, t_bias);
}

at::Tensor fc_plain_gemm_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias) {
  return fc_plain_kernel_stub(kCPU, t_in, t_wt, t_bias);
}

at::Tensor fc_out_gemm_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    double scale) {
  return fc_out_kernel_stub(kCPU, t_in, t_in1, t_in2, t_wt, t_bias, scale);
}

} // namespace cpu
} // namespace torch_ipex

namespace {


TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("qkv_gemm(Tensor (a!)t_in, Tensor (a!)t_wt)-> Tensor out");
  m.impl(
      "qkv_gemm", c10::DispatchKey::CPU, torch_ipex::cpu::qkv_gemm_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "fc_in_gemm(Tensor (a!)t_in, Tensor (a!)t_wt, Tensor (a!)t_bias)-> Tensor out");
  m.impl("fc_in_gemm", c10::DispatchKey::CPU, torch_ipex::cpu::fc_in_gemm_forward_cpu);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "fc_plain_gemm(Tensor (a!)t_in, Tensor (a!)t_wt, Tensor (a!)t_bias)-> Tensor out");
  m.impl("fc_plain_gemm", c10::DispatchKey::CPU, torch_ipex::cpu::fc_plain_gemm_forward_cpu);
}


TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "fc_out_gemm(Tensor (a!)t_in, Tensor (a!)t_in1, Tensor (a!)t_in2, Tensor (a!)t_wt, Tensor (a!)t_bias, float scale )-> Tensor out");
  m.impl("fc_out_gemm", c10::DispatchKey::CPU,
  torch_ipex::cpu::fc_out_gemm_forward_cpu);
}

} // namespace