#include <torch/extension.h>

#include <torch/csrc/autograd/function.h>
#include "LinearMKL.h"
#include "csrc/cpu/ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(mkl_sgemm_packB_stub);
DEFINE_DISPATCH(mkl_sgemm_kernel_stub);
DEFINE_DISPATCH(mkl_prepack_sgemm_kernel_stub);

/**
 * FP32 Linear inplace version with MKL SGEMM kernel.
 *
 *@param self Activatin input for Linear
 *@param mkl_weight MKL prepacked weight for Linear
 *@param bias Bias for Linear
 *@param out_features Size of N-dim for Linear
 *@param output Output tensor provided by user.
 */

at::Tensor mkl_sgemm_pack_weight(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight) {
  return mkl_sgemm_packB_stub(kCPU, M, N, K, ori_weight);
}

void mkl_sgemm_kernel_output(
    const at::Tensor& self,
    const at::Tensor& ori_weight,
    const at::Tensor& bias,
    at::Tensor& output) {
  mkl_sgemm_kernel_stub(kCPU, self, ori_weight, bias, output);
}

at::Tensor mkl_sgemm_kernel(
    const at::Tensor& self,
    const at::Tensor& ori_weight,
    const at::Tensor& bias) {
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(ori_weight.size(0));
  auto output = at::empty(output_size, self.options());
  output.set_requires_grad(self.requires_grad());
  mkl_sgemm_kernel_output(self, ori_weight, bias, output);
  return output;
}

void mkl_prepack_sgemm_kernel_output(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features,
    at::Tensor& output) {
  mkl_prepack_sgemm_kernel_stub(
      kCPU, self, mkl_weight, bias, out_features, output);
}

at::Tensor mkl_prepack_sgemm_kernel(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features) {
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(out_features);
  auto output = at::empty(output_size, self.options());
  output.set_requires_grad(self.requires_grad());
  mkl_prepack_sgemm_kernel_output(self, mkl_weight, bias, out_features, output);
  return output;
}

at::Tensor mkl_sgemm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context) {
  return reinterpret_cast<IpexLinearMKLOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run(input);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "ipex_MKLSGEMM(Tensor input, Tensor weight, Tensor? bias, "
      "Tensor W_prepack) -> Tensor");
  m.impl(
      "ipex_MKLSGEMM",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mkl_sgemm_forward);
}

} // namespace
