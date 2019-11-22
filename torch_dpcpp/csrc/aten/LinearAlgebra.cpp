#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsSYCL.h>

namespace at { namespace native {

Tensor baddbmm_dpcpp(const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::sycl::_th_baddbmm(self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_out_dpcpp(Tensor &result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::sycl::_th_baddbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__dpcpp(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::sycl::_th_baddbmm_out(self, self, batch1, batch2, beta, alpha);
}

Tensor bmm_dpcpp(const Tensor& self, const Tensor& mat2) {
  return legacy::sycl::_th_bmm(self, mat2);
}

Tensor& bmm_out_dpcpp(Tensor &result, const Tensor& batch1, const Tensor& batch2) {
  return legacy::sycl::_th_bmm_out(result, batch1, batch2);
}

} }
