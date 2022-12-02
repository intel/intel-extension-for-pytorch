#include <torch/csrc/autograd/function.h>
#include "aten/LinearMKL.h"
#include "aten/utils/utils.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

void _mkl_sgemm_packB_impl(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight,
    at::Tensor& mkl_weight) {
  cblas_sgemm_pack(
      CblasRowMajor,
      CblasBMatrix,
      CblasTrans,
      M,
      N,
      K,
      1.0f,
      ori_weight.data_ptr<float>(),
      K,
      mkl_weight.data_ptr<float>());
}

at::Tensor mkl_sgemm_packB_impl(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight) {
  int64_t pack_size =
      (int64_t)(cblas_sgemm_pack_get_size(CblasBMatrix, M, N, K) / sizeof(float) + 1);
  at::Tensor mkl_weight = at::empty(pack_size, at::kFloat);
  _mkl_sgemm_packB_impl(M, N, K, ori_weight, mkl_weight);
  return mkl_weight;
}

void mkl_sgemm_base_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int64_t N,
    at::Tensor& output,
    bool pack) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  auto M = self_reshaped.size(0);
  auto K = self_reshaped.size(1);

  auto in_ptr = self_.data_ptr<float>();
  auto weight_ptr = weight.data_ptr<float>();
  auto out_ptr = output.data_ptr<float>();

  if (bias.defined()) {
    auto bias_ = self.is_contiguous() ? bias : bias.contiguous();
    auto bias_ptr = bias_.data_ptr<float>();
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
    for (int64_t i = 0; i < M; ++i)
      memcpy(out_ptr + i * N, bias_ptr, sizeof(float) * N);
  }
  if (pack) {
    cblas_sgemm_compute(
        CblasRowMajor,
        CblasNoTrans,
        CblasPacked,
        M,
        N,
        K,
        in_ptr,
        K,
        weight_ptr,
        K,
        bias.defined() ? 1.f : 0.f,
        out_ptr,
        N);
  } else {
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M,
        N,
        K,
        1.0f,
        in_ptr,
        K,
        weight_ptr,
        K,
        bias.defined() ? 1.f : 0.f,
        out_ptr,
        N);
  }
}

void mkl_sgemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& ori_weight,
    const at::Tensor& bias,
    at::Tensor& output) {
  mkl_sgemm_base_kernel_impl(
      self, ori_weight, bias, ori_weight.size(0), output, false);
}

void mkl_prepack_sgemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features,
    at::Tensor& output) {
  mkl_sgemm_base_kernel_impl(
      self, mkl_weight, bias, out_features, output, true);
}

} // anonymous namespace

REGISTER_DISPATCH(mkl_sgemm_packB_stub, &mkl_sgemm_packB_impl);
REGISTER_DISPATCH(mkl_sgemm_kernel_stub, &mkl_sgemm_kernel_impl);
REGISTER_DISPATCH(
    mkl_prepack_sgemm_kernel_stub,
    &mkl_prepack_sgemm_kernel_impl);

} // namespace cpu
} // namespace torch_ipex