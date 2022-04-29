#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>
#include <core/TensorImplUtils.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include <vector>

#include <quantized/QUtil.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include <c10/util/typeid.h>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace impl {

bool check_broadcast(const Tensor& src, const IntArrayRef& shape) {
  auto src_dim = src.dim();
  auto tgt_dim = shape.size();
  if (src_dim == 0 && src_dim < tgt_dim)
    return true;
  if (src_dim > tgt_dim)
    return false;
  do {
    src_dim--;
    tgt_dim--;
    auto size = src.size(src_dim);
    if (size != 1 && size != shape[tgt_dim])
      return false;
  } while (src_dim);
  return true;
}

#ifdef USE_ONEMKL
template <typename scalar_t>
void gemm_batch(
    DPCPP::queue& queue,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int64_t m,
    int64_t n,
    int64_t k,
    scalar_t alpha,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    scalar_t* b,
    int64_t ldb,
    int64_t stride_b,
    scalar_t beta,
    scalar_t* c,
    int64_t ldc,
    int64_t stride_c,
    int64_t batch_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::column_major::gemm_batch,
      queue,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a,
      lda,
      stride_a,
      b,
      ldb,
      stride_b,
      beta,
      c,
      ldc,
      stride_c,
      batch_size);
}

template <>
void gemm_batch<c10::complex<double>>(
    DPCPP::queue& queue,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int64_t m,
    int64_t n,
    int64_t k,
    c10::complex<double> alpha,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    c10::complex<double>* b,
    int64_t ldb,
    int64_t stride_b,
    c10::complex<double> beta,
    c10::complex<double>* c,
    int64_t ldc,
    int64_t stride_c,
    int64_t batch_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::column_major::gemm_batch,
      queue,
      transa,
      transb,
      m,
      n,
      k,
      *reinterpret_cast<std::complex<double>*>(&alpha),
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      reinterpret_cast<std::complex<double>*>(b),
      ldb,
      stride_b,
      *reinterpret_cast<std::complex<double>*>(&beta),
      reinterpret_cast<std::complex<double>*>(c),
      ldc,
      stride_c,
      batch_size);
}

template <>
void gemm_batch<c10::complex<float>>(
    DPCPP::queue& queue,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int64_t m,
    int64_t n,
    int64_t k,
    c10::complex<float> alpha,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    c10::complex<float>* b,
    int64_t ldb,
    int64_t stride_b,
    c10::complex<float> beta,
    c10::complex<float>* c,
    int64_t ldc,
    int64_t stride_c,
    int64_t batch_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::column_major::gemm_batch,
      queue,
      transa,
      transb,
      m,
      n,
      k,
      *reinterpret_cast<std::complex<float>*>(&alpha),
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      reinterpret_cast<std::complex<float>*>(b),
      ldb,
      stride_b,
      *reinterpret_cast<std::complex<float>*>(&beta),
      reinterpret_cast<std::complex<float>*>(c),
      ldc,
      stride_c,
      batch_size);
}
#endif

void mkl_baddbmm(
    Tensor& result,
    const Tensor& self,
    Tensor batch1,
    Tensor batch2,
    const Scalar& beta,
    const Scalar& alpha) {
#ifdef USE_ONEMKL
  // colum major
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  auto batch1_sizes = batch1.sizes();
  auto batch2_sizes = batch2.sizes();
  auto batch1_strides = batch1.strides();
  auto batch2_strides = batch2.strides();
  auto self_sizes = self.sizes();

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    auto b_self = expand_size(
        self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "mkl_matmul");
    result.resize_as_(*b_self).copy_(*b_self);
  } else {
    result.resize_({batch1.size(0), batch1.size(1), batch2.size(2)});
  }

  TORCH_CHECK(
      self_sizes[0] == batch1_sizes[0], "self dim 0 must match batch1 dim 0");
  TORCH_CHECK(
      self_sizes[0] == batch2_sizes[0], "self dim 0 must match batch2 dim 0");
  TORCH_CHECK(
      self_sizes[1] == batch1_sizes[1], "self dim 1 must match batch1 dim 1");
  TORCH_CHECK(
      self_sizes[2] == batch2_sizes[2], "self dim 2 must match batch2 dim 2");
  TORCH_CHECK(
      batch1_sizes[2] == batch2_sizes[1],
      "batch1 dim 2 must match batch2 dim 1");

  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      result.zero_();
    }
  }

  bool transpose_c = false;
  Tensor c;

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) ||
       (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    // colum major
    transpose_c = false;
    c = result.resolve_conj();
  } else if (
      (result_strides[2] == 1) &&
      (result_sizes[1] == 1 ||
       (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    // row major
    std::swap(batch1, batch2);
    std::swap(batch1_sizes, batch2_sizes);
    std::swap(batch1_strides, batch2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    c = result.resolve_conj().transpose(1, 2).contiguous().transpose_(1, 2);
  }

  const int64_t m = result_sizes[transpose_c ? 2 : 1];
  const int64_t n = result_sizes[transpose_c ? 1 : 2];
  const int64_t k = batch1_sizes[transpose_c ? 1 : 2];

  // Cast batch1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (batch1_strides[transpose_c ? 2 : 1] == 1 &&
      batch1_strides[transpose_c ? 1 : 2] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = batch1.resolve_conj();
  } else if (
      batch1_strides[transpose_c ? 1 : 2] == 1 &&
      batch1_strides[transpose_c ? 2 : 1] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = batch1;
  } else {
    transpose_a = !transpose_c;
    a = batch1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast batch2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (batch2_strides[transpose_c ? 2 : 1] == 1 &&
      batch2_strides[transpose_c ? 1 : 2] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = batch2.resolve_conj();
  } else if (
      batch2_strides[transpose_c ? 1 : 2] == 1 &&
      batch2_strides[transpose_c ? 2 : 1] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = batch2;
  } else {
    transpose_b = !transpose_c;
    b = batch2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 2 : 1];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 2 : 1];
  const int64_t ldc = c.strides()[transpose_c ? 1 : 2];

  const int64_t stridea = a.strides()[0];
  const int64_t strideb = b.strides()[0];
  const int64_t stridec = c.strides()[0];
  int64_t num_batch = c.sizes()[0];

  // Always ensure the conjugation for c is resolved since there's no way to
  // specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "mkl_baddbmm", [&] {
        gemm_batch<scalar_t>(
            dpcpp_queue,
            transpose_a ? a.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N, // nontrans = 0, trans = 1,
                                                     // conjtrans = 3,
            transpose_b ? b.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N,
            m,
            n,
            k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(),
            lda,
            stridea,
            b.data_ptr<scalar_t>(),
            ldb,
            strideb,
            beta.to<scalar_t>(),
            c.data_ptr<scalar_t>(),
            ldc,
            stridec,
            num_batch);
      });

  if (!result.is_same(c)) {
    result.copy_(c);
  }
#endif
}

void mkl_matmul(
    Tensor& result,
    const Tensor& self,
    Tensor m1,
    Tensor m2,
    Scalar beta,
    Scalar alpha) {
#ifdef USE_ONEMKL
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    auto b_self = expand_size(self, {m1_sizes[0], m2_sizes[1]}, "mkl_matmul");
    result.resize_as_(*b_self).copy_(*b_self);
  } else {
    result.resize_({m1_sizes[0], m2_sizes[1]});
  }

  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  }

  bool transpose_c = false;
  Tensor c;

  // Cast result as matrix a
  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 ||
       result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (
      result_strides[1] == 1 &&
      (result_sizes[0] == 1 ||
       result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    // make c FORTRAN contiguous
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  // Cast m1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (
      m1_strides[transpose_c ? 0 : 1] == 1 &&
      m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast m2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (
      m2_strides[transpose_c ? 0 : 1] == 1 &&
      m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

  // Always ensure the conjugation for c is resolved since there's no way to
  // specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  // use colum major
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "mkl_matmul", [&] {
        gemm_batch<scalar_t>(
            dpcpp_queue,
            transpose_a ? a.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N, // nontrans = 0, trans = 1,
                                                     // conjtrans = 3,
            transpose_b ? b.is_conj() ? oneapi::mkl::transpose::C
                                      : oneapi::mkl::transpose::T
                        : oneapi::mkl::transpose::N,
            m,
            n,
            k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(),
            lda,
            a.numel(),
            b.data_ptr<scalar_t>(),
            ldb,
            b.numel(),
            beta.to<scalar_t>(),
            c.data_ptr<scalar_t>(),
            ldc,
            c.numel(),
            1);
      });

  if (!c.is_same(result)) {
    result.copy_(c);
  }
#endif
}

} // namespace impl

namespace AtenIpexTypeXPU {

using namespace impl;

// ((m1 x m2 + b) * alpha + beta * c) - relu
void matmul(
    Tensor& result,
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& b, // tensor for bias
    const Tensor& po, // tensor for post_sum
    float beta, // beta only for post sum
    float alpha, // oscale
    bool m2_trans,
    int fusion) {
  if (m1.is_quantized()) {
    if (m2.sizes()[1] == m1.sizes()[1])
      m2.transpose_(0, 1);
  }

  MatmulAttr attr(alpha, beta, fusion, m2_trans);

  std::vector<int64_t> result_shape;
  auto dim = m1.dim();
  if (dim == 2) {
    result_shape = attr.m2_trans_
        ? std::vector<int64_t>{m1.size(0), m2.size(1)}
        : std::vector<int64_t>{m1.size(0), m2.size(0)};
  } else {
    result_shape = attr.m2_trans_
        ? std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(2)}
        : std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(1)};
  }
  if (po.defined() && beta != 0) {
    TORCH_CHECK(
        check_broadcast(po, result_shape),
        "tensor for accumulate ",
        po.sizes(),
        " cannot broadcast to ",
        result_shape);
    c10::MaybeOwned<Tensor> bc_po =
        expand_size(po, result_shape, "gemm_broadcast");
    if (!result.is_same(*bc_po))
      result.resize_(result_shape).copy_(*bc_po);
  } else {
    result.resize_(result_shape);
  }

  xpu::oneDNN::matmul(result, m1, m2, b, attr);
}

Tensor& addmm_out(
    const Tensor& self,
    const Tensor& m1,
    const Tensor& m2,
    const Scalar& beta,
    const Scalar& alpha,
    at::Tensor& result) {
  checkBackend("addmm_out", {result, self, m1, m2}, Backend::XPU);
  TORCH_CHECK(m1.dim() == 2 && m2.dim() == 2, "tensors must be 2-D");

  if (m1.is_complex() || m1.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, self, m1, m2, beta, alpha);
    return result;
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

  if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f ||
      self.is_same(result)) {
    // post sum
    matmul(
        result,
        m1,
        m2.scalar_type() == m1.scalar_type()
            ? m2
            : (m1.scalar_type() == ScalarType::BFloat16 ||
               m2.scalar_type() == ScalarType::BFloat16)
                ? m2
                : m2.to(m1.scalar_type()),
        at::Tensor(),
        self,
        beta.to<float>(),
        alpha.to<float>(),
        true,
        MatmulAttr::kind_with_sum);
  } else {
    // bias
    matmul(
        result,
        m1,
        m2.scalar_type() == m1.scalar_type()
            ? m2
            : (m1.scalar_type() == ScalarType::BFloat16 ||
               m2.scalar_type() == ScalarType::BFloat16)
                ? m2
                : m2.to(m1.scalar_type()),
        self,
        at::Tensor(),
        beta.to<float>(),
        alpha.to<float>(),
        true,
        0);
  }
  return result;
}

Tensor& addmm_(
    Tensor& self,
    const Tensor& m1,
    const Tensor& m2,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor bias = at::empty_like(self).copy_(self);
  // oneDNN cannot support result/bias (write/read) use the same memory.
  // we will remove copy to keep performance once matmul refactor done.
  at::AtenIpexTypeXPU::addmm_out(bias, m1, m2, beta, alpha, self);
  return self;
}

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16) {
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  at::AtenIpexTypeXPU::addmm_out(input, m1, m2, beta, alpha, result);
  return result;
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(mat2.dim() == 2, "expected 2D tensor");

  if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, result, self, mat2, Scalar(0), Scalar(1));
    return result;
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

  auto self_dt = self.scalar_type();
  auto result_dt = result.scalar_type();
  auto mat2_dt = mat2.scalar_type();

  matmul(
      result,
      (self_dt == result_dt ||
       ((self_dt == ScalarType::BFloat16 &&
         result_dt != ScalarType::BFloat16) ||
        (result_dt == ScalarType::BFloat16 && self_dt != ScalarType::BFloat16)))
          ? self
          : self.to(result_dt),
      (mat2_dt == result_dt ||
       ((mat2_dt == ScalarType::BFloat16 &&
         result_dt != ScalarType::BFloat16) ||
        (result_dt == ScalarType::BFloat16 && mat2_dt != ScalarType::BFloat16)))
          ? mat2
          : mat2.to(result_dt),
      at::Tensor(),
      at::Tensor(),
      0.f,
      1.f,
      true,
      0);
  return result;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::mm_out(result, self, mat2);
  return result;
}

Tensor mv(const Tensor& self, const Tensor& vec) {
  Tensor result = at::empty({self.size(0)}, self.options());
  return at::addmv_(result, self, vec, 0, 1);
}

Tensor& baddbmm_out(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  if (batch1.is_complex() || batch2.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_baddbmm(result, input, batch1, batch2, beta, alpha);
    return result;
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

  if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f ||
      input.is_same(result)) {
    matmul(
        result,
        batch1,
        batch2,
        at::Tensor(),
        input,
        beta.to<float>(),
        alpha.to<float>(),
        true,
        MatmulAttr::kind_with_sum);
  } else {
    matmul(
        result,
        batch1,
        batch2,
        input,
        at::Tensor(),
        beta.to<float>(),
        alpha.to<float>(),
        true,
        0);
  }
  return result;
}

Tensor& baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  return at::AtenIpexTypeXPU::baddbmm_out(
      self, batch1, batch2, beta, alpha, self);
}

Tensor baddbmm(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor r = at::empty({0}, input.options());
  at::AtenIpexTypeXPU::baddbmm_out(input, batch1, batch2, beta, alpha, r);
  return r;
}

Tensor& addbmm_out(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  checkBackend("addbmm_out", {out, self, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(
      batch1.dim() == 3 && batch2.dim() == 3,
      "Batch tensors should be 3D, got dimensions ",
      batch1.dim(),
      " and ",
      batch2.dim());

  Tensor b1;
  if (batch1.size(0) > 1) {
    b1 = batch1.transpose(0, 1).contiguous().view({batch1.size(1), -1});
  } else {
    b1 = batch1.view({batch1.size(1), -1});
  }
  auto b2 = batch2.view({-1, batch2.size(2)});
  at::AtenIpexTypeXPU::addmm_out(self, b1, b2, beta, alpha, out);

  return out;
}

Tensor& addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  at::AtenIpexTypeXPU::addbmm_out(self, batch1, batch2, beta, alpha, self);
  return self;
}

Tensor addbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor out = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::addbmm_out(self, batch1, batch2, beta, alpha, out);
  return out;
}

Tensor& bmm_out(Tensor& result, const Tensor& self, const Tensor& batch2) {
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    return at::AtenIpexTypeXPU::baddbmm_out(
        result, self, batch2, Scalar(0), Scalar(1), result);
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }
  matmul(result, self, batch2, at::Tensor(), at::Tensor(), 0.f, 1.f, true, 0);
  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::bmm_out(result, self, batch2);
  return result;
}

// FIXME: should not be here
Tensor trans_addmm_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar beta,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_relu", std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_relu", {input, weight, bias}, Backend::XPU);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());

    if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f) {
      matmul(
          result,
          input,
          weight,
          at::Tensor(),
          bias,
          beta.to<float>(),
          alpha.to<float>(),
          false,
          (MatmulAttr::kind_with_sum | MatmulAttr::kind_with_relu));
    } else {
      matmul(
          result,
          input,
          weight,
          bias,
          at::Tensor(),
          beta.to<float>(),
          alpha.to<float>(),
          false,
          MatmulAttr::kind_with_relu);
    }
    return result;
  }

  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::relu(output);
}

Tensor trans_addmm_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar beta,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_sigmoid", std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_sigmoid", {input, weight, bias}, Backend::XPU);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());
    if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f) {
      matmul(
          result,
          input,
          weight,
          at::Tensor(),
          bias,
          beta.to<float>(),
          alpha.to<float>(),
          false,
          (MatmulAttr::kind_with_sum | MatmulAttr::kind_with_sigmoid));
    } else {
      matmul(
          result,
          input,
          weight,
          bias,
          at::Tensor(),
          beta.to<float>(),
          alpha.to<float>(),
          false,
          MatmulAttr::kind_with_sigmoid);
    }
    return result;
  }
  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::sigmoid(output);
}

Tensor trans_addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("addmm", {input, m1, m2}, Backend::XPU);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16) {
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f) {
    matmul(
        result,
        m1,
        m2.scalar_type() == m1.scalar_type()
            ? m2
            : (m1.scalar_type() == ScalarType::BFloat16 ||
               m2.scalar_type() == ScalarType::BFloat16)
                ? m2
                : m2.to(m1.scalar_type()),
        at::Tensor(),
        input,
        beta.to<float>(),
        alpha.to<float>(),
        false,
        MatmulAttr::kind_with_sum);
  } else {
    matmul(
        result,
        m1,
        m2.scalar_type() == m1.scalar_type()
            ? m2
            : (m1.scalar_type() == ScalarType::BFloat16 ||
               m2.scalar_type() == ScalarType::BFloat16)
                ? m2
                : m2.to(m1.scalar_type()),
        input,
        at::Tensor(),
        beta.to<float>(),
        alpha.to<float>(),
        false,
        0);
  }
  return result;
}

Tensor& addmv_out(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  Tensor self_v;
  TORCH_CHECK(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());
  if (self.dim() == 1 && self.size(0) != 1) {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
        "size mismatch, get ",
        self.size(0),
        ", ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self.view({self.size(0), 1});
  } else {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0)),
        "size mismatch, get ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self;
  }

  Tensor vec_v = vec.view({vec.size(0), 1});
  at::AtenIpexTypeXPU::addmm_out(self_v, mat, vec_v, beta, alpha, out);
  out.resize_({mat.size(0)});
  return out;
}

Tensor& addmv_(
    Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    at::Scalar beta,
    at::Scalar alpha) {
  Tensor self_v;
  TORCH_CHECK(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());
  if (self.dim() == 1 && self.size(0) != 1) {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
        "size mismatch, get ",
        self.size(0),
        ", ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self.view({self.size(0), 1});
  } else {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0)),
        "size mismatch, get ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self;
  }

  Tensor vec_v = vec.view({vec.size(0), 1});
  at::AtenIpexTypeXPU::addmm_(self_v, mat, vec_v, beta, alpha);
  return self;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    const Scalar& beta,
    const Scalar& alpha) {
  checkBackend("addmm", m1, Backend::QuantizedXPU);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");

  Tensor result;
  if (input.is_quantized()) {
    result = _empty_affine_quantized(
        {0},
        device(kXPU).dtype(input.scalar_type()),
        1.f,
        static_cast<int>(0),
        MemoryFormat::Contiguous);
  } else {
    result = at::empty({0}, input.options());
  }

  at::AtenIpexTypeXPU::matmul(
      result,
      m1,
      m2,
      Tensor(),
      input,
      beta.to<float>(),
      alpha.to<float>(),
      true,
      MatmulAttr::kind_with_sum);

  return result;
}

Tensor trans_addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16) {
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  at::AtenIpexTypeXPU::matmul(
      result,
      m1,
      m2,
      Tensor(),
      input,
      beta.to<float>(),
      alpha.to<float>(),
      false,
      MatmulAttr::kind_with_sum);

  return result;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
