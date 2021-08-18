#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>
#include <core/TensorImplUtils.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <vector>

#include <quantized/QUtil.h>
#include "comm/ATDispatch.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <utils/oneMKLUtils.h>
#endif

#include <c10/util/typeid.h>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace impl {

bool check_broadcast(const Tensor& src, const IntArrayRef& shape) {
  auto src_dim = src.dim();
  auto tgt_dim = shape.size();
  if (src_dim == 0 || src_dim > tgt_dim)
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

void mkl_matmul(
    Tensor& result,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
#ifdef USE_ONEMKL
  Tensor _result;
  if (result.is_contiguous() && result.scalar_type() == ScalarType::Double) {
    _result = result;
  } else {
    _result =
        at::empty_like(result, result.options().dtype(ScalarType::Double));
  }

  size_t dims = _result.dim();

  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneMKL matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == m1.dim() && dims == m2.dim(),
      "oneMKL input matrixes must have the same ranks");

  if (dims == 3) {
    Tensor _m1 = m1.contiguous().to(ScalarType::Double);
    Tensor _m2 = m2.contiguous().to(ScalarType::Double);

    int64_t m = _result.size(-2);
    int64_t k = _m1.size(-1);
    int64_t n = _result.size(-1);
    int64_t stridea = m * k;
    int64_t strideb = k * n;
    int64_t stridec = m * n;
    int64_t mb = 1;

    mb = _result.size(0);
    TORCH_CHECK(
        mb == _m1.size(0) && mb == _m2.size(0),
        "batch size mismatch, result mb: ",
        mb,
        "m1 mb",
        _m1.size(0),
        " m2 mb: ",
        _m2.size(0));

    auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::row_major::gemm_batch,
        dpcpp_queue,
        oneapi::mkl::transpose::N,
        oneapi::mkl::transpose::N,
        m,
        n,
        k,
        alpha.to<float>(),
        (double*)_m1.data_ptr(),
        m,
        stridea,
        (double*)_m2.data_ptr(),
        k,
        strideb,
        beta.to<float>(),
        (double*)_result.data_ptr(),
        m,
        stridec,
        mb);
  } else {
    Tensor _m1 = m1.to(ScalarType::Double);
    Tensor _m2 = m2.to(ScalarType::Double);

    auto m1_strides = _m1.strides();
    auto m1_sizes = _m1.sizes();
    auto m2_strides = _m2.strides();
    auto m2_sizes = _m2.sizes();

    const auto result_strides = _result.strides();
    const auto result_sizes = _result.sizes();

    bool transpose_c = false;
    Tensor c;

    // Cast result as matrix a
    if (result_strides[0] == 1 &&
        (result_sizes[1] == 1 ||
         result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
      transpose_c = false;
      c = _result;
    } else if (
        result_strides[1] == 1 &&
        (result_sizes[0] == 1 ||
         result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
      std::swap(_m1, _m2);
      std::swap(m1_sizes, m2_sizes);
      std::swap(m1_strides, m2_strides);
      transpose_c = true;
      c = _result;
    } else {
      transpose_c = false;
      // make c FORTRAN contiguous
      c = _result.transpose(0, 1).contiguous().transpose_(0, 1);
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
      a = _m1;
    } else if (
        m1_strides[transpose_c ? 0 : 1] == 1 &&
        m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
      transpose_a = true;
      a = _m1;
    } else {
      transpose_a = !transpose_c;
      a = _m1.clone(at::MemoryFormat::Contiguous);
    }

    // Cast m2 as matrix b
    bool transpose_b = false;
    Tensor b;
    /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
    if (m2_strides[transpose_c ? 1 : 0] == 1 &&
        m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
      transpose_b = false;
      b = _m2;
    } else if (
        m2_strides[transpose_c ? 0 : 1] == 1 &&
        m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
      transpose_b = true;
      b = _m2;
    } else {
      transpose_b = !transpose_c;
      b = _m2.clone(at::MemoryFormat::Contiguous);
    }

    const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
    const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
    const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

    int64_t stridea = m * k;
    int64_t strideb = k * n;
    int64_t stridec = m * n;
    int64_t mb = 1;

    auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::gemm_batch,
        dpcpp_queue,
        transpose_a ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N,
        transpose_b ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N,
        m,
        n,
        k,
        alpha.to<float>(),
        (double*)_m1.data_ptr(),
        m,
        stridea,
        (double*)_m2.data_ptr(),
        k,
        strideb,
        beta.to<float>(),
        (double*)_result.data_ptr(),
        m,
        stridec,
        mb);
  }
  if (!_result.is_same(result)) {
    result.copy_(_result);
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
  Tensor bc_po = po;
  if (po.defined() && beta != 0) {
    TORCH_CHECK(
        check_broadcast(po, result_shape),
        "tensor for accumulate ",
        po.sizes(),
        " cannot broadcast to ",
        result_shape);
    std::tie(bc_po) = expand_size(po, result_shape, "gemm_broadcast");
    if (!result.is_same(bc_po))
      result.resize_(result_shape).copy_(bc_po);
  } else {
    result.resize_(result_shape);
  }

  if (result.scalar_type() == ScalarType::Double ||
      m1.scalar_type() == ScalarType::Double ||
      m2.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, m1, m2, beta, alpha);
#else
    AT_ERROR(
        "Double datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  } else {
    xpu::oneDNN::matmul(result, m1, m2, b, attr);
  }
}

Tensor& addmm_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("addmm_out", {result, self, m1, m2}, Backend::XPU);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f) {
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
    Scalar beta,
    Scalar alpha) {
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(
      self.size(0) == m1.size(0) && self.size(1) == m2.size(1),
      "size mismatch input ",
      self.sizes(),
      " m1 ",
      m1.sizes(),
      " m2 ",
      m2.sizes());
  Tensor bias = at::empty_like(self).copy_(self);
  // oneDNN cannot support result/bias (write/read) use the same memory.
  // we will remove copy to keep performance once matmul refactor done.
  at::AtenIpexTypeXPU::addmm_out(self, bias, m1, m2, beta, alpha);
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

  at::AtenIpexTypeXPU::addmm_out(result, input, m1, m2, beta, alpha);
  return result;
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(mat2.dim() == 2, "expected 2D tensor");

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

Tensor& baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("baddbmm_", {self, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(
      self.size(0) == batch1.size(0) && self.size(1) == batch1.size(1) &&
          self.size(2) == batch2.size(2),
      "size mismatch input ",
      self.sizes(),
      " batch1 ",
      batch1.sizes(),
      " batch2 ",
      batch2.sizes());
  if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f) {
    matmul(
        self,
        batch1,
        batch2,
        at::Tensor(),
        self,
        beta.to<float>(),
        alpha.to<float>(),
        true,
        MatmulAttr::kind_with_sum);
  } else {
    matmul(
        self,
        batch1,
        batch2,
        self,
        at::Tensor(),
        beta.to<float>(),
        alpha.to<float>(),
        true,
        0);
  }
  return self;
}

Tensor& baddbmm_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  if (alpha.to<float>() != 1.f || beta.to<float>() != 1.f) {
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

Tensor baddbmm(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor r = at::empty({0}, input.options());
  at::AtenIpexTypeXPU::baddbmm_out(r, input, batch1, batch2, beta, alpha);
  return r;
}

Tensor& addbmm_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("addbmm_out", {out, self, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  Tensor b1;
  if (batch1.size(0) > 1) {
    b1 = batch1.transpose(0, 1).contiguous().view({batch1.size(1), -1});
  } else {
    b1 = batch1.view({batch1.size(1), -1});
  }
  auto b2 = batch2.view({-1, batch2.size(2)});
  at::AtenIpexTypeXPU::addmm_out(out, self, b1, b2, beta, alpha);

  return out;
}

Tensor& addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  at::AtenIpexTypeXPU::addbmm_out(self, self, batch1, batch2, beta, alpha);
  return self;
}

Tensor addbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor out = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::addbmm_out(out, self, batch1, batch2, beta, alpha);
  return out;
}

Tensor& bmm_out(Tensor& result, const Tensor& self, const Tensor& batch2) {
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
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

Tensor addmv(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    at::Scalar beta,
    at::Scalar alpha) {
  TORCH_CHECK(self.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(vec.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.size(1) == vec.size(0));

  Tensor vec_v = vec.view({vec.size(0), 1});
  Tensor self_v = self.view({self.size(0), 1});
  Tensor result = at::AtenIpexTypeXPU::addmm(self_v, mat, vec_v, beta, alpha);
  return result.view({mat.size(0)});
}

Tensor& addmv_(
    Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    at::Scalar beta,
    at::Scalar alpha) {
  TORCH_CHECK(self.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(vec.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.size(1) == vec.size(0));

  Tensor vec_v = vec.view({vec.size(0), 1});
  Tensor self_v = self.view({self.size(0), 1});
  at::AtenIpexTypeXPU::addmm_(self_v, mat, vec_v, beta, alpha);
  return self;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
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
