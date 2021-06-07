#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>
#include <ATen/CPUApplyUtils.h>
#include <core/TensorImplUtils.h>

#include <oneDNN/oneDNN.h>
#include <vector>

#include "comm/ATDispatch.h"
#include "comm/QUtil.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <core/oneMKLUtils.h>
#endif

#include <c10/util/typeid.h>


using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace impl {

bool check_broadcast(const Tensor& src, const IntArrayRef& shape){
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
  } while(src_dim);
  return true;
}

void gemm_broadcast(Tensor& result,
                    const Tensor& m1,
                    const Tensor& m2,
                    MatmulAttr attr,
                    const Tensor bias = at::Tensor()) {
  std::vector<int64_t> result_shape;
  auto dim = m1.dim();

  if(m1.is_quantized()){
    if(m2.sizes()[1] == m1.sizes()[1]){
      m2.transpose_(0,1);
    }
  }

  if (dim == 2) {
    result_shape = attr.m2_trans_ ? std::vector<int64_t>{m1.size(0), m2.size(1)} :
    std::vector<int64_t>{m1.size(0), m2.size(0)};
  } else {
    result_shape = attr.m2_trans_ ? std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(2)} :
    std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(1)};
  }

  Tensor bc_bias = bias;
  if (bias.defined() && attr.beta_ && (attr.beta_ != 1.f || m1.is_quantized())) {
    TORCH_CHECK(check_broadcast(bias, result_shape),
                "bias ", bias.sizes(), " cannot broadcast to ", result_shape);
    std::tie(bc_bias) = expand_size(bias, result_shape, "gemm_broadcast");
    if (!result.is_same(bc_bias))
      result.resize_(bc_bias.sizes()).copy_(bc_bias);
  } else {
    result.resize_(result_shape);
  }

  if(result.scalar_type() == ScalarType::Double ||
      m1.scalar_type() == ScalarType::Double ||
      m2.scalar_type() == ScalarType::Double){
#ifdef USE_ONEMKL
    Tensor _result;
    if(result.is_contiguous() && result.scalar_type() == ScalarType::Double){
      _result = result;
    } else{
      _result = at::empty_like(result, result.options().dtype(ScalarType::Double));
    }
    Tensor _m1 = m1.contiguous().to(ScalarType::Double);
    Tensor _m2 = m2.contiguous().to(ScalarType::Double);
    size_t dims = _result.dim();
    TORCH_CHECK(dims == 2 || dims == 3, "oneMKL matmul only works with 2D or 3D, got ", dims);
    TORCH_CHECK(dims == _m1.dim() && dims == _m2.dim(), "oneMKL input matrixes must have the same ranks");

    int64_t m = _result.size(-2);
    int64_t k = _m1.size(-1);
    int64_t n = _result.size(-1);
    int64_t stridea = m*k;
    int64_t strideb = k*n;
    int64_t stridec = m*n;
    int64_t mb = 1;

    if (dims == 3) {
      mb = _result.size(0);
      TORCH_CHECK(mb == _m1.size(0) && mb == _m2.size(0), "batch size mismatch, result mb: ",\
          mb, "m1 mb", _m1.size(0), " m2 mb: ", _m2.size(0));
    }

    auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
    DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::blas::row_major::gemm_batch, dpcpp_queue, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, m, n, k, attr.alpha_, (double *)_m1.data_ptr(), m, stridea, (double *)_m2.data_ptr(), k, strideb, attr.beta_, (double *)_result.data_ptr(), m, stridec, mb);
    if(!_result.is_same(result)){
      result.copy_(_result);
    }
#else
    AT_ERROR("Double datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  } else {
    xpu::oneDNN::matmul(result, m1, m2, bc_bias, attr);
  }
}
} // namespace impl


namespace AtenIpexTypeXPU {

using namespace impl;

Tensor& addmm_out(
        Tensor &result,
        const Tensor& self,
        const Tensor& m1,
        const Tensor& m2,
        Scalar beta,
        Scalar alpha) {
  MatmulAttr attr(
          alpha.to<float>(),
          beta.to<float>(),
          0,
          true);
  checkBackend("addmm_out", {result, self, m1, m2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(self.size(0) ==  m1.size(0) && self.size(1) == m2.size(1),
              "size mismatch input ", self.sizes(), " m1 ", m1.sizes(), " m2 ", m2.sizes());

    impl::gemm_broadcast(
            result,
            m1,
            m2.scalar_type() == m1.scalar_type() ? m2 : m2.to(m1.scalar_type()),
            // bias convert to fp32 for accuracy when self is fp16 or bf16
            attr,
            // bias convert to fp32 for accuracy when self is fp16 or bf16
            self.scalar_type() == ScalarType::Half ||
            self.scalar_type() == ScalarType::BFloat16
            ? self.to(ScalarType::Float) : self);

  return result;
}

Tensor& addmm_(
    Tensor& self,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);
  checkBackend("addmm_", {self, m1, m2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(self.size(0) ==  m1.size(0) && self.size(1) == m2.size(1),
              "size mismatch input ", self.sizes(), " m1 ", m1.sizes(), " m2 ", m2.sizes());

  impl::gemm_broadcast(
  self,
  m1,
  m2.scalar_type() == m1.scalar_type() ? m2 :
                     (m1.scalar_type() == ScalarType::BFloat16 || m2.scalar_type() == ScalarType::BFloat16) ? m2 :
                      m2.to(m1.scalar_type()),
  // bias convert to fp32 for accuracy when self is fp16 or bf16
  attr,
  self.scalar_type() == ScalarType::Half ||
          self.scalar_type() == ScalarType::BFloat16
      ? self.to(ScalarType::Float) : self);

  return self;
}

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);

  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  checkBackend("addmm", {input, m1, m2}, Backend::XPU);

  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16){
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  impl::gemm_broadcast(
  result,
  m1,
  m2.scalar_type() == m1.scalar_type() ? m2 :
                     (m1.scalar_type() == ScalarType::BFloat16 || m2.scalar_type() == ScalarType::BFloat16) ? m2 :
                      m2.to(m1.scalar_type()),
  // bias convert to fp32 for accuracy when input is fp16 or bf16
  attr,
  input.scalar_type() == ScalarType::Half ||
          input.scalar_type() == ScalarType::BFloat16
      ? input.to(ScalarType::Float) : input);

  return result;
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  MatmulAttr attr(1.f, 0.f, 0, true);
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(mat2.dim() == 2, "expected 2D tensor");

  auto self_dt = self.scalar_type();
  auto result_dt = result.scalar_type();
  auto mat2_dt = mat2.scalar_type();

  impl::gemm_broadcast(
  result,
  (self_dt == result_dt ||
   ((self_dt == ScalarType::BFloat16 && result_dt != ScalarType::BFloat16) || (result_dt == ScalarType::BFloat16 && self_dt != ScalarType::BFloat16))) ? self : self.to(result_dt),
  (mat2_dt == result_dt || 
   ((mat2_dt == ScalarType::BFloat16 && result_dt != ScalarType::BFloat16) || (result_dt == ScalarType::BFloat16 && mat2_dt != ScalarType::BFloat16))) ? mat2 : mat2.to(result_dt),
  attr);

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
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);
  checkBackend("baddbmm_", {self, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(self.size(0) == batch1.size(0) && \
              self.size(1) == batch1.size(1) && \
              self.size(2) == batch2.size(2),
              "size mismatch input ", self.sizes(),
              " batch1 ", batch1.sizes(), " batch2 ", batch2.sizes());
  impl::gemm_broadcast(self, batch1, batch2, attr, self);

  return self;
}

Tensor& baddbmm_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  impl::gemm_broadcast(result, batch1, batch2, attr, input);

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
  MatmulAttr attr(1, 0, 0, true);
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  impl::gemm_broadcast(result, self, batch2, attr);

  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::bmm_out(result, self, batch2);
  return result;
}

// FIXME: should not be here
Tensor linear_relu(const Tensor & input, const Tensor & weight, const Tensor & bias, Scalar beta, Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      MatmulAttr::kind_with_relu,
      false);
  RECORD_FUNCTION("linear_relu",
                  std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_relu", {input, weight, bias}, Backend::XPU);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());

    impl::gemm_broadcast(result, input, weight, attr, bias);

    return result;
  }

  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::relu(output);
}

Tensor linear_sigmoid(const Tensor & input, const Tensor & weight, const Tensor & bias, Scalar beta, Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      MatmulAttr::kind_with_sigmoid,
      false);
  RECORD_FUNCTION("linear_sigmoid",
                  std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_sigmoid", {input, weight, bias}, Backend::XPU);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());
    impl::gemm_broadcast(result, input, weight, attr, bias);

    return result;
  }
  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::sigmoid(output);

}

Tensor trans_linear(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      false);

  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  checkBackend("addmm", {input, m1, m2}, Backend::XPU);

  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16){
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  impl::gemm_broadcast(
      result,
      m1,
      m2.scalar_type() == m1.scalar_type() ? m2 :
                         (m1.scalar_type() == ScalarType::BFloat16 ||
                          m2.scalar_type() == ScalarType::BFloat16) ? m2 :
                          m2.to(m1.scalar_type()),
      // bias convert to fp32 for accuracy when input is fp16 or bf16
      attr,
      input.scalar_type() == ScalarType::Half ||
                             input.scalar_type() == ScalarType::BFloat16 ?
                             input.to(ScalarType::Float) : input);

  return result;
}

Tensor addmv(
    const Tensor & self,
    const Tensor & mat,
    const Tensor & vec,
    at::Scalar beta,
    at::Scalar alpha) {
  TORCH_CHECK(self.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(vec.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.size(1) ==  vec.size(0));

  Tensor vec_v = vec.view({vec.size(0), 1});
  Tensor self_v = self.view({self.size(0), 1});
  Tensor result = at::AtenIpexTypeXPU::addmm(self_v, mat, vec_v, beta, alpha);
  return result.view({mat.size(0)});
}

Tensor& addmv_(
    Tensor & self,
    const Tensor & mat,
    const Tensor & vec,
    at::Scalar beta,
    at::Scalar alpha) {
  TORCH_CHECK(self.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(vec.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.size(1) ==  vec.size(0));

  Tensor vec_v = vec.view({vec.size(0), 1});
  Tensor self_v = self.view({self.size(0), 1});
  at::AtenIpexTypeXPU::addmm_(self_v, mat, vec_v, beta, alpha);
  return self;
}

Tensor matmul_sum(
    Tensor& accumu,
    const Tensor& m1,
    const Tensor& m2,
    at::Scalar beta) {
  Tensor result, bias;

  TORCH_CHECK(m1.dim() == 2 || m2.dim() == 2, "expected 2D tensor");
  if (accumu.dim() == 1) {
    if (beta.to<float>() == 1.0f) {
      result = at::empty({0}, m1.options());
      bias = accumu;
    } else {
      std::tie(result) = expand_size(
          accumu, m1.dim() == 2 ? m1.sizes() : m2.sizes());
    }
  } else {
    result = accumu;
  }

  // collaps a,b,c to axb,c for m1
  // FIXME: no m2 to collaps so far
  std::vector<int64_t> m1_shape, r_shape;
  if (m1.dim() != 2) {
    for (int i = 0; i < m1.sizes().size() - 1; i++) {
      m1_shape.push_back(m1.sizes()[i]);
      r_shape.push_back(m1.sizes()[i]);
    }
    m1_shape.push_back(m1.sizes()[m1.sizes().size() - 1]);
    r_shape.push_back(m2.sizes()[1]);

    std::vector<int64_t> sizes = m1.sizes().vec();
    std::vector<int64_t> strides = m1.strides().vec();
    at::collapse_dims(sizes.data(), strides.data(), m1.dim(), m1.dim() - 1);
    m1.resize_({sizes.data()[0], sizes.data()[1]});
  }

  MatmulAttr attr(
      1.f,
      beta.to<float>(),
      0,
      true);

  impl::gemm_broadcast(
      result,
      m1,
      m2.scalar_type() == m1.scalar_type() ? m2 : m2.to(m1.scalar_type()),
      attr,
      bias);

  if (r_shape.size()) {
    m1.resize_(m1_shape);
    result.resize_(r_shape);
  }

  return result;
}

Tensor& trans_baddbmm_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      false);
  checkBackend("trans_baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  impl::gemm_broadcast(result, batch1, batch2, attr, input);

  return result;
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor addmm(
  const Tensor& input,
  const Tensor& m1,
  const Tensor& m2,
  Scalar beta,
  Scalar alpha) {
  MatmulAttr attr(
          alpha.to<float>(),
          beta.to<float>(),
          0,
          true);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");

  checkBackend("addmm", m1, Backend::QuantizedXPU);

  Tensor result;
  if(input.is_quantized()){
    result = _empty_affine_quantized({0},
                                     device(kXPU).dtype(input.scalar_type()),
                                     1.f,
                                     static_cast<int>(0),
                                     MemoryFormat::Contiguous);
  } else {
    result = at::empty({0}, input.options());
  }

  impl::gemm_broadcast(result, m1, m2, attr, input);

  return result;
}

Tensor trans_linear(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  MatmulAttr attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      false);

  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16){
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  impl::gemm_broadcast(result, m1, m2, attr, input);

  return result;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
