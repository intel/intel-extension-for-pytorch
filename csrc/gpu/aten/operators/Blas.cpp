#include "Blas.h"
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <runtime/Device.h>
#include <runtime/Utils.h>
#include "BlasImpl.h"
#include "utils/CustomOperatorRegistration.h"

#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include "comm/RegisterUtils.h"
#endif

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
#include "XeGemm.h"
#endif

#define RECORD_ONEDNN_FUNCTION_IMPL(F)                     \
  char str__[100];                                         \
  sprintf(str__, "onednn_%s(%d, %d, %d)", "" #F, m, n, k); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;

// result = beta * self + alpha * (mat1 * mat2)
Tensor& addmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    at::Tensor& result) {
  checkBackend("addmm_out", {result, self, mat1, mat2}, Backend::XPU);
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.dtype() == mat2.dtype(),
      "expected mat1 and mat2 to have the same dtype, but got: ",
      mat1.dtype(),
      " != ",
      mat2.dtype())
  int m = mat1.sizes()[0];
  int n = mat2.sizes()[1];
  int k = mat2.sizes()[0];
  TORCH_CHECK(
      mat1.sizes()[1] == k,
      "mat1 and mat2 shapes cannot be multiplied (",
      m,
      "x",
      mat1.sizes()[1],
      " and ",
      k,
      "x",
      n,
      ")");

  std::vector<int64_t> result_shape = {mat1.size(0), mat2.size(1)};
  result.resize_(result_shape);
  if (result.numel() == 0)
    return result;
  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    return at::mul_out(
        result,
        self.expand(result.sizes()),
        at::native::wrapped_scalar_tensor(at::Scalar(beta)));
  }
  TORCH_CHECK(
      are_expandable(self.sizes(), result_shape),
      "addmm_out input must be expanable to:",
      result_shape,
      " but got:",
      self.sizes());

  // special case
  if (alpha.to<float>() == 0.f) {
    if (self.defined() && beta.to<float>() != 0.f) {
      result = at::mul_out(
          result, self, at::native::wrapped_scalar_tensor(at::Scalar(beta)));
    } else {
      result.zero_();
    }
    return result;
  }

  // complex case
  if (mat1.is_complex()) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, self, mat1, mat2, beta, alpha);
    return result;
#else
    AT_ERROR(
        "Complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(mat1, mat2, self)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    if (alpha.to<float>() == 1.f && self.dim() == 1) {
      HGEMM_XETLA<1> hgemm_bias(arch_tag);
      auto policy =
          hgemm_bias.add_operands(result, mat1, mat2)
              .add_epilogue(self, EpilogueType::BIAS, beta.to<float>())
              .build();
      if (policy.valid()) {
        if (policy.run() == torch_ipex::xpu::xetla::GemmStatus::kSuccess)
          return result;
      }
    } else if (
        self.dim() == 2 && self.sizes()[0] == mat1.sizes()[0] &&
        self.sizes()[1] == mat2.sizes()[1]) {
      HGEMM_XETLA<1> hgemm_res(arch_tag);
      auto policy =
          hgemm_res.add_operands(result, mat1, mat2, alpha.to<float>())
              .add_epilogue(self, EpilogueType::RES_ADD, beta.to<float>())
              .build();
      if (policy.valid()) {
        if (policy.run() == torch_ipex::xpu::xetla::GemmStatus::kSuccess)
          return result;
      }
    }
  }
#endif
  // general case
  RECORD_ONEDNN_FUNCTION_IMPL(addmm)
  Tensor bias = at::Tensor();
  Attr attr;
  float beta_ = beta.to<float>();
  if (beta_ == 0.f) {
    if (alpha.to<float>() != 1.f) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
    }
  } else {
    if (alpha.to<float>() == 1.f && beta_ == 1.f) {
      bias = self;
    } else {
      Tensor binary = self.dim() == 1 ? self.unsqueeze(0) : self;
      // Tensor binary = self.expand_as(result);
      // For post-binary-add, onednn needs binary scale=1.f
      // Thus we need the following transformation
      // alpha * matmul(mat1, mat2) + beta * binary
      // beta * (alpha/beta * matmul(src, wei) + binary)
      float alpha_ = alpha.to<float>() / beta_;
      if (alpha_ != 1.f)
        attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      attr.append_post_binary(attr.kind_with_binary_add, binary);
      if (beta_ != 1.f)
        attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }
  torch_ipex::xpu::oneDNN::matmul(result, mat1, mat2, bias, true, attr);
  return result;
}

Tensor& _addmm_activation_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    bool use_gelu,
    at::Tensor& result) {
  addmm_out(self, mat1, mat2, beta, alpha, result);
  if (use_gelu) {
    at::gelu_(result);
  } else {
    at::relu_(result);
  }
  return result;
}

Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& result) {
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      self.dtype() == mat2.dtype(),
      "expected self and mat2 to have the same dtype, but got: ",
      self.dtype(),
      " != ",
      mat2.dtype())

  int m = self.sizes()[0];
  int n = mat2.sizes()[1];
  int k = mat2.sizes()[0];
  TORCH_CHECK(
      self.sizes()[1] == k,
      "mat1 and mat2 shapes cannot be multiplied (",
      m,
      "x",
      self.sizes()[1],
      " and ",
      k,
      "x",
      n,
      ")");

  result.resize_({self.size(0), mat2.size(1)});
  if (self.numel() == 0 || mat2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  if (self.is_complex()) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, result, self, mat2, Scalar(0), Scalar(1));
    return result;
#else
    AT_ERROR(
        "Complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (hgemm_xetla_valid(self, mat2)) {
    gpu::xetla::gpu_arch arch_tag = gpu::xetla::get_xetla_current_arch_tag();
    HGEMM_XETLA<0> hgemm_common(arch_tag);
    auto policy = hgemm_common.add_operands(result, self, mat2).build();
    if (policy.valid()) {
      if (policy.run() == torch_ipex::xpu::xetla::GemmStatus::kSuccess)
        return result;
    }
  }
#endif

  RECORD_ONEDNN_FUNCTION_IMPL(mm_out)
  torch_ipex::xpu::oneDNN::matmul(
      result, self, mat2, at::Tensor(), true, Attr());
  return result;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::mm_out(self, mat2, result);
  return result;
}

// result = beta * input + alpha * (batch1 @ batch2)
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
  TORCH_CHECK(
      batch1.dtype() == batch2.dtype(),
      "expected batch1 and batch2 to have the same dtype, but got: ",
      batch1.dtype(),
      " != ",
      batch2.dtype())

  std::vector<int64_t> result_shape = {
      batch1.size(0), batch1.size(1), batch2.size(2)};
  result.resize_(result_shape);
  if (result.numel() == 0)
    return result;

  TORCH_CHECK(
      are_expandable(input.sizes(), result_shape),
      "baddbmm_out input must be expanable to:",
      result_shape,
      " but got:",
      input.sizes());

  // special case
  if (alpha.to<float>() == 0.f || batch1.numel() == 0 || batch2.numel() == 0) {
    if (input.defined() && beta.to<float>() != 0.f) {
      result = at::AtenIpexTypeXPU::mul_out(
          input, at::native::wrapped_scalar_tensor(at::Scalar(beta)), result);
    } else {
      result.zero_();
    }
    return result;
  }

  // complex case
  if (batch1.is_complex()) {
#ifdef USE_ONEMKL
    impl::mkl_baddbmm(result, input, batch1, batch2, beta, alpha);
    return result;
#else
    AT_ERROR(
        "Complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

  // general case
  Attr attr;
  float beta_ = beta.to<float>();
  if (beta_ == 0.f) {
    if (alpha.to<float>() != 1.f) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
    }
  } else {
    Tensor binary = input;
    while (binary.dim() < 3) {
      binary.unsqueeze_(0);
    }
    float alpha_ = alpha.to<float>() / beta_;
    if (alpha_ != 1.f)
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
    attr.append_post_binary(attr.kind_with_binary_add, binary);
    if (beta_ != 1.f)
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
  }
  torch_ipex::xpu::oneDNN::matmul(
      result, batch1, batch2, at::Tensor(), true, attr);
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

  out.resize_({batch1.size(1), batch2.size(2)});
  if (alpha.to<float>() == 0.f || batch1.numel() == 0 || batch2.numel() == 0) {
    out.resize_({batch1.size(1), batch2.size(2)});
    if (out.numel() == 0)
      return out;

    if (self.defined() && beta.to<float>() != 0.f) {
      out = at::mul_out(
          out, self, at::native::wrapped_scalar_tensor(at::Scalar(beta)));
    } else {
      out.zero_();
    }
    return out;
  }

  Tensor b1;
  if (batch1.size(0) > 1) {
    b1 = batch1.transpose(0, 1).contiguous().view({batch1.size(1), -1});
  } else {
    b1 = batch1.contiguous().view({batch1.size(1), -1});
  }
  auto b2 = batch2.contiguous().view({-1, batch2.size(2)});
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

Tensor& bmm_out(const Tensor& self, const Tensor& batch2, Tensor& result) {
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  result.resize_({self.size(0), self.size(1), batch2.size(2)});
  if (self.numel() == 0 || batch2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  if (self.is_complex()) {
#ifdef USE_ONEMKL
    return at::AtenIpexTypeXPU::baddbmm_out(
        result, self, batch2, Scalar(0), Scalar(1), result);
#else
    AT_ERROR(
        "Complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }
  torch_ipex::xpu::oneDNN::matmul(
      result, self, batch2, at::Tensor(), true, Attr());
  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::bmm_out(self, batch2, result);
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
  if (mat.numel() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else {
      at::mul_out(
          const_cast<Tensor&>(out),
          self,
          at::native::wrapped_scalar_tensor(at::Scalar(beta)));
    }
  } else {
    Tensor vec_v = vec.view({vec.size(0), 1});
    at::AtenIpexTypeXPU::addmm_out(self_v, mat, vec_v, beta, alpha, out);
    out.resize_({mat.size(0)});
  }
  return out;
}

Tensor& addmv_(
    Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
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
  self_v.addmm_(mat, vec_v, beta, alpha);
  return self;
}

Tensor tensordot(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2) {
  TORCH_CHECK(
      dims1.size() == dims2.size(),
      "both dimension lists should have same length");
  TORCH_CHECK(
      input1.dtype() == input2.dtype(),
      "expected input1 and input2 to have the same dtype, but got: ",
      input1.dtype(),
      " != ",
      input2.dtype())
  int64_t csize = 1; // total size of the contracted dimensions
  Tensor t1 = input1;
  Tensor t2 = input2;
  for (const auto i : c10::irange(dims1.size())) {
    int s1 = input1.size(dims1[i]);
    int s2 = input2.size(dims2[i]);
    if (s2 == 1) { // broadcasted dimensions can be summed right away
      t1 = t1.sum(dims1[i], true);
    } else if (s1 == 1) {
      t2 = t2.sum(dims2[i], true);
    } else {
      TORCH_CHECK(
          s1 == s2,
          "contracted dimensions need to match, but first has size ",
          s1,
          " in dim ",
          dims1[i],
          " and second has size ",
          s2,
          " in dim ",
          dims2[i]);
      csize *= s1;
    }
  }
  auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
  auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
  std::vector<int64_t> p1, p2,
      rsizes; // p1, p2: input permutations, rsizes: sizes of the result
  p1.reserve(input1.dim());
  p2.reserve(input2.dim());
  rsizes.reserve(input1.dim() + input2.dim() - (int64_t)dims1.size());
  int64_t size1 = 1; // number of non-contracted elements in input1
  int64_t size2 = 1; // number of non-contracted elements in input2

  // fill the permutations and compute sizes
  for (const auto i : c10::irange(input1.dim())) {
    if (!cdims1[i]) {
      p1.emplace_back(i);
      size1 *= t1.size(i);
      rsizes.emplace_back(t1.size(i));
    }
  }
  for (const auto x : dims1) {
    p1.emplace_back(x);
  }
  for (const auto x : dims2) {
    p2.emplace_back(x);
  }
  for (const auto i : c10::irange(input2.dim())) {
    if (!cdims2[i]) {
      p2.emplace_back(i);
      size2 *= t2.size(i);
      rsizes.emplace_back(t2.size(i));
    }
  }
  // permut and reshape for matrix multiplication
  t1 = t1.permute(p1).reshape({size1, csize});
  t2 = t2.permute(p2).reshape({csize, size2});
  // multiply and reshape to target size
  return at::mm(t1, t2).reshape(rsizes);
}

Tensor& tensordot_out(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2,
    Tensor& result) {
  Tensor result_tmp = at::tensordot(input1, input2, dims1, dims2);
  auto result_dtype = result_tmp.scalar_type();
  auto output_tensor_dtype = result.scalar_type();
  auto output_device = result.device();
  auto input1_device = input1.device();
  auto input2_device = input2.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
      (output_device == input1_device) && (input1_device == input2_device),
      "tensordot: Expected the output and input tensors to be on the "
      "same device, but got the output tensor on ",
      output_device,
      ", input tensor a on ",
      input1_device,
      ", and input tensor b on ",
      input2_device);
  // check if the computed result has the same dtype as the out tensor
  // (because tensordot does not support type promotion)
  TORCH_CHECK(
      result_dtype == output_tensor_dtype,
      "tensordot",
      ": Expected the output tensor to have dtype ",
      result_dtype,
      ", but got an output tensor with dtype ",
      output_tensor_dtype);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

/************************** matmul fusion path **************************/
#define IPEX_MATMUL_DEFINATION(func)                                      \
  at::Tensor matmul_##func(                                               \
      const at::Tensor& tensor1, const at::Tensor& tensor2) {             \
    RECORD_FUNCTION(                                                      \
        "matmul_" #func, std::vector<c10::IValue>({tensor1, tensor2}));   \
    Attr attr;                                                            \
    attr.append_post_eltwise(                                             \
        /* scale */ 1.f,                                                  \
        /* alpha */ 0.f,                                                  \
        /* beta */ 0.f,                                                   \
        attr.kind_with_##func);                                           \
    bool is_fused;                                                        \
    Tensor result;                                                        \
    return matmul_fusion_variants(                                        \
        result, tensor1, tensor2, true, attr, is_fused);                  \
  }                                                                       \
  at::Tensor t_matmul_##func(                                             \
      const at::Tensor& tensor2, const at::Tensor& tensor1) {             \
    RECORD_FUNCTION(                                                      \
        "t_matmul_" #func, std::vector<c10::IValue>({tensor1, tensor2})); \
    Attr attr;                                                            \
    attr.append_post_eltwise(                                             \
        /* scale */ 1.f,                                                  \
        /* alpha */ 0.f,                                                  \
        /* beta */ 0.f,                                                   \
        attr.kind_with_##func);                                           \
    bool is_fused;                                                        \
    Tensor result;                                                        \
    return matmul_fusion_variants(                                        \
        result, tensor1, tensor2, false, attr, is_fused);                 \
  }

IPEX_MATMUL_DEFINATION(sqrt)
IPEX_MATMUL_DEFINATION(abs)
IPEX_MATMUL_DEFINATION(tanh)
IPEX_MATMUL_DEFINATION(square)
IPEX_MATMUL_DEFINATION(exp)
IPEX_MATMUL_DEFINATION(log)
IPEX_MATMUL_DEFINATION(round)
IPEX_MATMUL_DEFINATION(sigmoid)
IPEX_MATMUL_DEFINATION(relu)
IPEX_MATMUL_DEFINATION(mish)

at::Tensor matmul_hardswish(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) {
  RECORD_FUNCTION(
      "matmul_hardswish", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f / 6.f,
      /* beta */ 1.f / 2.f,
      attr.kind_with_hardswish);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor t_matmul_hardswish(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1) {
  RECORD_FUNCTION(
      "t_matmul_hardswish", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f / 6.f,
      /* beta */ 1.f / 2.f,
      attr.kind_with_hardswish);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor matmul_log_sigmoid(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) {
  RECORD_FUNCTION(
      "matmul_log_sigmoid", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ -1.f,
      /* beta */ 0.f,
      attr.kind_with_soft_relu);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}
at::Tensor t_matmul_log_sigmoid(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1) {
  RECORD_FUNCTION(
      "t_matmul_log_sigmoid", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ -1.f,
      /* beta */ 0.f,
      attr.kind_with_soft_relu);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor matmul_silu(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  RECORD_FUNCTION("matmul_silu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_swish);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor matmul_gelu(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    c10::string_view approximate) {
  RECORD_FUNCTION("matmul_gelu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  algorithm algo;
  if (approximate == "none") {
    algo = attr.kind_with_gelu_erf;
  } else if (approximate == "tanh") {
    algo = attr.kind_with_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
  }
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      algo);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor matmul_hardsigmoid(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) {
  RECORD_FUNCTION(
      "matmul_hardsigmoid", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f / 6.,
      /* beta */ 1.f / 2.,
      attr.kind_with_hardsigmoid);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor matmul_pow(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    Scalar exponent) {
  RECORD_FUNCTION("matmul_pow", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ exponent.toFloat(),
      attr.kind_with_pow);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor matmul_leaky_relu(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    Scalar negative_slope) {
  RECORD_FUNCTION(
      "matmul_leaky_relu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ negative_slope.toFloat(),
      /* beta */ 0.f,
      attr.kind_with_relu);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor matmul_hardtanh(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    Scalar minval,
    Scalar maxval) {
  RECORD_FUNCTION(
      "matmul_hardtanh", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ minval.toFloat(),
      /* beta */ maxval.toFloat(),
      attr.kind_with_clip);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

at::Tensor matmul_elu(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  RECORD_FUNCTION("matmul_elu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ alpha.toFloat(),
      /* beta */ 1.f,
      attr.kind_with_elu);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
}

// outplace accumul1 tensor
// res = m1 * m2 + beta * accumu
at::Tensor matmul_add(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor& accumul1,
    Scalar beta1) {
  RECORD_FUNCTION(
      "matmul_add", std::vector<c10::IValue>({tensor1, tensor2, accumul1}));
  Attr attr;
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul1, beta1.to<float>());
  bool is_fused;
  Tensor result;
  result =
      matmul_fusion_variants(result, tensor1, tensor2, true, attr, is_fused);
  if (!is_fused) {
    result += at::mul(accumul1, beta1);
  }
  return result;
}

// inplace accumul tensor
// accumu = m1 * m2 + beta * accumul
at::Tensor matmul_sum(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor& accumul,
    Scalar beta) {
  RECORD_FUNCTION(
      "matmul_sum", std::vector<c10::IValue>({tensor1, tensor2, accumul}));
  Attr attr;
  attr.append_post_sum(beta.to<float>());
  bool is_fused;
  return matmul_fusion_variants(
      accumul, tensor1, tensor2, true, attr, is_fused);
}

// res = m1 * m2.transpose()
at::Tensor trans_matmul(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1) {
  RECORD_FUNCTION("trans_matmul", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

// res = m1 * m2.t()
at::Tensor t_matmul(const at::Tensor& tensor2, const at::Tensor& tensor1) {
  RECORD_FUNCTION("t_matmul", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

// res = m1 * m2.t() + beta * accumu
at::Tensor t_matmul_add(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1) {
  RECORD_FUNCTION(
      "t_matmul_add", std::vector<c10::IValue>({tensor1, tensor2, accumul1}));

  Attr attr;
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul1, beta1.to<float>());
  bool is_fused;
  Tensor result;
  result =
      matmul_fusion_variants(result, tensor1, tensor2, false, attr, is_fused);
  if (!is_fused) {
    result += at::mul(accumul1, beta1);
  }
  return result;
}

// res = GELU(m1 * m2.t() + beta * accumu)
at::Tensor t_matmul_add_gelu(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1,
    c10::string_view approximate) {
  RECORD_FUNCTION(
      "t_matmul_add_gelu",
      std::vector<c10::IValue>({tensor1, tensor2, accumul1}));
  Attr attr;
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul1, beta1.to<float>());
  algorithm algo;
  if (approximate == "none") {
    algo = attr.kind_with_gelu_erf;
  } else if (approximate == "tanh") {
    algo = attr.kind_with_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
  }
  attr.append_post_eltwise(
      /* gelu_scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      algo);
  bool is_fused;
  Tensor result;
  result =
      matmul_fusion_variants(result, tensor1, tensor2, false, attr, is_fused);
  if (!is_fused) {
    result = at::gelu((result + at::mul(accumul1, beta1)), approximate);
  }
  return result;
}

// res = alpha * (m1 * m2.t()) + beta1 * accumu1 + beta2 * accumu2
at::Tensor t_matmul_add_add(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1,
    at::Tensor& accumul2,
    Scalar beta2) {
  RECORD_FUNCTION(
      "t_matmul_add_add",
      std::vector<c10::IValue>({tensor1, tensor2, accumul1, accumul2}));
  bool is_fused;
  Attr attr;
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul1, beta1.to<float>());
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul2, beta2.to<float>());
  Tensor result;
  result =
      matmul_fusion_variants(result, tensor1, tensor2, false, attr, is_fused);
  if (!is_fused) {
    result += at::mul(accumul1, beta1) + at::mul(accumul2, beta2);
  }
  return result;
}

// res = (m1 * m2.transpose()) / oscale
at::Tensor trans_matmul_div_scalar(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    Scalar oscale) {
  RECORD_FUNCTION(
      "trans_matmul_div", std::vector<c10::IValue>({tensor1, tensor2}));
  TORCH_CHECK(oscale.to<float>() != 0, "expected non-zero value of oscale");
  Attr attr;
  attr.append_post_eltwise( // append post linear
      /* scale */ 1.f,
      /* alpha */ 1.f / oscale.to<float>(),
      /* beta */ 0.f,
      attr.kind_with_linear);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor trans_matmul_div_tensor(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    const at::Tensor& div) {
  RECORD_FUNCTION(
      "trans_matmul_div", std::vector<c10::IValue>({tensor1, tensor2, div}));
  Attr attr;
  attr.append_post_binary(attr.kind_with_binary_div, div);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

// res = (m1 * m2.transpose()) / oscale + accumul
at::Tensor trans_matmul_div_add(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    Scalar oscale,
    Tensor& accumul,
    Scalar alpha) {
  RECORD_FUNCTION(
      "trans_matmul_div_add",
      std::vector<c10::IValue>({tensor1, tensor2, accumul}));
  TORCH_CHECK(oscale.to<float>() != 0, "expected non-zero value of oscale");
  Attr attr;
  attr.append_post_eltwise( // append post linear
      /* scale */ 1.f,
      /* alpha */ 1.f / oscale.to<float>(),
      /* beta */ 0.f,
      attr.kind_with_linear);
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul, alpha.to<float>());
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

// res = (m1 * m2.transpose() + accumul) / oscale
at::Tensor trans_matmul_add_div(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    Scalar oscale,
    Tensor& accumul,
    Scalar alpha) {
  RECORD_FUNCTION(
      "trans_matmul_add_div",
      std::vector<c10::IValue>({tensor1, tensor2, accumul}));
  TORCH_CHECK(oscale.to<float>() != 0, "expected non-zero value of oscale");
  Attr attr;
  attr.append_scale_binary(
      attr.kind_with_binary_add, accumul, alpha.to<float>());
  attr.append_post_eltwise( // append post linear
      /* scale */ 1.f,
      /* alpha */ 1.f / oscale.to<float>(),
      /* beta */ 0.f,
      attr.kind_with_linear);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

// res = ((m1 * m2.transpose() + accumul1) / oscale) + accumul2
at::Tensor trans_matmul_add_div_add(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    const c10::optional<Scalar>& oscale,
    const c10::optional<Tensor>& accumul1,
    Scalar alpha1,
    const c10::optional<Tensor>& accumul2,
    Scalar alpha2) {
  RECORD_FUNCTION(
      "trans_matmul_add_div_add",
      std::vector<c10::IValue>({tensor1, tensor2, accumul1, oscale, accumul2}));
  Attr attr;
  if (accumul1.has_value()) {
    attr.append_scale_binary(
        attr.kind_with_binary_add, accumul1.value(), alpha1.to<float>());
  }
  if (oscale.has_value()) {
    TORCH_CHECK(
        oscale.value().to<float>() != 0, "expected non-zero value of oscale");
    attr.append_post_eltwise( // append post linear
        /* scale */ 1.f,
        /* alpha */ 1.f / oscale.value().to<float>(),
        /* beta */ 0.f,
        attr.kind_with_linear);
  }
  if (accumul2.has_value()) {
    attr.append_scale_binary(
        attr.kind_with_binary_add, accumul2.value(), alpha2.to<float>());
  }
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_silu(const at::Tensor& tensor2, const at::Tensor& tensor1) {
  RECORD_FUNCTION(
      "t_matmul_silu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_swish);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_hardsigmoid(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1) {
  RECORD_FUNCTION(
      "t_matmul_hardsigmoid", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f / 6.,
      /* beta */ 1.f / 2.,
      attr.kind_with_hardsigmoid);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_pow(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    Scalar exponent) {
  RECORD_FUNCTION("t_matmul_pow", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ exponent.toFloat(),
      attr.kind_with_pow);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_leaky_relu(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    Scalar negative_slope) {
  RECORD_FUNCTION(
      "t_matmul_leaky_relu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ negative_slope.toFloat(),
      /* beta */ 0.f,
      attr.kind_with_relu);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_hardtanh(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    Scalar minval,
    Scalar maxval) {
  RECORD_FUNCTION(
      "t_matmul_hardtanh", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ minval.toFloat(),
      /* beta */ maxval.toFloat(),
      attr.kind_with_clip);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_elu(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  RECORD_FUNCTION("t_matmul_elu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ alpha.toFloat(),
      /* beta */ 1.f,
      attr.kind_with_elu);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

at::Tensor t_matmul_gelu(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    c10::string_view approximate) {
  RECORD_FUNCTION(
      "t_matmul_gelu", std::vector<c10::IValue>({tensor1, tensor2}));
  Attr attr;
  algorithm algo;
  if (approximate == "none") {
    algo = attr.kind_with_gelu_erf;
  } else if (approximate == "tanh") {
    algo = attr.kind_with_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
  }
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      algo);
  bool is_fused;
  Tensor result;
  return matmul_fusion_variants(
      result, tensor1, tensor2, false, attr, is_fused);
}

Tensor matmul_bias_out(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(std::in_place);

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight);
  }
  if (input.dim() == 3 && bias->defined() && input.is_contiguous() &&
      !input.is_xla()) {
    // Also hit the fused path for contiguous 3D input, if not using xla
    // backend. Reshaping/flattening has some performance implications on xla.
    const auto input_sizes = input.sym_sizes();
    const auto result = at::addmm(
        *bias,
        input.view_symint({input_sizes[0] * input_sizes[1], input_sizes[2]}),
        weight);
    return result.view_symint(
        {input_sizes[0], input_sizes[1], result.sym_size(1)});
  }
  auto output = at::matmul(input, weight);
  if (bias->defined()) {
    output = at::add(output, *bias);
  }
  return output;
}
#ifdef USE_OVERRIDE_OP
void addmm_meta(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& output) {
  TORCH_CHECK(
      self.scalar_type() == mat2.scalar_type(),
      "self and mat2 must have the same dtype, but got ",
      self.scalar_type(),
      " and ",
      mat2.scalar_type());
  TORCH_CHECK(
      mat1.scalar_type() == mat2.scalar_type(),
      "mat1 and mat2 must have the same dtype, but got ",
      mat1.scalar_type(),
      " and ",
      mat2.scalar_type());
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  if (output.defined()) {
    at::AtenIpexTypeXPU::resize_out(
        output, {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options());
  } else {
    output = at::AtenIpexTypeXPU::create_out(
        {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options());
  }
}
Tensor addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor r;
  addmm_meta(self, mat1, mat2, beta, alpha, r);
  at::AtenIpexTypeXPU::addmm_out(self, mat1, mat2, beta, alpha, r);
  return r;
}

void _addmm_activation_meta(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    bool use_gelu,
    Tensor& output) {
  addmm_meta(self, mat1, mat2, beta, alpha, output);
}
Tensor _addmm_activation(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    bool use_gelu) {
  Tensor r;
  _addmm_activation_meta(self, mat1, mat2, beta, alpha, use_gelu, r);
  at::AtenIpexTypeXPU::_addmm_activation_out(
      self, mat1, mat2, beta, alpha, use_gelu, r);
  return r;
}

void addmv_meta(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& output) {
  TORCH_CHECK(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());

  TORCH_CHECK(
      mat.size(1) == vec.size(0) &&
          (mat.size(0) == self.numel() || self.numel() == 1),
      "size mismatch, got input (",
      self.size(0),
      "), mat (",
      mat.size(0),
      "x",
      mat.size(1),
      "), vec (",
      vec.size(0),
      ")");

  if (output.defined()) {
    at::AtenIpexTypeXPU::resize_out(
        output, IntArrayRef(mat.sizes().data(), 1), {}, vec.options());
  } else {
    output = at::AtenIpexTypeXPU::create_out(
        IntArrayRef(mat.sizes().data(), 1), {}, vec.options());
  }
}

at::Tensor addmv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  Tensor out;
  addmv_meta(self, mat, vec, beta, alpha, out);
  return addmv_out(self, mat, vec, beta, alpha, out);
}

void addmm__meta(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& output) {
  TORCH_CHECK(
      self.scalar_type() == mat2.scalar_type(),
      "self and mat2 must have the same dtype, but got ",
      self.scalar_type(),
      " and ",
      mat2.scalar_type());
  TORCH_CHECK(
      mat1.scalar_type() == mat2.scalar_type(),
      "mat1 and mat2 must have the same dtype, but got ",
      mat1.scalar_type(),
      " and ",
      mat2.scalar_type());
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  check_inplace(output, {mat1.sizes()[0], mat2.sizes()[1]}, mat1.options());
}
#endif
} // namespace AtenIpexTypeXPU

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("matmul_bias_out", at::AtenIpexTypeXPU::matmul_bias_out);
}
} // namespace

namespace AtenIpexTypeQuantizedXPU {

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    const Scalar& beta,
    const Scalar& alpha) {
  checkBackend("addmm", m1, Backend::QuantizedXPU);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  std::vector<int64_t> result_shape = {m1.size(0), m2.size(1)};
  Tensor result;
  if (input.is_quantized()) {
    result = at::_empty_affine_quantized(
        result_shape,
        device(kXPU).dtype(input.scalar_type()),
        1.f,
        static_cast<int>(0),
        MemoryFormat::Contiguous);
  } else {
    result = at::empty(result_shape, input.options());
  }

  Attr attr;
  if (!input.is_quantized() && beta.to<float>() == 1.f) {
    Tensor binary = input.dim() == 1 ? input.unsqueeze(0) : input;
    attr.append_post_binary(attr.kind_with_binary_add, binary);
  } else {
    c10::MaybeOwned<Tensor> accumu =
        expand_size(input, result_shape, "gemm_broadcast");
    result.copy_(*accumu);
    attr.append_post_sum(/* sum_scale */ beta.to<float>());
  }
  torch_ipex::xpu::oneDNN::matmul(result, m1, m2, at::Tensor(), true, attr);
  return result;
}

#define IPEX_OP_REGISTER_MATMUL(op)             \
  IPEX_OP_REGISTER("matmul_" #op, matmul_##op); \
  IPEX_OP_REGISTER("t_matmul_" #op, t_matmul_##op);

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("matmul_add", matmul_add);
  IPEX_OP_REGISTER("trans_matmul", trans_matmul);
  IPEX_OP_REGISTER("t_matmul", t_matmul);
  IPEX_OP_REGISTER("t_matmul_add", t_matmul_add);
  IPEX_OP_REGISTER("t_matmul_add_gelu", t_matmul_add_gelu);
  IPEX_OP_REGISTER("t_matmul_add_add", t_matmul_add_add);
  IPEX_OP_REGISTER("trans_matmul_div", trans_matmul_div_scalar);
  IPEX_OP_REGISTER("trans_matmul_div.Tensor", trans_matmul_div_tensor);
  IPEX_OP_REGISTER("trans_matmul_div_add", trans_matmul_div_add);
  IPEX_OP_REGISTER("trans_matmul_add_div", trans_matmul_add_div);
  IPEX_OP_REGISTER("trans_matmul_add_div_add", trans_matmul_add_div_add);
  IPEX_OP_REGISTER_MATMUL(sqrt);
  IPEX_OP_REGISTER_MATMUL(abs);
  IPEX_OP_REGISTER_MATMUL(tanh);
  IPEX_OP_REGISTER_MATMUL(square);
  IPEX_OP_REGISTER_MATMUL(exp);
  IPEX_OP_REGISTER_MATMUL(log);
  IPEX_OP_REGISTER_MATMUL(round);
  IPEX_OP_REGISTER_MATMUL(sigmoid);
  IPEX_OP_REGISTER_MATMUL(relu);
  IPEX_OP_REGISTER_MATMUL(log_sigmoid);
  IPEX_OP_REGISTER_MATMUL(hardswish);
  IPEX_OP_REGISTER_MATMUL(mish);
  IPEX_OP_REGISTER_MATMUL(silu);
  IPEX_OP_REGISTER_MATMUL(gelu);
  IPEX_OP_REGISTER_MATMUL(hardsigmoid);
  IPEX_OP_REGISTER_MATMUL(pow);
  IPEX_OP_REGISTER_MATMUL(leaky_relu);
  IPEX_OP_REGISTER_MATMUL(hardtanh);
  IPEX_OP_REGISTER_MATMUL(elu);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {
at::Tensor& wrapper_XPU_out__addmm_activation_out(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    bool use_gelu,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out__addmm_activation_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out__addmm_activation_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat1, "wrapper_XPU_out__addmm_activation_out", "mat1");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_out__addmm_activation_out", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::_addmm_activation_out(
      self, mat1, mat2, beta, alpha, use_gelu, out);
}

at::Tensor& wrapper_XPU_out_addmm_out(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_addmm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_addmm_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat1, "wrapper_XPU_out_addmm_out", "mat1");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_out_addmm_out", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addmm_out(self, mat1, mat2, beta, alpha, out);
}

at::Tensor& wrapper_XPU_out_addmv_out(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_addmv_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_addmv_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat, "wrapper_XPU_out_addmv_out", "mat");
  c10::impl::check_and_update_common_device(
      common_device, vec, "wrapper_XPU_out_addmv_out", "vec");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addmv_out(self, mat, vec, beta, alpha, out);
}

at::Tensor wrapper_XPU_mm(const at::Tensor& self, const at::Tensor& mat2) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_mm", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_mm", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::AtenIpexTypeXPU::mm(self, mat2);
}

at::Tensor& wrapper_XPU_out_mm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_mm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_mm_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_out_mm_out", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::AtenIpexTypeXPU::mm_out(self, mat2, out);
}

at::Tensor wrapper_XPU__baddbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__baddbmm", "self");
  c10::impl::check_and_update_common_device(
      common_device, batch1, "wrapper_XPU__baddbmm", "batch1");
  c10::impl::check_and_update_common_device(
      common_device, batch2, "wrapper_XPU__baddbmm", "batch2");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto _self = AtenIpexTypeXPU::to_plain_if_needed(self);
  auto _batch1 = AtenIpexTypeXPU::to_plain_if_needed(batch1);
  auto _batch2 = AtenIpexTypeXPU::to_plain_if_needed(batch2);
  return at::AtenIpexTypeXPU::baddbmm(_self, _batch1, _batch2, beta, alpha);
}

at::Tensor& wrapper_XPU_out_baddbmm_out(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_baddbmm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_baddbmm_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, batch1, "wrapper_XPU_out_baddbmm_out", "batch1");
  c10::impl::check_and_update_common_device(
      common_device, batch2, "wrapper_XPU_out_baddbmm_out", "batch2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::baddbmm_out(
      self, batch1, batch2, beta, alpha, out);
}

at::Tensor& wrapper_XPU__baddbmm_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__baddbmm_", "self");
  c10::impl::check_and_update_common_device(
      common_device, batch1, "wrapper_XPU__baddbmm_", "batch1");
  c10::impl::check_and_update_common_device(
      common_device, batch2, "wrapper_XPU__baddbmm_", "batch2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::baddbmm_(self, batch1, batch2, beta, alpha);
}

at::Tensor wrapper_XPU__addbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__addbmm", "self");
  c10::impl::check_and_update_common_device(
      common_device, batch1, "wrapper_XPU__addbmm", "batch1");
  c10::impl::check_and_update_common_device(
      common_device, batch2, "wrapper_XPU__addbmm", "batch2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addbmm(self, batch1, batch2, beta, alpha);
}

at::Tensor& wrapper_XPU_out_addbmm_out(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_addbmm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_addbmm_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, batch1, "wrapper_XPU_out_addbmm_out", "batch1");
  c10::impl::check_and_update_common_device(
      common_device, batch2, "wrapper_XPU_out_addbmm_out", "batch2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addbmm_out(
      self, batch1, batch2, beta, alpha, out);
}

at::Tensor& wrapper_XPU__addbmm_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__addbmm_", "self");
  c10::impl::check_and_update_common_device(
      common_device, batch1, "wrapper_XPU__addbmm_", "batch1");
  c10::impl::check_and_update_common_device(
      common_device, batch2, "wrapper_XPU__addbmm_", "batch2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addbmm_(self, batch1, batch2, beta, alpha);
}

at::Tensor wrapper_XPU__bmm(const at::Tensor& self, const at::Tensor& mat2) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__bmm", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU__bmm", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::bmm(self, mat2);
}

at::Tensor& wrapper_XPU_out_bmm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_bmm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_bmm_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_out_bmm_out", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::bmm_out(self, mat2, out);
}

at::Tensor& wrapper_XPU_out_tensordot_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::IntArrayRef dims_self,
    at::IntArrayRef dims_other,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_tensordot_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_tensordot_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, other, "wrapper_XPU_out_tensordot_out", "other");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::tensordot_out(
      self, other, dims_self, dims_other, out);
}

at::Tensor wrapper_XPU__addmm_activation(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    bool use_gelu) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__addmm_activation", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat1, "wrapper_XPU__addmm_activation", "mat1");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU__addmm_activation", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::_addmm_activation(
      self, mat1, mat2, beta, alpha, use_gelu);
}

at::Tensor wrapper_XPU_addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_addmm", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat1, "wrapper_XPU_addmm", "mat1");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_addmm", "mat2");

  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addmm(self, mat1, mat2, beta, alpha);
}

at::Tensor& wrapper_XPU_addmv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_addmv_", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat, "wrapper_XPU_addmv_", "mat");
  c10::impl::check_and_update_common_device(
      common_device, vec, "wrapper_XPU_addmv_", "vec");

  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addmv_(self, mat, vec, beta, alpha);
}

at::Tensor wrapper_XPU_addmv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_addmv", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat, "wrapper_XPU_addmv", "mat");
  c10::impl::check_and_update_common_device(
      common_device, vec, "wrapper_XPU_addmv", "vec");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::addmv(self, mat, vec, beta, alpha);
}

at::Tensor& wrapper_XPU_addmm_(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_addmm_", "self");
  c10::impl::check_and_update_common_device(
      common_device, mat1, "wrapper_XPU_addmm_", "mat1");
  c10::impl::check_and_update_common_device(
      common_device, mat2, "wrapper_XPU_addmm_", "mat2");
  const OptionalDeviceGuard device_guard(device_of(self));
  addmm__meta(self, mat1, mat2, beta, alpha, self);
  return at::AtenIpexTypeXPU::addmm_out(self, mat1, mat2, beta, alpha, self);
}

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("_addmm_activation", TORCH_FN((&wrapper_XPU__addmm_activation)));
  m.impl(
      "_addmm_activation.out",
      TORCH_FN((&wrapper_XPU_out__addmm_activation_out)));
  m.impl("addmm", TORCH_FN((&wrapper_XPU_addmm)));
  m.impl("addmm_", TORCH_FN((&wrapper_XPU_addmm_)));
  m.impl("addmm.out", TORCH_FN((&wrapper_XPU_out_addmm_out)));
  m.impl("addmv", TORCH_FN((&wrapper_XPU_addmv)));
  m.impl("addmv_", TORCH_FN((&wrapper_XPU_addmv_)));
  m.impl("addmv.out", TORCH_FN((&wrapper_XPU_out_addmv_out)));
  m.impl("mm", TORCH_FN((&wrapper_XPU_mm)));
  m.impl("mm.out", TORCH_FN((&wrapper_XPU_out_mm_out)));
  m.impl("baddbmm", TORCH_FN((&wrapper_XPU__baddbmm)));
  m.impl("baddbmm.out", TORCH_FN((&wrapper_XPU_out_baddbmm_out)));
  m.impl("baddbmm_", TORCH_FN((&wrapper_XPU__baddbmm_)));
  m.impl("addbmm", TORCH_FN((&wrapper_XPU__addbmm)));
  m.impl("addbmm.out", TORCH_FN((&wrapper_XPU_out_addbmm_out)));
  m.impl("addbmm_", TORCH_FN((&wrapper_XPU__addbmm_)));
  m.impl("bmm", TORCH_FN((&wrapper_XPU__bmm)));
  m.impl("bmm.out", TORCH_FN((&wrapper_XPU_out_bmm_out)));
  m.impl("tensordot.out", TORCH_FN((&wrapper_XPU_out_tensordot_out)));
}

} // namespace
#endif
