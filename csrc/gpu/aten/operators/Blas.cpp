#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include "BlasImpl.h"
#include "utils/CustomOperatorRegistration.h"
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

  std::vector<int64_t> result_shape = {mat1.size(0), mat2.size(1)};
  result.resize_(result_shape);
  if (result.numel() == 0)
    return result;

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

  // complex/double case
  if (mat1.is_complex() || mat1.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, self, mat1, mat2, beta, alpha);
    return result;
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

  // general case
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
  xpu::oneDNN::matmul(result, mat1, mat2, bias, true, attr);
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
      self.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0],
      "x",
      self.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  result.resize_({self.size(0), mat2.size(1)});
  if (self.numel() == 0 || mat2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_matmul(result, result, self, mat2, Scalar(0), Scalar(1));
    return result;
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }

  xpu::oneDNN::matmul(result, self, mat2, at::Tensor(), true, Attr());
  return result;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::mm_out(self, mat2, result);
  return result;
}

Tensor mv(const Tensor& self, const Tensor& vec) {
  Tensor result = at::empty({self.size(0)}, self.options());
  return at::addmv_(result, self, vec, 0, 1);
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

  // complex and double case
  if (batch1.is_complex() || batch2.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    impl::mkl_baddbmm(result, input, batch1, batch2, beta, alpha);
    return result;
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
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
    Tensor binary = binary.dim() < 3 ? input.unsqueeze(0) : input;
    binary = binary.dim() < 3 ? binary.unsqueeze_(0) : binary;
    float alpha_ = alpha.to<float>() / beta_;
    if (alpha_ != 1.f)
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
    attr.append_post_binary(attr.kind_with_binary_add, binary);
    if (beta_ != 1.f)
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
  }
  xpu::oneDNN::matmul(result, batch1, batch2, at::Tensor(), true, attr);
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

  if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
#ifdef USE_ONEMKL
    return at::AtenIpexTypeXPU::baddbmm_out(
        result, self, batch2, Scalar(0), Scalar(1), result);
#else
    AT_ERROR(
        "Double and complex datatype matmul is not supported. Include oneMKL library in compilation");
#endif
  }
  xpu::oneDNN::matmul(result, self, batch2, at::Tensor(), true, Attr());
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
  xpu::oneDNN::matmul(result, m1, m2, at::Tensor(), true, attr);
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
  IPEX_OP_REGISTER("trans_matmul_div.Tensor", trans_matmul_div_tensor)
  IPEX_OP_REGISTER("trans_matmul_div_add", trans_matmul_div_add);
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
