#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include <vector>

#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include <c10/util/typeid.h>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static inline bool check_broadcast(
    const Tensor& src,
    const IntArrayRef& shape) {
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
static void gemm_batch(
    sycl::queue& queue,
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
    sycl::queue& queue,
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
    sycl::queue& queue,
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

static void mkl_baddbmm(
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
    // For mkl_baddbmm, have to convert it to contiguous format(only update meta
    // data, and don't copy memory) for such kind of tensor below: E.g.: the
    // tensor whose size is [10, 12, 50], and stride is [50, 500, 1], where
    // oneMKL lib cannot handle this kind of stride. Because stridec from oneMKL
    // strided style API means step size for each sample in the same batch.
    // However, for mkl_matmul, the stridec is always c.numel(), because we only
    // have 1 sample when we do addmm.
    result.resize_(
        {batch1.size(0), batch1.size(1), batch2.size(2)},
        at::MemoryFormat::Contiguous);
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
  // for the corner case: result tensor with size [b, m, 1], stride [m, 1, 1]
  // we cannot use stride to get its leading dimension, whose value should be m.
  int64_t ldc;
  if (c.strides()[1] == c.strides()[2] == 1) {
    ldc = c.sizes()[transpose_c ? 2 : 1];
  } else {
    ldc = c.strides()[transpose_c ? 1 : 2];
  }

  const int64_t stridea = a.strides()[0];
  const int64_t strideb = b.strides()[0];
  const int64_t stridec = c.strides()[0];
  int64_t num_batch = c.sizes()[0];

  // Always ensure the conjugation for c is resolved since there's no way to
  // specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
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

static void mkl_matmul(
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
  // for the corner case: result tensor with size [m, 1], stride [1, 1]
  // we cannot use stride to get its leading dimension, whose value should be m.
  int64_t ldc;
  if (1 == c.strides()[0] == c.strides()[1]) {
    ldc = c.sizes()[transpose_c ? 1 : 0];
  } else {
    ldc = c.strides()[transpose_c ? 0 : 1];
  }

  // Always ensure the conjugation for c is resolved since there's no way to
  // specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
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

// Eltwise((m1 x m2 + b) * alpha + beta * accumu)
// Eltwise((m1 x m2 + b) * alpha) + beta * accumu
static void onednn_matmul(
    Tensor& result,
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& bias,
    const Tensor& accumu, // tensor for post_sum
    bool m2_trans,
    Attr attr) {
  if (m1.is_quantized()) {
    if (m2.sizes()[1] == m1.sizes()[1])
      m2.transpose_(0, 1);
  }
  std::vector<int64_t> result_shape;
  auto dim = m1.dim();
  if (dim == 2) {
    result_shape = m2_trans ? std::vector<int64_t>{m1.size(0), m2.size(1)}
                            : std::vector<int64_t>{m1.size(0), m2.size(0)};
  } else {
    result_shape = m2_trans
        ? std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(2)}
        : std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(1)};
  }
  if (!result.defined())
    result = at::empty(result_shape, m1.options());

  if (attr.with_sum()) {
    TORCH_CHECK(
        check_broadcast(accumu, result_shape),
        "tensor for accumulate ",
        accumu.sizes(),
        " cannot broadcast to ",
        result_shape);
    c10::MaybeOwned<Tensor> bc_accumu =
        expand_size(accumu, result_shape, "gemm_broadcast");
    if (!result.is_same(*bc_accumu))
      result.resize_(result_shape).copy_(*bc_accumu);
  } else {
    result.resize_(result_shape);
  }

  // mat1, mat2, res should satisfy oneDNN supported strides
  Tensor mat1 =
      xpu::oneDNN::is_onednn_matmul_strides(m1) ? m1 : m1.contiguous();
  Tensor mat2 =
      xpu::oneDNN::is_onednn_matmul_strides(m2) ? m2 : m2.contiguous();
  Tensor res = xpu::oneDNN::is_onednn_matmul_strides(result, true)
      ? result
      : result.contiguous();
  // bias should always contiguous in oneDNN
  Tensor _bias = bias.defined() ? bias.contiguous() : bias;

  xpu::oneDNN::matmul(res, mat1, mat2, _bias, m2_trans, attr);
  if (!res.is_same(result)) {
    result.copy_(res);
  }
}

static Attr get_onednn_linear_sum_attr(
    const Tensor& input,
    const Tensor& weight,
    Tensor& accumu,
    Tensor& output,
    float scale,
    bool& is_fused) {
  is_fused = true;
  Attr attr;
  if (scale == 0.f)
    return attr;

  const auto input_sizes = input.sizes();
  const auto weight_sizes = weight.sizes();
  std::vector<int64_t> output_sizes;
  if (input.dim() == 2) {
    output_sizes = {input_sizes[0], weight_sizes[1]};
  } else if (input.dim() == 3) {
    output_sizes = {input_sizes[0], input_sizes[1], weight_sizes[1]};
  }

  Tensor out = at::empty(output_sizes, input.options());
  if (!xpu::oneDNN::binary_valid(out, accumu)) {
    is_fused = false;
    return attr;
  }

  // For post-sum and post-binary-add, onednn needs sum/binary scale=1.f
  // Thus we need the following transformation
  // conv(src, wei) + scale * accumu
  // scale * (1/scale * linear(src, wei) + sum/binary)
  if (scale != 1.f)
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f / scale,
        /* beta */ 0.f,
        attr.kind_with_linear);

  auto accumu_ctx = DPCPPTensorContext::get_tensor_ctx(accumu);
  if (accumu_ctx.is_plain())
    accumu = accumu.contiguous();

  if (accumu.sizes() == output_sizes) {
    // If sizes are the same, post sum is used.
    output = accumu;
    attr.append_post_sum(/* sum_scale */ 1.f);
  } else {
    // If sizes are different, post binary is used.
    if (input.dim() == 3)
      accumu = accumu.view({-1, accumu.sizes()[2]});
    attr.append_post_binary(attr.kind_with_binary_add, accumu);
  }

  if (scale != 1.f)
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ scale,
        /* beta */ 0.f,
        attr.kind_with_linear);

  return attr;
}

/***** The helper function to get bias or post sum for onednn_matmul *****
Decide the accumul tensor should be bias or post sum.
In onednn, it support accumul = alpha * (m1 x m2 + bias) + accumul.
We prefer to use bias when the scale of accumul tensor is equal to alpha.
Otherwise, post sum will be selected. Sometimes, post sum will introduce extra
D2D copy to between accumul and result buffers, because oneDNN only supports the
post sum buffer and the result buffer are the same one.
*/
static Attr get_onednn_matmul_attr(
    at::Tensor& result,
    const at::Tensor& accumul1,
    const at::Tensor& accumul2,
    float alpha,
    float beta1,
    float beta2,
    at::Tensor& bias,
    at::Tensor& accumul) {
  TORCH_CHECK(alpha != 0.f, "Alpha should not be equal to 0.f");
  Attr attr;
  bool need_add_accumul1 = accumul1.defined() && (beta1 != 0.f);
  bool need_add_accumul2 = accumul2.defined() && (beta2 != 0.f);

  if (!need_add_accumul1 && !need_add_accumul2) {
    // no bias and no accumul
    Attr attr;
    return attr;
  }

  if (need_add_accumul1 && !need_add_accumul2) {
    if (alpha == beta1 && !result.is_same(accumul1)) {
      // only bias
      // result = alpha * (m1 x m2 + accumul1)
      bias = accumul1;
      if (alpha != 1.f)
        attr.append_post_eltwise(
            /* scale */ 1.f,
            alpha,
            /* beta */ 0.f,
            attr.kind_with_linear);
    } else {
      // only accumul
      // result = alpha * (m1 x m2) + beta1 * accumul1
      // result = beta1 * (alpha / beta1 * (mat1 * mat2) + accumul1)
      // Since oneDNN only supports sum_scale=1.0 for non-int8 case,
      // we do this formula transformation.
      accumul = accumul1;
      alpha /= beta1;
      if (alpha != 1.f)
        attr.append_post_eltwise(1.f, alpha, 0.f, attr.kind_with_linear);
      attr.append_post_sum(/* sum_scale */ 1.f);
      if (beta1 != 1.f)
        attr.append_post_eltwise(1.f, beta1, 0.f, attr.kind_with_linear);
    }
  }

  if (!need_add_accumul1 && need_add_accumul2) {
    if (alpha == beta2 && !result.is_same(accumul2)) {
      // only bias
      // result = alpha * (m1 * m2 + accumul2)
      bias = accumul2;
      if (alpha != 1.f)
        attr.append_post_eltwise(
            /* scale */ 1.f,
            alpha,
            /* beta */ 0.f,
            attr.kind_with_linear);
    } else {
      // only accumul
      // result = alpha * (m1 x m2) + beta2 * accumul2
      // result = beta2 * (alpha / beta2 * (mat1 * mat2) + accumul2)
      accumul = accumul2;
      alpha /= beta2;
      if (alpha != 1.f)
        attr.append_post_eltwise(1.f, alpha, 0.f, attr.kind_with_linear);
      attr.append_post_sum(/* sum_scale */ 1.f);
      if (beta2 != 1.f)
        attr.append_post_eltwise(1.f, beta2, 0.f, attr.kind_with_linear);
    }
  }

  if (need_add_accumul1 && need_add_accumul2) {
    // both need_add_accumul1 and need_add_accumul2
    if (alpha == beta1 && !result.is_same(accumul1)) {
      // result = alpha * (m1 x m2 + accumul1) + beta2 * accumul2
      // result = beta2 * (alpha / beta2 * (mat1 * mat2 + accumul1) + accumul2)
      bias = accumul1;
      accumul = accumul2;
      alpha /= beta2;
      if (alpha != 1.f)
        attr.append_post_eltwise(1.f, alpha, 0.f, attr.kind_with_linear);
      attr.append_post_sum(/* sum_scale */ 1.f);
      if (beta2 != 1.f)
        attr.append_post_eltwise(1.f, beta2, 0.f, attr.kind_with_linear);
    } else if (alpha == beta2 && !result.is_same(accumul2)) {
      // result = alpha * (m1 x m2 + accumul2) + beta1 * accumul1
      // result = beta1 * (alpha / beta1 * (mat1 * mat2 + accumul2) + accumul1)
      bias = accumul2;
      accumul = accumul1;
      alpha /= beta1;
      if (alpha != 1.f)
        attr.append_post_eltwise(1.f, alpha, 0.f, attr.kind_with_linear);
      attr.append_post_sum(/* sum_scale */ 1.f);
      if (beta1 != 1.f)
        attr.append_post_eltwise(1.f, beta1, 0.f, attr.kind_with_linear);
    } else {
      // result = beta1 * accumul1 + beta2 * accumul2;
      // result = alpha * (m1 x m2) + result
      result = accumul1 * beta1;
      result = at::AtenIpexTypeXPU::add_out(result, accumul2, beta2, result);
      accumul = result;
      if (alpha != 1.f)
        attr.append_post_eltwise(1.f, alpha, 0.f, attr.kind_with_linear);
      attr.append_post_sum(/* sum_scale */ 1.f);
    }
  }

  return attr;
}

// result = alpha x (tensor1 x tensor2 + bias) + accmul_scale x accumul
static at::Tensor matmul_fusion_variants(
    at::Tensor& result,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor& bias,
    at::Tensor& accumul,
    bool trans,
    bool& fallback,
    Attr attr) {
  const OptionalDeviceGuard device_guard(device_of(accumul));
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  // TODO: matmul case is complicated
  // supported fusion cases,
  // 1. 2D x 2D
  // 2. 3D x 3D
  fallback = false;
  if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    onednn_matmul(result, tensor1, tensor2, bias, accumul, trans, attr);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    Tensor t1 = tensor1;
    std::vector<int64_t> t1_shape, r_shape;

    for (int i = 0; i < t1.sizes().size() - 1; i++) {
      t1_shape.push_back(t1.sizes()[i]);
      r_shape.push_back(t1.sizes()[i]);
    }
    t1_shape.push_back(t1.sizes()[t1.sizes().size() - 1]);
    r_shape.push_back(trans ? tensor2.sizes()[1] : tensor2.sizes()[0]);

    std::vector<int64_t> sizes = t1.sizes().vec();
    std::vector<int64_t> strides = t1.strides().vec();
    at::collapse_dims(sizes.data(), strides.data(), t1.dim(), t1.dim() - 1);
    t1.resize_({sizes.data()[0], sizes.data()[1]});

    bool can_be_fused = accumul.defined() && (accumul.sizes() == r_shape);
    if (!accumul.defined() || can_be_fused) {
      if (can_be_fused) {
        accumul.resize_({t1.size(0), r_shape[2]});
      }

      onednn_matmul(result, t1, tensor2, bias, accumul, trans, attr);

      if (r_shape.size()) {
        result.resize_(r_shape);
      }
      return result;
    }

  } else if (
      (dim_tensor1 >= 1 && dim_tensor2 >= 1) &&
      (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error
    // messages
    TORCH_CHECK(
        !(bias.defined() && accumul.defined()),
        "for 3D matmul, we only support one accumulate tensor");

    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    at::IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));

    // inverse dims in non-transpose case
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-1) : 1;
    int64_t p = tensor2.size(-2);

    at::IntArrayRef batch_tensor2(
        tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion =
        at::infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    if (!trans)
      tensor2_expand_size.insert(tensor2_expand_size.end(), {p, m2});
    else
      tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});
    int expand_batch_product = std::accumulate(
        expand_batch_portion.begin(),
        expand_batch_portion.end(),
        1,
        std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    if (!trans)
      tensor2_bmm_view.insert(tensor2_bmm_view.end(), {p, m2});
    else
      tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});
    // flatten expanded batches
    at::Tensor tensor1_expanded =
        tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    at::Tensor tensor2_expanded =
        tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    TORCH_CHECK(tensor1_expanded.dim() == 3, "expected 3D tensor");
    TORCH_CHECK(tensor2_expanded.dim() == 3, "expected 3D tensor");

    // reshape batches back into result
    std::vector<int64_t> output_shape(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    // 3D matmul does not support bias
    if (bias.defined()) {
      attr.append_post_sum(/* sum_scale */ 1.f);
      accumul = bias;
      bias = at::Tensor();
    }

    bool can_be_fused =
        accumul.defined() && (accumul.sizes().vec() == output_shape);
    if (!accumul.defined() || can_be_fused) {
      onednn_matmul(
          result,
          tensor1_expanded,
          tensor2_expanded,
          bias,
          accumul,
          trans,
          attr);
      Tensor output = at::_unsafe_view(result, output_shape);
      return output;
    }
  }

  // fallback
  fallback = true;
  return result;
}

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
