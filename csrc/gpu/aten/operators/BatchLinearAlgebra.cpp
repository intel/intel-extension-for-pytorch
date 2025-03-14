#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <runtime/Utils.h>
#include <tensor/TensorMeta.h>
#include <utils/oneMKLUtils.h>
#include <xpu/ATen/ops/_linalg_det_native.h>
#include <xpu/ATen/ops/_linalg_slogdet_native.h>

#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include "utils/CustomOperatorRegistration.h"
#endif

#include "Resize.h"

#include <torch/custom_class.h>
#include "comm/ParamUtils.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

// Used as an interface between the different BLAS-like libraries
enum class TransposeType {
  NoTranspose,
  Transpose,
  ConjTranspose,
};

#ifdef USE_ONEMKL
// Transforms TransposeType into the BLAS / LAPACK format
static oneapi::mkl::transpose to_blas(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose:
      return oneapi::mkl::transpose::trans;
    case TransposeType::NoTranspose:
      return oneapi::mkl::transpose::nontrans;
    case TransposeType::ConjTranspose:
      return oneapi::mkl::transpose::conjtrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif

namespace impl {

#ifdef USE_ONEMKL
template <typename scalar_t>
int64_t mkl_getrs_scratchpad(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<scalar_t>(
      queue,
      trans,
      n,
      nrhs,
      lda,
      stride_a,
      stride_ipiv,
      ldb,
      stride_b,
      batch_size);
}

template <>
int64_t mkl_getrs_scratchpad<c10::complex<double>>(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<double>>(
      queue,
      trans,
      n,
      nrhs,
      lda,
      stride_a,
      stride_ipiv,
      ldb,
      stride_b,
      batch_size);
}

template <>
int64_t mkl_getrs_scratchpad<c10::complex<float>>(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<float>>(
      queue,
      trans,
      n,
      nrhs,
      lda,
      stride_a,
      stride_ipiv,
      ldb,
      stride_b,
      batch_size);
}

template <typename scalar_t>
void mkl_getrs(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    scalar_t* b,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size,
    scalar_t* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      a,
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      b,
      ldb,
      stride_b,
      batch_size,
      scratchpad,
      scratchpad_size);
}

template <>
void mkl_getrs<c10::complex<double>>(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    c10::complex<double>* b,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size,
    c10::complex<double>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      ldb,
      stride_b,
      batch_size,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpad_size);
}

template <>
void mkl_getrs<c10::complex<float>>(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    c10::complex<float>* b,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size,
    c10::complex<float>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      ldb,
      stride_b,
      batch_size,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpad_size);
}

#endif

#ifdef USE_ONEMKL
void error_handle(
    std::vector<int32_t>& infos,
    oneapi::mkl::lapack::batch_error& be) {
  auto errs = be.exceptions();
  auto ids = be.ids();
  for (auto& i : ids) {
    try {
      std::rethrow_exception(errs[i]);
    } catch (oneapi::mkl::lapack::exception e) {
      std::cout << "Cathed lapack exception:"
                << "\nWhat: " << e.what() << "\nInfo: " << e.info()
                << "\nDetail: " << e.detail() << std::endl;
      infos[i] = e.info();
    } catch (sycl::exception e) {
      std::cout << "Catched SYCL exception:"
                << "\nWhat: " << e.what() << "\nInfo: -1" << std::endl;
      infos[i] = -1;
    }
  }
}
#endif

template <typename scalar_t>
static void apply_lu_solve_dpcpp_(
    const Tensor& b_,
    const Tensor& lu_,
    const Tensor& pivots_,
    std::vector<int32_t>& infos_,
    TransposeType t) {
#ifdef USE_ONEMKL
  // do nothing if empty input
  if (lu_.numel() == 0)
    return;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t batch_size = native::batchCount(b_);

  auto trans = to_blas(t);
  int64_t n = lu_.size(-2);
  int64_t nrhs = b_.size(-1);
  int64_t lda = lu_.size(-2);
  int64_t stride_a = native::matrixStride(lu_);
  int64_t stride_ipiv = pivots_.size(-1);
  int64_t ldb = b_.size(-2);
  int64_t stride_b = native::matrixStride(b_);

  scalar_t* a = (scalar_t*)(lu_.data_ptr());
  Tensor pivots = pivots_;
  if (pivots_.scalar_type() == at::ScalarType::Int)
    pivots = pivots_.to(kLong);
  int64_t* ipiv = (int64_t*)(pivots.data_ptr());
  scalar_t* b = (scalar_t*)(b_.data_ptr());

  int64_t scratchpadsize = mkl_getrs_scratchpad<scalar_t>(
      dpcpp_queue,
      trans,
      n,
      nrhs,
      lda,
      stride_a,
      stride_ipiv,
      ldb,
      stride_b,
      batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, b_.options());
  try {
    mkl_getrs<scalar_t>(
        dpcpp_queue,
        trans,
        n,
        nrhs,
        a,
        lda,
        stride_a,
        ipiv,
        stride_ipiv,
        b,
        ldb,
        stride_b,
        batch_size,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_, be);
  }
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

} // namespace impl

std::tuple<Tensor, Tensor> linalg_eig(const Tensor& input) {
  return at::native::linalg_eig(input);
}

std::tuple<Tensor&, Tensor&> linalg_eig_out(
    const Tensor& input,
    Tensor& values,
    Tensor& vectors) {
  auto input_tmp = input.cpu();
  // fall back to CPU
  // 1, mkl doesn't have GPU interface for GEEV routine. and Due to this lack of
  // uniqueness, different hardware and software may compute different
  // eigenvectors.
  // 2, we will try to dep on IPEX oneMKL package as long as if it supports CPU
  // device
  // 3, magma CPU is potential path, as well

  auto options = input.options().device(at::kCPU);
  ScalarType values_type = input.scalar_type();
  ScalarType vectors_type = input.scalar_type();
  if (!input.is_complex()) {
    // for real-valued input we can have either real- or complex-valued output
    ScalarType input_complex_dtype = toComplexType(input.scalar_type());
    values_type = values.is_complex() ? input_complex_dtype : values_type;
    vectors_type = vectors.is_complex() ? input_complex_dtype : vectors_type;
  }
  Tensor values_tmp = at::empty({0}, options.dtype(values_type));
  Tensor vectors_tmp = at::empty({0}, options.dtype(vectors_type));
  std::tie(values_tmp, vectors_tmp) =
      at::native::linalg_eig_out(input_tmp, values_tmp, vectors_tmp);
  resize_output(values, values_tmp.sizes());
  resize_output(vectors, vectors_tmp.sizes());
  values.copy_(values_tmp);
  vectors.copy_(vectors_tmp);
  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

// As P is a permutation matrix
// det(P) = 1 if it's an even permutation and det(P) = -1 if it's an odd
// permutation
Tensor lu_det_P(const Tensor& pivots) {
  return (at::arange(1, pivots.size(-1) + 1, pivots.options()) != pivots)
      .sum(-1, /*keepdim=*/false, /*dtype=*/at::kLong)
      .fmod_(2)
      // take 0 to 1 and 1 to -1
      .mul_(-2)
      .add_(1);
}

Tensor& linalg_lu_solve_out(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    bool left,
    bool adjoint,
    Tensor& result) {
  // Trivial case
  if (result.numel() == 0) {
    return result;
  }

  // Solve A^H X = B^H. Then we return X^H
  if (!left) {
    adjoint = !adjoint;
    result.transpose_(-2, -1);
  }

  // Copy B (or B^H) into result
  if (!result.is_same(B)) {
    result.copy_(left ? B : B.mH());
  }

  // Make LU / pivots F-contiguous
  auto pivots_ = pivots.expect_contiguous();
  auto LU_ = at::native::borrow_else_clone(
      LU.mT().is_contiguous(), LU, LU, /*row_major=*/false);

  const auto trans = !adjoint ? TransposeType::NoTranspose
      : LU.is_complex()       ? TransposeType::ConjTranspose
                              : TransposeType::Transpose;
  std::vector<int32_t> infos_vec(native::batchCount(LU), 0);

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LU_->scalar_type(), "lu_solve_dpcpp", [&] {
        impl::apply_lu_solve_dpcpp_<scalar_t>(
            result, *LU_, *pivots_, infos_vec, trans);
      });

  // Conj-transpose back in-place
  if (!left) {
    result.transpose_(-2, -1);
    if (result.is_complex()) {
      result._set_conj(!result.is_conj());
    }
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace at {
namespace native {

Tensor _linalg_eigvals(const Tensor& input) {
  ScalarType complex_dtype = toComplexType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));
  linalg_eigvals_out(input, values);
  return values;
}

Tensor& linalg_eigvals_out_xpu(const Tensor& input, Tensor& values) {
  native::squareCheckInputs(input, "linalg.eigvals");

  // unlike NumPy for real-valued inputs the output is always complex-valued
  native::checkLinalgCompatibleDtype(
      "torch.linalg.eigvals",
      values.scalar_type(),
      toComplexType(input.scalar_type()),
      "eigenvalues");
  native::checkSameDevice("torch.linalg.eigvals", values, input, "eigenvalues");
  // MAGMA doesn't have GPU interface for GEEV routine, it requires inputs to be
  // on CPU
  auto options = input.options().device(at::kCPU);
  auto infos = at::zeros(
      {std::max<int64_t>(1, native::batchCount(input))}, options.dtype(kInt));

  bool values_expected_type =
      (values.scalar_type() == toComplexType(input.scalar_type()));

  auto expected_values_shape =
      IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  bool values_equal_expected_shape =
      values.sizes().equals(expected_values_shape);

  // if result is not empty and not in batched column major format
  bool values_tmp_needed = (values.numel() != 0 && !values.is_contiguous());
  // or result does not have the expected shape
  values_tmp_needed |= (values.numel() != 0 && !values_equal_expected_shape);
  // or result does not have the expected dtype
  values_tmp_needed |= !values_expected_type;
  // we will allocate a temporary tensor and do the copy

  // because MAGMA's GEEV takes CPU inputs and returns CPU outputs
  // 'values' tensor that is on GPU device can't be used directly
  values_tmp_needed |= (!values.is_cpu());

  // determine the appropriate scalar_type for the temporary tensors
  ScalarType values_type = input.scalar_type();
  if (!input.is_complex()) {
    // for real-valued input we can have either real- or complex-valued output
    ScalarType input_complex_dtype = toComplexType(input.scalar_type());
    values_type = values.is_complex() ? input_complex_dtype : values_type;
  }

  Tensor vectors =
      at::empty({0}, options.dtype(toComplexType(input.scalar_type())));
  if (values_tmp_needed) {
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    std::tie(values_tmp, std::ignore) =
        linalg_eig_out(input, values_tmp, vectors);
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
  } else { // use 'values' storage directly
    std::tie(values, std::ignore) = linalg_eig_out(input, values, vectors);
  }

  // Now check LAPACK/MAGMA error codes
  at::_linalg_check_errors(infos, "torch.linalg.eigvals", input.dim() == 2);
  return values;
}

TORCH_IMPL_FUNC(_linalg_det_out_xpu)
(const Tensor& A, const Tensor& det, const Tensor& LU, const Tensor& pivots) {
  auto shape = A.sizes();
  auto ndim = shape.size();

  // det
  auto det_new = AtenIpexTypeXPU::set_contiguous(
      const_cast<Tensor&>(det), shape.slice(0, ndim - 2), A.options());
  Tensor det_use = C10_UNLIKELY(det_new.has_value()) ? det_new.value() : det;

  // LU
  auto LU_strides =
      at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  auto LU_new = AtenIpexTypeXPU::set_strided(
      const_cast<Tensor&>(LU), shape, LU_strides, A.options());
  Tensor LU_use = C10_UNLIKELY(LU_new.has_value()) ? LU_new.value() : LU;

  // pivots
  AtenIpexTypeXPU::set_contiguous_no_create(
      const_cast<Tensor&>(pivots),
      shape.slice(0, ndim - 1),
      A.options().dtype(kInt));

  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it
  // copies Use the transpose of if A is contiguous since det(A^T) = det(A) We
  // limit this to real matrices, but it could also be implemented for complex
  // matrices
  at::linalg_lu_factor_ex_out(
      const_cast<Tensor&>(LU_use),
      const_cast<Tensor&>(pivots),
      const_cast<Tensor&>(info),
      A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  // det = det_P * prod(diag(LU))
  at::mul_out(
      const_cast<Tensor&>(det_use),
      AtenIpexTypeXPU::lu_det_P(pivots),
      at::prod(LU_use.diagonal(0, -2, -1), /*dim=*/-1));
  if (det_new.has_value())
    det.copy_(det_use);
  if (LU_new.has_value())
    LU.copy_(LU_use);
}

TORCH_IMPL_FUNC(_linalg_slogdet_out_xpu)
(const Tensor& A,
 const Tensor& sign,
 const Tensor& logabsdet,
 const Tensor& LU,
 const Tensor& pivots) {
  at::native::squareCheckInputs(A, "linalg.slogdet");
  at::native::checkFloatingOrComplex(
      A, "linalg.slogdet", /*low_precision*/ false);
  auto shape = A.sizes();
  auto ndim = shape.size();

  auto shape_outputs = shape.slice(0, ndim - 2);

  // sign
  auto sign_new = AtenIpexTypeXPU::set_contiguous(
      const_cast<Tensor&>(sign), shape_outputs, A.options());
  Tensor sign_use =
      C10_UNLIKELY(sign_new.has_value()) ? sign_new.value() : sign;

  // logabsdet
  auto logabsdet_new = AtenIpexTypeXPU::set_contiguous(
      const_cast<Tensor&>(logabsdet),
      shape_outputs,
      A.options().dtype(toRealValueType(A.scalar_type())));
  Tensor logabsdet_use = C10_UNLIKELY(logabsdet_new.has_value())
      ? logabsdet_new.value()
      : logabsdet;

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(
      shape,
      /*f-contig*=*/true);
  auto LU_new = AtenIpexTypeXPU::set_strided(
      const_cast<Tensor&>(LU), shape, LU_strides, A.options());
  Tensor LU_use = C10_UNLIKELY(LU_new.has_value()) ? LU_new.value() : LU;

  // pivots
  AtenIpexTypeXPU::set_contiguous_no_create(
      const_cast<Tensor&>(pivots),
      shape.slice(0, ndim - 1),
      A.options().dtype(kInt));

  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it
  // copies Use the transpose of if A is contiguous since det(A^T) = det(A) We
  // limit this to real matrices, but it could also be implemented for complex
  // matrices
  at::linalg_lu_factor_ex_out(
      const_cast<Tensor&>(LU_use),
      const_cast<Tensor&>(pivots),
      const_cast<Tensor&>(info),
      A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  auto diag_U = LU_use.diagonal(0, -2, -1);
  // sign
  at::mul_out(
      const_cast<Tensor&>(sign_use),
      diag_U.sgn().prod(-1),
      AtenIpexTypeXPU::lu_det_P(pivots));

  // logabsdet
  at::sum_out(const_cast<Tensor&>(logabsdet_use), diag_U.abs().log_(), -1);

  if (sign_new.has_value())
    sign.copy_(sign_use);
  if (logabsdet_new.has_value())
    logabsdet.copy_(logabsdet_use);
  if (LU_new.has_value())
    LU.copy_(LU_use);
}

} // namespace native
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {
at::Tensor& wrapper_XPU_out_linalg_lu_solve_out(
    const at::Tensor& LU,
    const at::Tensor& pivots,
    const at::Tensor& B,
    bool left,
    bool adjoint,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_linalg_lu_solve_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, LU, "wrapper_XPU_out_linalg_lu_solve_out", "LU");
  c10::impl::check_and_update_common_device(
      common_device, pivots, "wrapper_XPU_out_linalg_lu_solve_out", "pivots");
  c10::impl::check_and_update_common_device(
      common_device, B, "wrapper_XPU_out_linalg_lu_solve_out", "B");
  const OptionalDeviceGuard device_guard(device_of(out));

  return at::AtenIpexTypeXPU::linalg_lu_solve_out(
      LU, pivots, B, left, adjoint, out);
}

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(
      "linalg_lu_solve.out", TORCH_FN((&wrapper_XPU_out_linalg_lu_solve_out)));
}

} // namespace
#endif
