#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_linalg_check_errors.h>

#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>

#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Resize.h"

#include <torch/custom_class.h>
#include "comm/ParamUtils.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
void set_strided(Tensor& output, IntArrayRef sizes, IntArrayRef strides) {
  output.resize_(sizes);
  output.as_strided_(sizes, strides);
}

void set_contiguous(Tensor& output, IntArrayRef sizes) {
  auto strides = c10::contiguous_strides(sizes);
  set_strided(output, sizes, strides);
}

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

static inline std::tuple<bool, bool> _parse_qr_mode(c10::string_view mode) {
  bool compute_q;
  bool reduced;
  if (mode == "reduced") {
    compute_q = true;
    reduced = true;
  } else if (mode == "complete") {
    compute_q = true;
    reduced = false;
  } else if (mode == "r") {
    compute_q = false;
    reduced = true; // this is actually irrelevant in this mode
  } else {
    TORCH_CHECK(
        false,
        "qr received unrecognized mode '",
        mode,
        "' but expected one of 'reduced' (default), 'r', or 'complete'");
  }
  return std::make_tuple(compute_q, reduced);
}

namespace impl {

#ifdef USE_ONEMKL
template <typename scalar_t>
int64_t mkl_getrf_scratchpad(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrf_batch_scratchpad_size<scalar_t>(
      queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}

template <>
int64_t mkl_getrf_scratchpad<c10::complex<double>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<double>>(
      queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}

template <>
int64_t mkl_getrf_scratchpad<c10::complex<float>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<float>>(
      queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}

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
int64_t mkl_getri_scratchpad(
    sycl::queue& queue,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<scalar_t>(
      queue, n, lda, stride_a, stride_ipiv, ldainv, stride_ainv, batch_size);
}

template <>
int64_t mkl_getri_scratchpad<c10::complex<double>>(
    sycl::queue& queue,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<double>>(
      queue, n, lda, stride_a, stride_ipiv, ldainv, stride_ainv, batch_size);
}

template <>
int64_t mkl_getri_scratchpad<c10::complex<float>>(
    sycl::queue& queue,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<float>>(
      queue, n, lda, stride_a, stride_ipiv, ldainv, stride_ainv, batch_size);
}

template <typename scalar_t>
void mkl_getrf(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    int64_t batch_size,
    scalar_t* scratchpad,
    int scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrf_batch,
      queue,
      m,
      n,
      a,
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      batch_size,
      scratchpad,
      scratchpadsize);
}

template <>
void mkl_getrf<c10::complex<double>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    int64_t batch_size,
    c10::complex<double>* scratchpad,
    int scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      batch_size,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpadsize);
}

template <>
void mkl_getrf<c10::complex<float>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    int64_t batch_size,
    c10::complex<float>* scratchpad,
    int scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      batch_size,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpadsize);
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

template <typename scalar_t>
void mkl_getri(
    sycl::queue& queue,
    int64_t n,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    scalar_t* ainv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size,
    scalar_t* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getri_batch,
      queue,
      n,
      a,
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      ainv,
      ldainv,
      stride_ainv,
      batch_size,
      scratchpad,
      scratchpad_size);
}

template <>
void mkl_getri<c10::complex<double>>(
    sycl::queue& queue,
    int64_t n,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    c10::complex<double>* ainv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size,
    c10::complex<double>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getri_batch,
      queue,
      n,
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      reinterpret_cast<std::complex<double>*>(ainv),
      ldainv,
      stride_ainv,
      batch_size,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpad_size);
}

template <>
void mkl_getri<c10::complex<float>>(
    sycl::queue& queue,
    int64_t n,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    c10::complex<float>* ainv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size,
    c10::complex<float>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getri_batch,
      queue,
      n,
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      reinterpret_cast<std::complex<float>*>(ainv),
      ldainv,
      stride_ainv,
      batch_size,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpad_size);
}

template <typename scalar_t>
int64_t mkl_geqrf_batch_scratchpad_size(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_tau,
    int64_t batch_size) {
  return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<scalar_t>(
      queue, m, n, lda, stride_a, stride_tau, batch_size);
}

template <>
int64_t mkl_geqrf_batch_scratchpad_size<c10::complex<float>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_tau,
    int64_t batch_size) {
  return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<float>>(
      queue, m, n, lda, stride_a, stride_tau, batch_size);
}

template <>
int64_t mkl_geqrf_batch_scratchpad_size<c10::complex<double>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_tau,
    int64_t batch_size) {
  return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<double>>(
      queue, m, n, lda, stride_a, stride_tau, batch_size);
}

template <typename scalar_t>
void mkl_geqrf_batch(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    scalar_t* tau,
    int64_t stride_tau,
    int64_t batch_size,
    scalar_t* scratchpad,
    int64_t scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::geqrf_batch,
      queue,
      m,
      n,
      a,
      lda,
      stride_a,
      tau,
      stride_tau,
      batch_size,
      (scalar_t*)scratchpad,
      scratchpadsize);
}

template <>
void mkl_geqrf_batch<c10::complex<float>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    c10::complex<float>* tau,
    int64_t stride_tau,
    int64_t batch_size,
    c10::complex<float>* scratchpad,
    int64_t scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::geqrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      reinterpret_cast<std::complex<float>*>(tau),
      stride_tau,
      batch_size,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpadsize);
}

template <>
void mkl_geqrf_batch<c10::complex<double>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    c10::complex<double>* tau,
    int64_t stride_tau,
    int64_t batch_size,
    c10::complex<double>* scratchpad,
    int64_t scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::geqrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      reinterpret_cast<std::complex<double>*>(tau),
      stride_tau,
      batch_size,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpadsize);
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

template <typename scalar_t, typename IndexType, bool upper>
void apply_triu_tril(Tensor& result, const Tensor& self, const int64_t k) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto N = self.numel();
  int64_t group_size = dpcppMaxWorkGroupSize(dev_id);
  auto num_groups = CeilDiv(N, group_size);
  auto total_items = num_groups * group_size;
  IndexType self_size_0 = (IndexType)self.size(-2);
  IndexType self_size_1 = (IndexType)self.size(-1);
  IndexType self_stride = (IndexType)(self.dim() > 2 ? self.stride(-3) : 1);
  IndexType self_stride_0 = (IndexType)self.stride(-2);
  IndexType self_stride_1 = (IndexType)self.stride(-1);
  IndexType result_stride =
      (IndexType)(result.dim() > 2 ? result.stride(-3) : 1);
  IndexType result_stride_0 = (IndexType)result.stride(-2);
  IndexType result_stride_1 = (IndexType)result.stride(-1);

  scalar_t* result_ptr = (scalar_t*)(result.data_ptr());
  scalar_t* self_ptr = (scalar_t*)(self.data_ptr());

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      for (size_t linearIndex = item.get_global_id(0); linearIndex < (size_t)N;
           linearIndex += item.get_global_range()[0]) {
        IndexType batch_id = linearIndex / (self_size_0 * self_size_1);
        IndexType row =
            (linearIndex % (self_size_0 * self_size_1)) / self_size_1;
        IndexType col =
            (linearIndex % (self_size_0 * self_size_1)) % self_size_1;

        IndexType src_index =
            batch_id * self_stride + row * self_stride_0 + col * self_stride_1;
        IndexType tgt_index = batch_id * result_stride + row * result_stride_0 +
            col * result_stride_1;

        bool mask = upper ? (col - row >= k) : (col - row <= k);
        result_ptr[tgt_index] = mask ? self_ptr[src_index] : scalar_t(0);
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

#define TRIU_TRIL_LAMBDA(upper)                                   \
  [&] {                                                           \
    if (xpu::dpcpp::detail::canUse32BitIndexMath(self)) {         \
      apply_triu_tril<scalar_t, int32_t, upper>(result, self, k); \
    } else {                                                      \
      apply_triu_tril<scalar_t, int64_t, upper>(result, self, k); \
    }                                                             \
  }

Tensor& tril_dpcpp_out(Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      "tril",
      TRIU_TRIL_LAMBDA(false));

  return result;
}

Tensor& tril_dpcpp_(Tensor& self, int64_t k) {
  return tril_dpcpp_out(self, self, k);
}

Tensor& triu_dpcpp_out(Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      "triu",
      TRIU_TRIL_LAMBDA(true));

  return result;
}

Tensor& triu_dpcpp_(Tensor& self, int64_t k) {
  return triu_dpcpp_out(self, self, k);
}

template <typename scalar_t>
static void apply_lu_dpcpp_(
    Tensor& self_,
    Tensor& pivots_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  // do nothing if empty input.
  if (self_.numel() == 0)
    return;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t batch_size = native::batchCount(self_);
  int64_t m = self_.size(-2);
  int64_t n = self_.size(-1);
  int64_t lda = m;
  int64_t stride_a = lda * n;
  int64_t stride_ipiv = (m < n) ? m : n;
  scalar_t* a = (scalar_t*)(self_.data_ptr());
  int64_t* ipiv = (int64_t*)(pivots_.data_ptr());
  int64_t scratchpadsize = mkl_getrf_scratchpad<scalar_t>(
      dpcpp_queue, m, n, lda, stride_a, stride_ipiv, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    mkl_getrf<scalar_t>(
        dpcpp_queue,
        m,
        n,
        a,
        lda,
        stride_a,
        ipiv,
        stride_ipiv,
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

/*
Note: A workaround to align with MKL API to store infos_lu
and infos_getri in vector. For future efficiency concern,
the MKL API needs to accept tensor data_ptr as input and store
the error infos inplace.
*/
template <typename scalar_t>
static void apply_inverse_dpcpp_(
    Tensor& self_,
    Tensor& self_inv_,
    std::vector<int32_t>& infos_lu,
    std::vector<int32_t>& infos_getri) {
#ifdef USE_ONEMKL
  auto req_size = self_.sizes().vec();
  req_size.pop_back();
  Tensor pivots_ = at::empty(req_size, self_.options().dtype(kLong));
  impl::apply_lu_dpcpp_<scalar_t>(self_, pivots_, infos_lu);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t batch_size = native::batchCount(self_);

  int64_t n = self_.size(-2);
  // The lda, stride_a and stride_ipiv are assigned 0 when the shape of self_ is
  // [0, 0]. And this function will fail. Aligning with Pytorch cpu, they are
  // assigned 1 when the shape of self_ is [0, 0].
  int64_t lda = std::max<int64_t>(1, n);
  int64_t stride_a = std::max<int64_t>(1, native::matrixStride(self_));
  int64_t stride_ipiv = std::max<int64_t>(1, pivots_.size(-1));

  scalar_t* a = (scalar_t*)(self_.data_ptr());
  int64_t* ipiv = (int64_t*)(pivots_.data_ptr());
  scalar_t* ainv = (scalar_t*)(self_inv_.data_ptr());

  int64_t scratchpadsize = mkl_getri_scratchpad<scalar_t>(
      dpcpp_queue, n, lda, stride_a, stride_ipiv, lda, stride_a, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    mkl_getri<scalar_t>(
        dpcpp_queue,
        n,
        a,
        lda,
        stride_a,
        ipiv,
        stride_ipiv,
        ainv,
        lda,
        stride_a,
        batch_size,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_getri, be);
  }
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_geqrf_dpcpp_(
    Tensor& self_,
    Tensor& tau_,
    int64_t m_,
    int64_t n_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t batch_size = native::batchCount(self_);

  int64_t m = m_;
  int64_t n = n_;
  int64_t lda = self_.size(-2);
  int64_t stride_a = native::matrixStride(self_);
  int64_t stride_tau = tau_.size(-1);

  scalar_t* a = (scalar_t*)(self_.data_ptr());
  scalar_t* tau = (scalar_t*)(tau_.data_ptr());

  int64_t scratchpadsize = mkl_geqrf_batch_scratchpad_size<scalar_t>(
      dpcpp_queue, m, n, lda, stride_a, stride_tau, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    mkl_geqrf_batch<scalar_t>(
        dpcpp_queue,
        m,
        n,
        a,
        lda,
        stride_a,
        tau,
        stride_tau,
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

template <typename scalar_t>
static void apply_orgqr_dpcpp_(
    Tensor& self_,
    const Tensor& tau_,
    int64_t m_,
    int64_t n_columns_,
    int64_t k_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t batch_size = native::batchCount(self_);

  int64_t m = m_;
  int64_t n = n_columns_;
  int64_t k = k_;
  int64_t lda = self_.size(-2);
  int64_t stride_a = native::matrixStride(self_);
  int64_t stride_tau = tau_.size(-1);

  scalar_t* a = (scalar_t*)(self_.data_ptr());
  scalar_t* tau = (scalar_t*)(tau_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::orgqr_batch_scratchpad_size<scalar_t>(
          dpcpp_queue, m, n, k, lda, stride_a, stride_tau, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::orgqr_batch,
        dpcpp_queue,
        m,
        n,
        k,
        a,
        lda,
        stride_a,
        tau,
        stride_tau,
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

template <typename scalar_t>
static void apply_ormqr_dpcpp_(
    const Tensor& a_,
    const Tensor& tau_,
    Tensor& c_,
    const int64_t m_,
    const int64_t n_,
    const int64_t k_,
    const bool left_,
    const bool transpose_,
    int64_t& info_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto left_right =
      (left_ ? oneapi::mkl::side::left : oneapi::mkl::side::right);
  auto trans =
      (transpose_ ? oneapi::mkl::transpose::trans
                  : oneapi::mkl::transpose::nontrans);
  int64_t m = m_;
  int64_t n = n_;
  int64_t k = k_;
  int64_t lda = (left_ ? m : n);
  int64_t ldc = m;
  scalar_t* a = (scalar_t*)(a_.data_ptr());
  scalar_t* tau = (scalar_t*)(tau_.data_ptr());
  scalar_t* c = (scalar_t*)(c_.data_ptr());

  int64_t scratchpadsize = oneapi::mkl::lapack::ormqr_scratchpad_size<scalar_t>(
      dpcpp_queue, left_right, trans, m, n, k, lda, ldc);
  Tensor scratchpad_at = at::empty({scratchpadsize}, c_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::ormqr,
        dpcpp_queue,
        left_right,
        trans,
        m,
        n,
        k,
        a,
        lda,
        tau,
        c,
        ldc,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::exception e) {
    std::cout << "Cathed lapack exception:"
              << "\nWhat: " << e.what() << "\nInfo: " << e.info()
              << "\nDetail: " << e.detail() << std::endl;
    info_ = e.info();
  }
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

// Copy from PyTorch fmk. The utils is not compatible with kXPU backend in 1.10
static inline std::tuple<Tensor, Tensor, Tensor> _create_U_S_VT(
    const Tensor& input,
    bool some,
    bool compute_uv) {
  auto sizes = input.sizes().vec();
  int64_t m = input.size(-2), n = input.size(-1);

  sizes[input.dim() - 1] = (compute_uv && some) ? std::min(m, n) : m;
  auto strides = at::detail::defaultStrides(sizes);
  // U should be a column-major or a batch of column-major matrices
  // ... x m x ucol will have strides: ...., ucol, 1
  // We require: ...., 1, m
  strides[input.dim() - 1] = m;
  strides[input.dim() - 2] = 1;

  Tensor U_empty;
  U_empty = at::empty_strided(sizes, strides, input.options());

  sizes[input.dim() - 2] = n;
  sizes[input.dim() - 1] = n;
  // VT should be a row-major or a batch of row-major matrices
  Tensor VT_empty;
  VT_empty = at::empty(sizes, input.options());

  sizes.pop_back();
  sizes[input.dim() - 2] = std::min(m, n);
  Tensor S_empty;
  ScalarType dtype = toRealValueType(typeMetaToScalarType(input.dtype()));
  S_empty = at::empty(sizes, input.options().dtype(dtype));
  return std::tuple<Tensor, Tensor, Tensor>(U_empty, S_empty, VT_empty);
}

template <typename scalar_t, typename value_t>
static void apply_svd(
    sycl::queue& dpcpp_queue,
    scalar_t* self_data,
    int64_t lda,
    int64_t self_stride,
    int64_t batchsize,
    int64_t m,
    int64_t n,
    TensorOptions self_opt,
    scalar_t* U_data,
    int64_t ldu,
    int64_t U_stride,
    value_t* S_data,
    int64_t S_stride,
    scalar_t* VT_data,
    int64_t ldvt,
    int64_t VT_stride,
    char jobz) {
#ifdef USE_ONEMKL
  oneapi::mkl::jobsvd jobu, jobvt;
  if (jobz == 'N') {
    jobu = oneapi::mkl::jobsvd::N;
    jobvt = oneapi::mkl::jobsvd::N;
  } else if (jobz == 'S') {
    jobu = oneapi::mkl::jobsvd::S;
    jobvt = oneapi::mkl::jobsvd::S;
  } else {
    jobu = oneapi::mkl::jobsvd::A;
    jobvt = oneapi::mkl::jobsvd::A;
  }

  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::gesvd_scratchpad_size<scalar_t>(
          dpcpp_queue, jobu, jobvt, m, n, lda, ldu, ldvt);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_opt);

  for (int64_t i = 0; i < batchsize; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_stride];
    scalar_t* U_working_ptr = &U_data[i * U_stride];
    value_t* S_working_ptr = &S_data[i * S_stride];
    scalar_t* VT_working_ptr = &VT_data[i * VT_stride];

    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::gesvd,
        dpcpp_queue,
        jobu,
        jobvt,
        m,
        n,
        self_working_ptr,
        lda,
        S_working_ptr,
        U_working_ptr,
        ldu,
        VT_working_ptr,
        ldvt,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
#else
  AT_ERROR("svd: oneMKL library not found in compilation");
#endif
}

template <>
void apply_svd<c10::complex<double>, double>(
    sycl::queue& dpcpp_queue,
    c10::complex<double>* self_data,
    int64_t lda,
    int64_t self_stride,
    int64_t batchsize,
    int64_t m,
    int64_t n,
    TensorOptions self_opt,
    c10::complex<double>* U_data,
    int64_t ldu,
    int64_t U_stride,
    double* S_data,
    int64_t S_stride,
    c10::complex<double>* VT_data,
    int64_t ldvt,
    int64_t VT_stride,
    char jobz) {
#ifdef USE_ONEMKL
  oneapi::mkl::jobsvd jobu, jobvt;
  if (jobz == 'N') {
    jobu = oneapi::mkl::jobsvd::N;
    jobvt = oneapi::mkl::jobsvd::N;
  } else if (jobz == 'S') {
    jobu = oneapi::mkl::jobsvd::S;
    jobvt = oneapi::mkl::jobsvd::S;
  } else {
    jobu = oneapi::mkl::jobsvd::A;
    jobvt = oneapi::mkl::jobsvd::A;
  }

  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(
          dpcpp_queue, jobu, jobvt, m, n, lda, ldu, ldvt);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_opt);

  for (int64_t i = 0; i < batchsize; i++) {
    c10::complex<double>* self_working_ptr = &self_data[i * self_stride];
    c10::complex<double>* U_working_ptr = &U_data[i * U_stride];
    double* S_working_ptr = &S_data[i * S_stride];
    c10::complex<double>* VT_working_ptr = &VT_data[i * VT_stride];

    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::gesvd,
        dpcpp_queue,
        jobu,
        jobvt,
        m,
        n,
        reinterpret_cast<std::complex<double>*>(self_working_ptr),
        lda,
        S_working_ptr,
        reinterpret_cast<std::complex<double>*>(U_working_ptr),
        ldu,
        reinterpret_cast<std::complex<double>*>(VT_working_ptr),
        ldvt,
        reinterpret_cast<std::complex<double>*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
#else
  AT_ERROR("svd: oneMKL library not found in compilation");
#endif
}

template <>
void apply_svd<c10::complex<float>, float>(
    sycl::queue& dpcpp_queue,
    c10::complex<float>* self_data,
    int64_t lda,
    int64_t self_stride,
    int64_t batchsize,
    int64_t m,
    int64_t n,
    TensorOptions self_opt,
    c10::complex<float>* U_data,
    int64_t ldu,
    int64_t U_stride,
    float* S_data,
    int64_t S_stride,
    c10::complex<float>* VT_data,
    int64_t ldvt,
    int64_t VT_stride,
    char jobz) {
#ifdef USE_ONEMKL
  oneapi::mkl::jobsvd jobu, jobvt;
  if (jobz == 'N') {
    jobu = oneapi::mkl::jobsvd::N;
    jobvt = oneapi::mkl::jobsvd::N;
  } else if (jobz == 'S') {
    jobu = oneapi::mkl::jobsvd::S;
    jobvt = oneapi::mkl::jobsvd::S;
  } else {
    jobu = oneapi::mkl::jobsvd::A;
    jobvt = oneapi::mkl::jobsvd::A;
  }

  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(
          dpcpp_queue, jobu, jobvt, m, n, lda, ldu, ldvt);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_opt);

  for (int64_t i = 0; i < batchsize; i++) {
    c10::complex<float>* self_working_ptr = &self_data[i * self_stride];
    c10::complex<float>* U_working_ptr = &U_data[i * U_stride];
    float* S_working_ptr = &S_data[i * S_stride];
    c10::complex<float>* VT_working_ptr = &VT_data[i * VT_stride];

    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::gesvd,
        dpcpp_queue,
        jobu,
        jobvt,
        m,
        n,
        reinterpret_cast<std::complex<float>*>(self_working_ptr),
        lda,
        S_working_ptr,
        reinterpret_cast<std::complex<float>*>(U_working_ptr),
        ldu,
        reinterpret_cast<std::complex<float>*>(VT_working_ptr),
        ldvt,
        reinterpret_cast<std::complex<float>*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
#else
  AT_ERROR("svd: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_symeig(
    Tensor& self,
    Tensor& eigvals,
    bool eigenvectors,
    bool upper,
    std::vector<int32_t>& infos) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto n = self.size(-1);
  auto batch_size = native::batchCount(self);

  auto a_stride = native::matrixStride(self);
  auto w_stride = eigvals.size(-1);

  auto jobz = eigenvectors ? oneapi::mkl::job::vec : oneapi::mkl::job::novec;
  auto uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::syevd_scratchpad_size<scalar_t>(
          dpcpp_queue, jobz, uplo, n, n);

  for (const auto i : c10::irange(batch_size)) {
    Tensor scratchpad_at = at::empty({scratchpadsize}, self.options());
    scalar_t* a = &(self.data_ptr<scalar_t>()[i * a_stride]);
    scalar_t* w = &(eigvals.data_ptr<scalar_t>()[i * w_stride]);
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::syevd,
        dpcpp_queue,
        jobz,
        uplo,
        n,
        a,
        n,
        w,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
#else
  AT_ERROR("symeig: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_triangular_solve(
    Tensor& A,
    Tensor& B,
    bool left,
    bool upper,
    bool transpose,
    bool unitriangular) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo = upper ? oneapi::mkl::uplo::U : oneapi::mkl::uplo::L;
  oneapi::mkl::transpose trans =
      transpose ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
  oneapi::mkl::diag diag =
      unitriangular ? oneapi::mkl::diag::U : oneapi::mkl::diag::N;
  oneapi::mkl::side side =
      left ? oneapi::mkl::side::left : oneapi::mkl::side::right;

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = native::matrixStride(A);
  auto B_mat_stride = native::matrixStride(B);
  auto batch_size = native::batchCount(A);

  auto m = left ? A.size(-1) : B.size(-2);
  auto n = B.size(-1);
  std::int64_t lda = A.size(-2);
  std::int64_t ldb = B.size(-2);
  scalar_t alpha = 1.;
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::column_major::trsm,
        dpcpp_queue,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        A_working_ptr,
        lda,
        B_working_ptr,
        ldb);
  }
#else
  AT_ERROR("triangular_solve: oneMKL library not found in compilation");
#endif
}

template <>
void apply_triangular_solve<c10::complex<float>>(
    Tensor& A,
    Tensor& B,
    bool left,
    bool upper,
    bool transpose,
    bool unitriangular) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo = upper ? oneapi::mkl::uplo::U : oneapi::mkl::uplo::L;
  oneapi::mkl::transpose trans =
      transpose ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
  oneapi::mkl::diag diag =
      unitriangular ? oneapi::mkl::diag::U : oneapi::mkl::diag::N;
  oneapi::mkl::side side =
      left ? oneapi::mkl::side::left : oneapi::mkl::side::right;

  auto A_data = A.data_ptr<c10::complex<float>>();
  auto B_data = B.data_ptr<c10::complex<float>>();
  auto A_mat_stride = native::matrixStride(A);
  auto B_mat_stride = native::matrixStride(B);
  auto batch_size = native::batchCount(A);

  auto m = left ? A.size(-1) : B.size(-2);
  auto n = B.size(-1);
  std::int64_t lda = A.size(-2);
  std::int64_t ldb = B.size(-2);
  std::complex<float> alpha = 1.f;
  for (const auto i : c10::irange(batch_size)) {
    c10::complex<float>* A_working_ptr = &A_data[i * A_mat_stride];
    c10::complex<float>* B_working_ptr = &B_data[i * B_mat_stride];
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::column_major::trsm,
        dpcpp_queue,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        reinterpret_cast<std::complex<float>*>(A_working_ptr),
        lda,
        reinterpret_cast<std::complex<float>*>(B_working_ptr),
        ldb);
  }
#else
  AT_ERROR("triangular_solve: oneMKL library not found in compilation");
#endif
}

template <>
void apply_triangular_solve<c10::complex<double>>(
    Tensor& A,
    Tensor& B,
    bool left,
    bool upper,
    bool transpose,
    bool unitriangular) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo = upper ? oneapi::mkl::uplo::U : oneapi::mkl::uplo::L;
  oneapi::mkl::transpose trans =
      transpose ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
  oneapi::mkl::diag diag =
      unitriangular ? oneapi::mkl::diag::U : oneapi::mkl::diag::N;
  oneapi::mkl::side side =
      left ? oneapi::mkl::side::left : oneapi::mkl::side::right;

  auto A_data = A.data_ptr<c10::complex<double>>();
  auto B_data = B.data_ptr<c10::complex<double>>();
  auto A_mat_stride = native::matrixStride(A);
  auto B_mat_stride = native::matrixStride(B);
  auto batch_size = native::batchCount(A);

  auto m = left ? A.size(-1) : B.size(-2);
  auto n = B.size(-1);
  std::int64_t lda = A.size(-2);
  std::int64_t ldb = B.size(-2);
  std::complex<double> alpha = 1.;
  for (const auto i : c10::irange(batch_size)) {
    c10::complex<double>* A_working_ptr = &A_data[i * A_mat_stride];
    c10::complex<double>* B_working_ptr = &B_data[i * B_mat_stride];
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::column_major::trsm,
        dpcpp_queue,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        reinterpret_cast<std::complex<double>*>(A_working_ptr),
        lda,
        reinterpret_cast<std::complex<double>*>(B_working_ptr),
        ldb);
  }
#else
  AT_ERROR("triangular_solve: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_cholesky_solve_dpcpp_(
    const Tensor& b_,
    const Tensor& A_,
    bool upper_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo = upper_ ? oneapi::mkl::uplo::U : oneapi::mkl::uplo::L;
  int64_t batch_size = native::batchCount(b_);

  int64_t n = A_.size(-2);
  int64_t nrhs = b_.size(-1);
  int64_t lda = A_.size(-2);
  int64_t ldb = b_.size(-2);
  int64_t stride_a = native::matrixStride(A_);
  int64_t stride_b = native::matrixStride(b_);

  scalar_t* a = (scalar_t*)(A_.data_ptr());
  scalar_t* b = (scalar_t*)(b_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrs_batch_scratchpad_size<scalar_t>(
          dpcpp_queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, b_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrs_batch,
        dpcpp_queue,
        uplo,
        n,
        nrhs,
        a,
        lda,
        stride_a,
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
  AT_ERROR("cholesky_solve: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_cholesky_dpcpp(
    Tensor& self_,
    bool upper_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo =
      upper_ ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

  auto n = self_.size(-1);

  int64_t lda = self_.size(-2);

  int64_t stride = native::matrixStride(self_);
  int64_t batch_size = native::batchCount(self_);

  scalar_t* a = (scalar_t*)(self_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrf_batch_scratchpad_size<scalar_t>(
          dpcpp_queue, uplo, n, lda, stride, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrf_batch,
        dpcpp_queue,
        uplo,
        n,
        a,
        lda,
        stride,
        batch_size,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_, be);
  }
#else
  AT_ERROR("cholesky: LAPACK library not found in compilation");
#endif
}

template <>
void apply_cholesky_dpcpp<c10::complex<float>>(
    Tensor& self_,
    bool upper_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo =
      upper_ ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

  auto n = self_.size(-1);

  int64_t lda = self_.size(-2);

  int64_t stride = native::matrixStride(self_);
  int64_t batch_size = native::batchCount(self_);

  std::complex<float>* a = (std::complex<float>*)(self_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<float>>(
          dpcpp_queue, uplo, n, lda, stride, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrf_batch,
        dpcpp_queue,
        uplo,
        n,
        a,
        lda,
        stride,
        batch_size,
        reinterpret_cast<std::complex<float>*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_, be);
  }
#else
  AT_ERROR("cholesky: LAPACK library not found in compilation");
#endif
}

template <>
void apply_cholesky_dpcpp<c10::complex<double>>(
    Tensor& self_,
    bool upper_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo =
      upper_ ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

  auto n = self_.size(-1);

  int64_t lda = self_.size(-2);

  int64_t stride = native::matrixStride(self_);
  int64_t batch_size = native::batchCount(self_);

  std::complex<double>* a = (std::complex<double>*)(self_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<double>>(
          dpcpp_queue, uplo, n, lda, stride, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrf_batch,
        dpcpp_queue,
        uplo,
        n,
        a,
        lda,
        stride,
        batch_size,
        reinterpret_cast<std::complex<double>*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_, be);
  }
#else
  AT_ERROR("cholesky: LAPACK library not found in compilation");
#endif
}

void apply_linalg_qr_out_dpcpp(
    const Tensor& input,
    const Tensor& Q,
    const Tensor& R,
    bool compute_q,
    bool reduced_mode) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);

  TORCH_INTERNAL_ASSERT(input.scalar_type() == Q.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == Q.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == R.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == R.device());

  TORCH_CHECK(
      input.scalar_type() != at::ScalarType::ComplexDouble &&
          input.scalar_type() != at::ScalarType::ComplexFloat,
      "MKL GPU does not support linalg_qr with complex inputs currently.")

  auto m = input.size(-2);
  auto n = input.size(-1);
  auto mn = std::min(m, n);

  // Q must have the expected shape: reduced_mode ? (..., m, min(m, n)) : (...,
  // m, m)
  if (compute_q) {
    auto expected_Q_shape = input.sizes().vec();
    expected_Q_shape.back() = reduced_mode ? mn : m;
    TORCH_INTERNAL_ASSERT(Q.sizes().equals(expected_Q_shape));

    // Q tensor must be in batched column major order (Fortran contiguous)
    TORCH_INTERNAL_ASSERT(Q.transpose(-2, -1).is_contiguous());
  }

  // R must have the expected shape: (reduced_mode || !compute_q) ? (...,
  // min(m,n), n) : (..., m, n)
  auto expected_R_shape = input.sizes().vec();
  expected_R_shape.end()[-2] = (reduced_mode || !compute_q) ? mn : m;
  TORCH_INTERNAL_ASSERT(R.sizes().equals(expected_R_shape));

  // R tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(R.transpose(-2, -1).is_contiguous());

  auto tau_shape = input.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = mn;
  Tensor tau = at::empty(tau_shape, input.options());

  // geqrf requires m x n workspace input that is modified in-place
  // if m > n and reduced==true we use Q tensor for storing the result of geqrf
  // operation otherwise R tensor is used
  Tensor QR;
  if (m <= n) {
    QR = R;
  } else { // m > n
    if (compute_q) {
      QR = reduced_mode ? Q : R;
    } else {
      // if m > n and compute_q==false we need to allocate an additional
      // temporary tensor
      QR = at::empty(input.transpose(-2, -1).sizes(), input.options());
      QR.transpose_(-2, -1);
    }
  }

  // apply_geqrf_dpcpp_ performs calculations in-place and 'QR' must be a copy
  // of input
  QR.copy_(input);
  std::vector<int32_t> infos(native::batchCount(input), 0);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "qr_dpcpp", [&] {
        impl::apply_geqrf_dpcpp_<scalar_t>(QR, tau, m, n, infos);
      });

  // this is for mode='r'
  if (!compute_q) {
    // if m > n we used a temporary tensor to store the result of geqrf
    if (m > n) {
      R.copy_(QR.slice(-2, 0, mn));
    }
    R.triu_();
    return;
  }

  // if Q tensor was used for geqrf copy the result for R from QR
  if (m > n && reduced_mode) {
    R.copy_(Q.slice(-2, 0, n));
  } else {
    Q.slice(-1, 0, n).copy_(R.slice(-1, 0, m));
  }
  R.triu_();

  // Next perform orgqr for Q using the result from geqrf
  if (reduced_mode) {
    IPEX_DISPATCH_FLOATING_TYPES(input.scalar_type(), "qr_dpcpp", [&] {
      impl::apply_orgqr_dpcpp_<scalar_t>(
          const_cast<Tensor&>(Q), tau, m, mn, mn, infos);
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES(input.scalar_type(), "qr_dpcpp", [&] {
      impl::apply_orgqr_dpcpp_<scalar_t>(
          const_cast<Tensor&>(Q), tau, m, m, mn, infos);
    });
  }
}

} // namespace impl

Tensor& triu_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  impl::triu_dpcpp_out(out, self, diagonal);
  return out;
}

Tensor& tril_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  impl::tril_dpcpp_out(out, self, diagonal);
  return out;
}

Tensor& tril_(Tensor& self, int64_t diagonal) {
  return at::AtenIpexTypeXPU::tril_out(self, diagonal, self);
}

Tensor& triu_(Tensor& self, int64_t diagonal) {
  return at::AtenIpexTypeXPU::triu_out(self, diagonal, self);
}

std::tuple<Tensor, Tensor, Tensor> _lu_with_info(
    const Tensor& self,
    bool pivot,
    bool check_errors) {
  TORCH_CHECK(
      self.dim() >= 2,
      "expected tensor with 2 or more dimensions, got size: ",
      self.sizes(),
      " instead");
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  req_size.back() = std::min(m, n);
  auto pivots_tensor = at::empty(req_size, self.options().dtype(kLong));
  req_size.pop_back();
  auto infos_tensor =
      at::zeros(req_size, self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);

  Tensor self_working_copy;
  if (self.numel() == 0) {
    self_working_copy = at::empty_like(self);
  } else {
    self_working_copy = native::cloneBatchedColumnMajor(self);
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        self.scalar_type(), "lu_dpcpp", [&] {
          impl::apply_lu_dpcpp_<scalar_t>(
              self_working_copy, pivots_tensor, infos);
        });
  }
  if (check_errors) {
    at::_linalg_check_errors(infos_tensor, "lu_dpcpp", self.dim() == 2);
  }
  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  return std::make_tuple(
      self_working_copy, pivots_tensor.to(kInt), infos_tensor);
}

// Solves a system of linear equations matmul(input, x) = other in-place
static Tensor& linalg_solve_out_info(
    Tensor& result,
    Tensor& infos,
    const Tensor& input,
    const Tensor& other) {
  at::native::checkSameDevice("linalg_solve", result, input);
  at::native::checkSameDevice("linalg_solve", other, input, "other");
  at::native::checkLinalgCompatibleDtype("linalg_solve", result, input);

  TORCH_CHECK(
      input.scalar_type() == other.scalar_type(),
      "input dtype ",
      input.scalar_type(),
      " does not match other dtype ",
      other.scalar_type());

  TORCH_CHECK(
      input.dim() >= 2,
      "input should have at least 2 dimensions, but has ",
      input.dim(),
      " dimensions instead");
  TORCH_CHECK(
      other.dim() >= 1,
      "other should have at least 1 dimension, but has ",
      other.dim(),
      " dimensions instead");

  // Two types of 'other' tensors are supported:
  // - 1-dimensional (1D) tensor or batch of 1D tensors (vector case)
  // - 2-dimensional (2D) tensor or batch of 2D tensors (matrix case)
  // original torch.solve supported only the matrix case, while NumPy works for
  // both cases for the batched input we need to be able to distinguish them
  bool vector_case = at::native::linalg_solve_is_vector_rhs(input, other);

  bool is_batched_column_major = false;
  if (vector_case) {
    is_batched_column_major = result.is_contiguous();
  } else if (!vector_case && result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if 'other' is a batch of 2D tensors, then 'input' can be non-batched and
  // will be broadcasted
  auto expected_shape =
      IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  if (!vector_case && other.dim() > 2) {
    expected_shape = other.sizes();
  }

  bool result_equal_expected_shape = result.sizes().equals(expected_shape);
  bool result_input_same_type = (result.scalar_type() == input.scalar_type());

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type; // or result does not have the same
                                          // dtype as input
  copy_needed |=
      (result.numel() != 0 &&
       !result_equal_expected_shape); // or result does not have the expected
                                      // shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = linalg_solve_out_info(result_tmp, infos, input, other);
    resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
    return result;
  }
  // else use result's storage directly

  // we need to unsqueeze 'other' because 2-dimensional tensors are expected in
  // the implementation
  Tensor other_ = vector_case ? other.unsqueeze(-1) : other;

  // _linalg_broadcast_batch_dims also includes linearSolveCheckInputs
  // it checks for squareness of 'input' and 'shape' compatibility of 'other'
  // and 'input'
  Tensor other_broadcasted, input_broadcasted;
  std::tie(other_broadcasted, input_broadcasted) =
      at::native::_linalg_broadcast_batch_dims(other_, input, "linalg_solve");

  auto squeezed_other_broadcasted = at::squeeze(other_broadcasted, -1);
  auto squeezed_result_shape = squeezed_other_broadcasted.sizes();

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    if (vector_case) {
      result.resize_(squeezed_result_shape);
    } else {
      at::native::resize_as_(
          result,
          other_broadcasted.transpose(-2, -1),
          MemoryFormat::Contiguous);
      result.transpose_(-2, -1);
    }
  }

  auto expected_result_shape =
      vector_case ? squeezed_result_shape : other_broadcasted.sizes();
  TORCH_INTERNAL_ASSERT(result.sizes().equals(expected_result_shape));
  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // result tensor must be in batched column major order (Fortran contiguous)
  // for 2D inputs or C contiguous for 1D input
  if (vector_case) {
    TORCH_INTERNAL_ASSERT(result.is_contiguous());
  } else {
    TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  }

  // for 1-dimensional 'other', we need to unsqueeze the result before passing
  // to "apply_solve"
  if (vector_case) {
    result = result.unsqueeze_(-1);
  }

  // lu_stub+lu_solve_stub perform calculations in-place and 'result' must be a
  // copy of 'other_broadcasted'
  result.copy_(other_broadcasted);

  auto input_working_copy =
      at::native::cloneBatchedColumnMajor(input_broadcasted);

  infos.resize_({std::max<int64_t>(1, native::batchCount(input_broadcasted))})
      .zero_();
  std::vector<int32_t> infos_vec_1(native::batchCount(input_broadcasted), 0);
  std::vector<int32_t> infos_vec_2(native::batchCount(input_broadcasted), 0);
  // compute the LU factorization of 'input_working_copy'
  auto pivots_shape =
      IntArrayRef(input_broadcasted.sizes().data(), input_broadcasted.dim() - 2)
          .vec(); // input_broadcasted.shape[:-2]
  pivots_shape.push_back(std::min(input.size(-2), input.size(-1)));
  Tensor pivots = at::empty(pivots_shape, input.options().dtype(kLong));
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input_working_copy.scalar_type(), "linalg_solve_dpcpp", [&] {
        impl::apply_lu_dpcpp_<scalar_t>(
            input_working_copy, pivots, infos_vec_1);
        // solve the linear system using the LU factorization
        impl::apply_lu_solve_dpcpp_<scalar_t>(
            result,
            input_working_copy,
            pivots,
            infos_vec_2,
            TransposeType::NoTranspose);
      });

  std::copy(
      infos_vec_1.begin(),
      infos_vec_1.end(),
      infos.template data_ptr<int32_t>());

  at::_linalg_check_errors(
      infos, "lu_solve_dpcpp", input_working_copy.dim() == 2);

  // for 1-dimensional 'other', we need to squeeze the result after
  // "apply_solve"
  if (vector_case) {
    result = result.squeeze_(-1);
  }

  return result;
}

Tensor _lu_solve_helper(
    const Tensor& self,
    const Tensor& LU_data,
    const Tensor& LU_pivots) {
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto LU_data_working_copy = native::cloneBatchedColumnMajor(LU_data);
  auto LU_pivots_working_copy =
      LU_pivots.is_contiguous() ? LU_pivots : LU_pivots.contiguous();
  // FIXME: oneMKL only support int64_t datatype of pivots
  LU_pivots_working_copy = LU_pivots.to(kLong);
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);

  if (self.numel() == 0 || LU_data.numel() == 0) {
    return at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lu_solve_dpcpp", [&] {
    impl::apply_lu_solve_dpcpp_<scalar_t>(
        self_working_copy,
        LU_data_working_copy,
        LU_pivots_working_copy,
        infos,
        TransposeType::NoTranspose);
  });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(infos_tensor, "lu_solve_dpcpp", self.dim() == 2);

  return self_working_copy;
}

Tensor lu_solve(
    const Tensor& self,
    const Tensor& LU_data,
    const Tensor& LU_pivots) {
  TORCH_CHECK(
      self.dim() >= 2,
      "b should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(
      LU_data.dim() >= 2,
      "LU_data should have at least 2 dimensions, but has ",
      LU_data.dim(),
      " dimensions instead");
  TORCH_CHECK(
      LU_pivots.size(-1) == LU_data.size(-1),
      "Number of pivots per batch should be same as the dimension of the matrix");
  TORCH_CHECK(
      LU_pivots.device() == LU_data.device(),
      "Expected LU_pivots and LU_data to be on the same device, "
      "but found LU_pivots on ",
      LU_pivots.device(),
      " and LU_data on ",
      LU_data.device(),
      " instead");

  IntArrayRef pivots_sizes(LU_pivots.sizes().data(), LU_pivots.dim() - 1);
  IntArrayRef lu_sizes(LU_data.sizes().data(), LU_data.dim() - 2);
  TORCH_CHECK(
      pivots_sizes == lu_sizes,
      "batch dimensions of LU_pivots doesn't match batch dimensions of LU_data");

  Tensor self_broadcasted, LU_data_broadcasted;
  std::tie(self_broadcasted, LU_data_broadcasted) =
      native::_linalg_broadcast_batch_dims(self, LU_data, "lu_solve_dpcpp");

  IntArrayRef new_pivots_sizes(
      LU_data_broadcasted.sizes().data(), LU_data_broadcasted.dim() - 1);
  Tensor LU_pivots_broadcasted = LU_pivots.expand(new_pivots_sizes);
  return at::AtenIpexTypeXPU::_lu_solve_helper(
      self_broadcasted, LU_data_broadcasted, LU_pivots_broadcasted);
}

Tensor& lu_solve_out(
    const Tensor& self,
    const Tensor& LU_data,
    const Tensor& LU_pivots,
    Tensor& out) {
  Tensor out_tmp = at::AtenIpexTypeXPU::lu_solve(self, LU_data, LU_pivots);
  out.resize_as_(out_tmp).copy_(out_tmp);
  return out;
}

Tensor _inverse_helper(const Tensor& self) {
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos_lu_vec(native::batchCount(self), 0);
  std::vector<int32_t> infos_getri_vec(native::batchCount(self), 0);

  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto self_inv_working_copy = native::cloneBatchedColumnMajor(self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self.scalar_type(), "inverse_dpcpp", [&] {
        impl::apply_inverse_dpcpp_<scalar_t>(
            self_working_copy,
            self_inv_working_copy,
            infos_lu_vec,
            infos_getri_vec);
      });

  std::copy(
      infos_lu_vec.begin(),
      infos_lu_vec.end(),
      infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(infos_tensor, "infos_lu_vec", self.dim() == 2);
  std::copy(
      infos_getri_vec.begin(),
      infos_getri_vec.end(),
      infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(infos_tensor, "infos_getri_vec", self.dim() == 2);

  return self_inv_working_copy;
}

Tensor inverse(const Tensor& self) {
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  native::squareCheckInputs(self, "inverse");
  return at::AtenIpexTypeXPU::_inverse_helper(self);
}

Tensor& inverse_out(const Tensor& self, Tensor& out) {
  if (self.size(-1) == 0) {
    out.resize_as_(self);
    return out;
  }
  out.copy_(at::AtenIpexTypeXPU::inverse(self));
  return out;
}

// A type dispatching helper function for 'apply_inverse_dpcpp_'.
Tensor& _linalg_inv_out_helper_(
    Tensor& result,
    Tensor& infos_lu,
    Tensor& infos_getri) {
  /*
  [Note:] Current mkl API `getrf_batch` and `getri_batch` does not accept
  info_arrays as input to store the error infos, instead, the errors are throwed
  out as exceptions. As a workaround, we store the errors in `vector<int64_t>
  infos`, and convert the vectors to tensors.

  'infos_lu' is for holding LU errors, and 'infos_getri' is for holding getri
  errors

  The `infos_lu` and `infos_getri` are following:

  = 0: successful exit
  < 0: if INFO = -i, the i-th argument had an illegal value or another error
  occured, such as memory allocation failed.
  > 0: if INFO = i, U(i,i) is exactly zero.
      The factorization has been completed, but the factor U is exactly
      singular, and division by zero will occur if it is used to solve a
      system of equation.
  */

  std::vector<int32_t> infos_lu_vec(native::batchCount(result), 0);
  std::vector<int32_t> infos_getri_vec(native::batchCount(result), 0);
  auto self_inv_working_copy = native::cloneBatchedColumnMajor(result);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "linalg_inv_out_dpcpp", [&] {
        impl::apply_inverse_dpcpp_<scalar_t>(
            self_inv_working_copy, result, infos_lu_vec, infos_getri_vec);
      });
  // Needs to handle the copy for scalar tensor separately.
  // Because the copy from 1D tensor to 0D scalar mismatch.
  auto expected_info_shape =
      IntArrayRef(result.sizes().cbegin(), result.sizes().cend() - 2);

  infos_lu.copy_(at::from_blob(
      (int32_t*)(infos_lu_vec.data()),
      expected_info_shape,
      c10::toRealValueType(infos_lu.scalar_type())));
  infos_getri.copy_(at::from_blob(
      (int32_t*)(infos_getri_vec.data()),
      expected_info_shape,
      c10::toRealValueType(infos_getri.scalar_type())));

  return result;
}

std::tuple<Tensor, Tensor> _linalg_qr_helper(
    const Tensor& input,
    c10::string_view mode) {
  bool compute_q, reduced_mode;
  std::tie(compute_q, reduced_mode) = _parse_qr_mode(mode);
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto mn = std::min(m, n);

  // Allocate Q, R tensors with correct shape and memory layout
  Tensor Q;
  if (compute_q) {
    auto Qt_shape = input.sizes().vec();
    Qt_shape.end()[-2] = reduced_mode ? mn : m;
    Qt_shape.end()[-1] = m;
    Q = at::empty(Qt_shape, input.options());
    Q.transpose_(-2, -1); // make 'Q' with Fortran contiguous memory layout
  } else {
    Q = at::empty({0}, input.options());
  }

  auto Rt_shape = input.sizes().vec();
  Rt_shape.end()[-2] = n;
  Rt_shape.end()[-1] = (reduced_mode || !compute_q) ? mn : m;
  Tensor R = at::empty(Rt_shape, input.options());
  R.transpose_(-2, -1); // make 'R' with Fortran contiguous memory layout

  if (input.numel()) {
    // Now fill Q, R tensors with the result
    impl::apply_linalg_qr_out_dpcpp(input, Q, R, compute_q, reduced_mode);
  }

  return std::make_tuple(Q, R);
}

std::tuple<Tensor, Tensor> geqrf(const Tensor& self) {
  TORCH_CHECK(
      self.dim() >= 2,
      "input should have at least 2 dimensions. but has ",
      self.dim(),
      " dimensions instead");
  int64_t m = self.size(-2), n = self.size(-1);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  req_size[self.dim() - 2] = std::min(m, n);
  if (self.numel() == 0) {
    return std::tuple<Tensor, Tensor>(
        at::empty(self.sizes().vec(), self.options()),
        at::empty(req_size, self.options()));
  }

  std::vector<int32_t> infos(native::batchCount(self), 0);
  Tensor self_working_copy = native::cloneBatchedColumnMajor(self);
  Tensor tau_working_copy = at::empty(req_size, self.options());

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "geqrf_dpcpp", [&] {
    impl::apply_geqrf_dpcpp_<scalar_t>(
        self_working_copy, tau_working_copy, m, n, infos);
  });
  return std::tuple<Tensor, Tensor>(self_working_copy, tau_working_copy);
}

std::tuple<Tensor&, Tensor&> geqrf_out(
    const Tensor& self,
    Tensor& a,
    Tensor& tau) {
  TORCH_CHECK(
      self.dim() >= 2,
      "input should have at least 2 dimensions. but has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(self.numel() != 0, "input must not be empty");

  Tensor a_tmp, tau_tmp;
  std::tie(a_tmp, tau_tmp) = at::AtenIpexTypeXPU::geqrf(self);
  a.resize_as_(a_tmp).copy_(a_tmp);
  tau.resize_as_(tau_tmp).copy_(tau_tmp);
  return std::tuple<Tensor&, Tensor&>(a, tau);
}

std::tuple<Tensor&, Tensor&> linalg_qr_out(
    const Tensor& A,
    c10::string_view mode,
    Tensor& Q,
    Tensor& R) {
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::min(m, n);
  bool compute_q, reduced_mode;
  std::tie(compute_q, reduced_mode) = at::native::_parse_qr_mode(mode);
  if (A.numel()) {
    // Now fill Q, R tensors with the result
    impl::apply_linalg_qr_out_dpcpp(A, Q, R, compute_q, reduced_mode);
  }
  return std::forward_as_tuple(Q, R);
}

Tensor linalg_householder_product(const Tensor& self, const Tensor& input2) {
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  std::vector<int32_t> infos(native::batchCount(self), 0);
  int64_t m = self.size(-2), n_columns_q = self.size(-1), n = input2.size(-1);
  auto q_working_copy = native::cloneBatchedColumnMajor(self);

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "orgqr_dpcpp", [&] {
    impl::apply_orgqr_dpcpp_<scalar_t>(
        q_working_copy, input2, m, n_columns_q, std::min(m, n), infos);
  });
  return q_working_copy;
}

Tensor& linalg_householder_product_out(
    const Tensor& self,
    const Tensor& input2,
    Tensor& out) {
  if (self.size(-1) == 0) {
    out.resize_as_(self);
    return out;
  }
  out.copy_(at::AtenIpexTypeXPU::linalg_householder_product(self, input2));
  return out;
}

Tensor ormqr(
    const Tensor& self,
    const Tensor& input2,
    const Tensor& input3,
    bool left,
    bool transpose) {
  TORCH_CHECK(
      self.dim() >= 2, "torch.ormqr: input must have at least 2 dimensions.");
  TORCH_CHECK(
      input3.dim() >= 2, "torch.ormqr: other must have at least 2 dimensions.");

  int64_t left_size_condition = left ? -2 : -1;
  TORCH_CHECK(
      input3.size(left_size_condition) >= input2.size(-1),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be greater than or equal to tau.shape[-1]");
  TORCH_CHECK(
      input3.size(left_size_condition) == self.size(-2),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be equal to input.shape[-2]");
  TORCH_CHECK(
      self.dim() - input2.dim() == 1,
      "torch.ormqr: ",
      "Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      input2.dim(),
      " and input.ndim is equal to ",
      self.dim());
  TORCH_CHECK(
      self.dim() == input3.dim(),
      "torch.ormqr: ",
      "Expected other to have the same number of dimensions as input, but got other.ndim equal to ",
      input3.dim(),
      " and input.ndim is equal to ",
      self.dim());

  if (self.dim() > 2) {
    auto expected_batch_shape =
        IntArrayRef(self.sizes().data(), self.dim() - 2); // self.shape[:-2]
    auto actual_batch_tau_shape = IntArrayRef(
        input2.sizes().data(), input2.dim() - 1); // input2.shape[:-1]
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);

    auto actual_batch_other_shape = IntArrayRef(
        input3.sizes().data(), input3.dim() - 2); // input3.shape[:-2]
    TORCH_CHECK(
        actual_batch_other_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of other to be equal to input.shape[:-2], but got ",
        actual_batch_other_shape);
  }

  TORCH_CHECK(
      input2.scalar_type() == self.scalar_type(),
      "torch.ormqr: Expected input and tau to have the same dtype, but input has dtype",
      self.scalar_type(),
      " and tau has dtype ",
      input2.scalar_type());
  TORCH_CHECK(
      input3.scalar_type() == self.scalar_type(),
      "torch.ormqr: Expected input and other to have the same dtype, but input has dtype",
      self.scalar_type(),
      " and other has dtype ",
      input3.scalar_type());

  native::checkSameDevice("torch.ormqr", input2, self, "tau");
  native::checkSameDevice("torch.ormqr", input3, self, "other");

  int64_t infos = 0;
  int64_t m = input3.size(0), n = input3.size(1), k = input2.size(-1);
  auto c_working_copy = native::cloneBatchedColumnMajor(input3);

  native::checkSameDevice("torch.ormqr", c_working_copy, self);

  TORCH_CHECK(
      c_working_copy.scalar_type() == self.scalar_type(),
      "torch.ormqr: Expected input and result to have the same dtype, but input has dtype",
      self.scalar_type(),
      " and result has dtype ",
      c_working_copy.scalar_type());

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "ormqr_dpcpp", [&] {
    impl::apply_ormqr_dpcpp_<scalar_t>(
        self,
        input2,
        c_working_copy,
        m,
        n,
        std::min(m, k),
        left,
        transpose,
        infos);
  });

  return c_working_copy;
}

Tensor& ormqr_out(
    const Tensor& self,
    const Tensor& input2,
    const Tensor& input3,
    bool left,
    bool transpose,
    Tensor& out) {
  TORCH_CHECK(
      self.dim() >= 2, "torch.ormqr: input must have at least 2 dimensions.");
  TORCH_CHECK(
      input3.dim() >= 2, "torch.ormqr: other must have at least 2 dimensions.");

  int64_t left_size_condition = left ? -2 : -1;
  TORCH_CHECK(
      input3.size(left_size_condition) >= input2.size(-1),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be greater than or equal to tau.shape[-1]");

  TORCH_CHECK(
      input3.size(left_size_condition) == self.size(-2),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be equal to input.shape[-2]");

  TORCH_CHECK(
      self.dim() - input2.dim() == 1,
      "torch.ormqr: ",
      "Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      input2.dim(),
      " and input.ndim is equal to ",
      self.dim());
  TORCH_CHECK(
      self.dim() == input3.dim(),
      "torch.ormqr: ",
      "Expected other to have the same number of dimensions as input, but got other.ndim equal to ",
      input3.dim(),
      " and input.ndim is equal to ",
      self.dim());

  if (self.dim() > 2) {
    auto expected_batch_shape =
        IntArrayRef(self.sizes().data(), self.dim() - 2); // self.shape[:-2]
    auto actual_batch_tau_shape = IntArrayRef(
        input2.sizes().data(), input2.dim() - 1); // input2.shape[:-1]
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);

    auto actual_batch_other_shape = IntArrayRef(
        input3.sizes().data(), input3.dim() - 2); // input3.shape[:-2]
    TORCH_CHECK(
        actual_batch_other_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of other to be equal to input.shape[:-2], but got ",
        actual_batch_other_shape);
  }

  TORCH_CHECK(
      input2.scalar_type() == self.scalar_type(),
      "torch.ormqr: Expected input and tau to have the same dtype, but input has dtype",
      self.scalar_type(),
      " and tau has dtype ",
      input2.scalar_type());
  TORCH_CHECK(
      input3.scalar_type() == self.scalar_type(),
      "torch.ormqr: Expected input and other to have the same dtype, but input has dtype",
      self.scalar_type(),
      " and other has dtype ",
      input3.scalar_type());
  TORCH_CHECK(
      out.scalar_type() == self.scalar_type(),
      "torch.ormqr: Expected input and result to have the same dtype, but input has dtype",
      self.scalar_type(),
      " and result has dtype ",
      out.scalar_type());

  native::checkSameDevice("torch.ormqr", input2, self, "tau");
  native::checkSameDevice("torch.ormqr", input3, self, "other");
  native::checkSameDevice("torch.ormqr", out, self);

  if (self.size(-1) == 0) {
    out.resize_as_(input3);
    return out;
  }
  out.resize_as_(input3).copy_(
      at::AtenIpexTypeXPU::ormqr(self, input2, input3, left, transpose));
  return out;
}

std::tuple<Tensor, Tensor, Tensor> _svd_helper(
    const Tensor& self,
    bool some,
    bool compute_uv) {
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);
  int64_t m = self.size(-2), n = self.size(-1);
  int64_t k = std::min(m, n);

  char jobz = compute_uv ? (some ? 'S' : 'A') : 'N';

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) =
      impl::_create_U_S_VT(self, some, compute_uv);

  if (self.numel() > 0) {
    auto self_working_copy = native::cloneBatchedColumnMajor(self);
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto self_stride = at::native::matrixStride(self_working_copy);
    auto U_stride = at::native::matrixStride(U_working_copy);
    auto S_stride = S_working_copy.size(-1);
    auto VT_stride = at::native::matrixStride(VT_working_copy);
    auto batchsize = at::native::batchCount(self_working_copy);

    auto m = self_working_copy.size(-2);
    auto n = self_working_copy.size(-1);
    std::int64_t lda = m;
    std::int64_t ldu = m;
    std::int64_t ldvt = n;
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        self.scalar_type(), "svd_xpu", [&] {
          using value_t = typename c10::scalar_value_type<scalar_t>::type;
          impl::apply_svd<scalar_t, value_t>(
              dpcpp_queue,
              self_working_copy.data_ptr<scalar_t>(),
              lda,
              self_stride,
              batchsize,
              m,
              n,
              self.options(),
              U_working_copy.data_ptr<scalar_t>(),
              ldu,
              U_stride,
              S_working_copy.data_ptr<value_t>(),
              S_stride,
              VT_working_copy.data_ptr<scalar_t>(),
              ldvt,
              VT_stride,
              jobz);
        });

    std::copy(
        infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
    at::_linalg_check_errors(infos_tensor, "svd_xpu", self.dim() == 2);

    if (!compute_uv) {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  } else {
    U_working_copy.zero_();
    VT_working_copy.zero_();
  }
  if (compute_uv) {
    if (some) {
      VT_working_copy = VT_working_copy.narrow(-1, 0, k);
    }
  }
  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

static void svd_resize_and_copy(
    const char* name,
    const Tensor& src,
    Tensor& dst) {
  TORCH_CHECK(
      src.device() == dst.device(),
      "svd output tensor ",
      name,
      " is on the wrong device: expected ",
      src.device(),
      " got ",
      dst.device());
  at::native::resize_output(dst, src.sizes());
  dst.copy_(src);
}

// We follow PyTorch1.10 temporarily for much in PyTorch1.13, will switch to
// PyTorch1.13 API later
std::tuple<Tensor&, Tensor&, Tensor&> _linalg_svd_out(
    const Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver,
    Tensor& U,
    Tensor& S,
    Tensor& Vh) {
  Tensor U_tmp, S_tmp, Vh_tmp;
  bool some = !full_matrices;
  std::tie(U_tmp, S_tmp, Vh_tmp) = _svd_helper(A, some, /*compute_uv=*/true);
  Tensor Vh_c = Vh_tmp.conj().transpose(-2, -1);
  svd_resize_and_copy("U", U_tmp, U);
  svd_resize_and_copy("S", S_tmp, S);
  svd_resize_and_copy("V", Vh_c, Vh);
  return std::tuple<Tensor&, Tensor&, Tensor&>(U, S, Vh);
}

std::tuple<Tensor, Tensor> _symeig_helper(
    const Tensor& self,
    bool eigenvectors,
    bool upper) {
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);

  auto self_sizes = self.sizes().vec();
  self_sizes.pop_back();
  auto eigvals = at::empty(self_sizes, self.options());

  if (self.numel() == 0) {
    return std::tuple<Tensor, Tensor>(
        eigvals, at::empty_like(self, MemoryFormat::Contiguous));
  }

  auto self_working_copy = native::cloneBatchedColumnMajor(self);

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "symeig", [&] {
    impl::apply_symeig<scalar_t>(
        self_working_copy, eigvals, eigenvectors, upper, infos);
  });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(infos_tensor, "symeig", self.dim() == 2);

  if (eigenvectors) {
    return std::tuple<Tensor, Tensor>(eigvals, self_working_copy);
  } else {
    return std::tuple<Tensor, Tensor>(eigvals, at::empty({0}, self.options()));
  }
}

std::tuple<Tensor, Tensor> _triangular_solve_helper(
    Tensor& result,
    Tensor& clone_input,
    Tensor& infos,
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool transpose,
    bool unitriangular) {
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-2) == input.size(-1));

  TORCH_INTERNAL_ASSERT(input.device() == other.device());
  TORCH_INTERNAL_ASSERT(input.device() == result.device());
  TORCH_INTERNAL_ASSERT(input.device() == clone_input.device());
  TORCH_INTERNAL_ASSERT(input.device() == infos.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT(input.scalar_type() == result.scalar_type());
  TORCH_INTERNAL_ASSERT(input.scalar_type() == clone_input.scalar_type());

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(
      infos.numel() == std::max<int64_t>(1, native::batchCount(input)));
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());

  // if 'result' has no elements we can modify it
  if (result.numel() == 0) {
    result.resize_(other.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
    result.transpose_(
        -2, -1); // make 'result' to have Fortran contiguous memory layout
  }

  // if 'clone_input' has no elements we can modify it
  if (clone_input.numel() == 0) {
    clone_input.resize_(
        input.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
    clone_input.transpose_(-2, -1);
  }

  // 'result' and 'clone_input' must be in batched column major order
  TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT(clone_input.transpose(-2, -1).is_contiguous());

  TORCH_INTERNAL_ASSERT(result.sizes().equals(other.sizes()));
  TORCH_INTERNAL_ASSERT(clone_input.sizes().equals(input.sizes()));
  result.copy_(other);
  clone_input.copy_(input);

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      clone_input.scalar_type(), "triangular_solve_xpu", [&] {
        impl::apply_triangular_solve<scalar_t>(
            clone_input,
            result,
            /*left=*/true,
            upper,
            transpose,
            unitriangular);
      });

  return std::tuple<Tensor, Tensor>(result, clone_input);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor, Tensor> triangular_solve(
    const Tensor& self,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular) {
  TORCH_CHECK(
      self.dim() >= 2,
      "torch.triangular_solve: Expected b to have at least 2 dimensions, but it has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(
      A.dim() >= 2,
      "torch.triangular_solve: Expected A to have at least 2 dimensions, but it has ",
      A.dim(),
      " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) =
      native::_linalg_broadcast_batch_dims(self, A, "triangular_solve");

  Tensor result = at::empty({0}, self.options());
  Tensor clone_A = at::empty({0}, self.options());
  Tensor infos = at::zeros(
      {std::max<int64_t>(1, native::batchCount(self_broadcasted))},
      self.options().dtype(kInt));

  at::AtenIpexTypeXPU::_triangular_solve_helper(
      result,
      clone_A,
      infos,
      A_broadcasted,
      self_broadcasted,
      upper,
      transpose,
      unitriangular);

  // TODO: need to rebase 1.13
  // if (self_broadcasted.dim() > 2) {
  //   native::batchCheckErrors(infos, "triangular_solve");
  // } else {
  //   native::singleCheckErrors(infos.item().toInt(), "triangular_solve");
  // }

  return std::tuple<Tensor, Tensor>(result, clone_A);
}

std::tuple<Tensor&, Tensor&> triangular_solve_out(
    const Tensor& self,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    Tensor& result,
    Tensor& clone_A) {
  Tensor result_tmp, clone_A_tmp;
  std::tie(result_tmp, clone_A_tmp) = at::AtenIpexTypeXPU::triangular_solve(
      self, A, upper, transpose, unitriangular);
  result.resize_as_(result_tmp).copy_(result_tmp);
  clone_A.resize_as_(clone_A_tmp).copy_(clone_A_tmp);
  return std::tuple<Tensor&, Tensor&>(result, clone_A);
}

Tensor _cholesky_solve_helper(
    const Tensor& self,
    const Tensor& input2,
    bool upper) {
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto input2_working_copy = native::cloneBatchedColumnMajor(input2);
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "cholesky_solve_dpcpp", [&] {
    impl::apply_cholesky_solve_dpcpp_<scalar_t>(
        self_working_copy, input2_working_copy, upper, infos);
  });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(
      infos_tensor, "cholesky_solve_dpcpp", self.dim() == 2);

  return self_working_copy;
}

Tensor _cholesky_helper(const Tensor& self, bool upper) {
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self.scalar_type(), "cholesky_dpcpp", [&] {
        impl::apply_cholesky_dpcpp<scalar_t>(self_working_copy, upper, infos);
      });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(infos_tensor, "cholesky_dpcpp", self.dim() == 2);

  return self_working_copy;
}

Tensor cholesky(const Tensor& self, bool upper) {
  TORCH_CHECK(
      self.dim() >= 2,
      "input must be 2-d matrix at least, input shape=",
      self.sizes());
  if (self.size(-1) == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  native::squareCheckInputs(self, "cholesky");
  auto raw_cholesky_output = at::AtenIpexTypeXPU::_cholesky_helper(self, upper);
  if (upper) {
    return raw_cholesky_output.triu_();
  } else {
    return raw_cholesky_output.tril_();
  }
}

Tensor& cholesky_out(const Tensor& self, bool upper, Tensor& out) {
  Tensor out_tmp = at::AtenIpexTypeXPU::cholesky(self, upper);
  out.resize_as_(out_tmp).copy_(out_tmp);
  return out;
}

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

Tensor _det_lu_based_helper_backward_helper(
    const Tensor& det_grad,
    const Tensor& det,
    const Tensor& self,
    const Tensor& lu,
    const Tensor& pivs) {
  auto eps = at::native::_get_epsilon(c10::toRealValueType(self.scalar_type()));
  auto n = self.size(-1);
  auto eps_tensor = at::tensor(eps, self.options());
  auto condition_diagonal = [&](const Tensor& x) {
    auto x_diag = x.diagonal(0, -2, -1);
    auto x_diag_conditioned = at::where(x_diag == 0.0, eps_tensor, x_diag);
    x_diag.copy_(x_diag_conditioned);
  };

  // create a matrix d := (det_grad * det.conj()) I
  // NOTE: we do not use the shorter version
  // auto d = at::zeros_like(self);
  // d.diagonal(0, -2, -1).copy_((det_grad * det.conj()).unsqueeze(-1));
  // to avoid in-place operations to eliminate potential issues with Vmap
  auto det_expanded_sizes = det.sizes().vec();
  det_expanded_sizes.push_back(n);
  auto d_diag = det_grad * det.conj();
  auto d = at::diag_embed(d_diag.unsqueeze(-1).expand(det_expanded_sizes));
  // make sure that d is Fortran-contiguous. The transposition is sufficient as
  // d is a diagonal square matrix
  d = d.transpose(-2, -1);

  // we want to condition the diagonal of the lu Tensor, but it is not allowed
  // to modify arguments of backward functions in-place, hence the cloning.
  auto lu_clone = lu.clone();
  condition_diagonal(lu_clone);

  auto trans = self.is_complex() ? TransposeType::ConjTranspose
                                 : TransposeType::Transpose;
  auto infos_tensor = at::zeros(
      native::batchCount(d),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(d), 0);

  // d is modified in-place and will contain the result
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      d.scalar_type(), "_det_lu_based_helper_backward_helper", [&] {
        impl::apply_lu_solve_dpcpp_<scalar_t>(d, lu_clone, pivs, infos, trans);
      });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(
      infos_tensor, "_det_lu_based_helper_backward_helper", self.dim() == 2);

  return d;
}

void linalg_eigh_impl(
    Tensor& eigenvalues,
    Tensor& eigenvectors,
    Tensor& infos,
    bool upper,
    bool compute_eigenvectors) {
  std::vector<int32_t> infos_vec(at::native::batchCount(eigenvectors), 0);

  auto self_sizes = eigenvectors.sizes().vec();
  self_sizes.pop_back();

  auto self_working_copy = at::native::cloneBatchedColumnMajor(eigenvectors);
  IPEX_DISPATCH_FLOATING_TYPES(eigenvectors.scalar_type(), "symeig", [&] {
    impl::apply_symeig<scalar_t>(
        self_working_copy, eigenvalues, compute_eigenvectors, upper, infos_vec);
  });

  Tensor infos_tensor = from_blob(infos_vec.data(), {1, infos_vec.size()});
  infos_tensor = infos_tensor.to(kInt);
  at::_linalg_check_errors(infos_tensor, "symeig", infos.dim() == 2);
  // at::_linalg_check_errors(infos, "symeig", infos.dim() == 2);

  if (compute_eigenvectors) {
    eigenvectors.copy_(self_working_copy);
  } else {
    eigenvectors = at::empty({0}, eigenvectors.options());
  }
}

void linalg_eigh_out_info(
    const Tensor& input,
    const Tensor& values,
    const Tensor& vectors,
    const Tensor& infos,
    bool compute_eigenvectors,
    const c10::string_view uplo_str) {
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  // TODO: enable complex type for this method
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == vectors.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == values.device());

  // eigenvalues are always real-valued
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  ScalarType real_dtype = c10::toRealValueType(input.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.scalar_type() == real_dtype);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      input.scalar_type() == vectors.scalar_type());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == input.device());

  // infos can have the shape equal to input.shape[:-2] or (batchCount(input),
  // ), both would work with the current implementation. infos.shape ==
  // input.shape[:-2] might be useful in the future for easier checking the
  // error code for the specific matrix in batched input when we would have a
  // user-exposed way to get infos tensor. 1-dimensional tensor of shape
  // (batchCount(input), ) is currently used for the internal implementation
  // everywhere.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      infos.numel() == std::max<int64_t>(1, native::batchCount(input)));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.is_contiguous());

  // if 'vectors' has no elements we can modify it
  if (vectors.numel() == 0) {
    vectors.resize_(input.sizes(), MemoryFormat::Contiguous);
    vectors.transpose_(
        -2, -1); // make 'vectors' to have Fortran contiguous memory layout
  }

  // if 'values' has no elements we can modify it
  auto values_shape =
      IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  if (values.numel() == 0) {
    values.resize_(values_shape, MemoryFormat::Contiguous);
  }

  // 'vectors' must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.sizes().equals(input.sizes()));

  // 'values' must be contiguous
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.sizes().equals(values_shape));

  vectors.copy_(input);

  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  char uplo = std::toupper(uplo_str[0]);
  bool upper = (uplo == 'U');

  Tensor value_non_const = values;
  Tensor vectors_non_const = vectors;
  Tensor infos_non_const = infos;
  linalg_eigh_impl(
      value_non_const,
      vectors_non_const,
      infos_non_const,
      upper,
      compute_eigenvectors);
}

std::tuple<Tensor, Tensor> linalg_eigh(
    const Tensor& input,
    c10::string_view uplo) {
  at::native::squareCheckInputs(input, "eigh input");
  at::native::checkUplo(uplo);
  ScalarType real_dtype = c10::toRealValueType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(real_dtype));
  Tensor vectors = at::empty({0}, input.options());
  Tensor infos = at::zeros(
      {std::max<int64_t>(1, native::batchCount(input))},
      input.options().dtype(kInt));
  linalg_eigh_out_info(input, values, vectors, infos, true, uplo);
  at::_linalg_check_errors(infos, "torch.linalg.eigh", infos.dim() == 2);

  return std::tuple<Tensor, Tensor>(values, vectors);
}

// TODO: it's possible to make the _out variant to be a primal function and
// implement linalg_eigh on top of _out
// TODO: implement _out variant avoiding copy and using already allocated
// storage directly
std::tuple<Tensor&, Tensor&> linalg_eigh_out(
    const Tensor& input,
    c10::string_view uplo,
    Tensor& eigvals,
    Tensor& eigvecs) {
  TORCH_CHECK(
      input.device() == eigvals.device(),
      "Expected two tensors to be on the same device, but "
      "found at least two devices, ",
      input.device(),
      " and ",
      eigvals.device(),
      "!");

  TORCH_CHECK(
      input.device() == eigvecs.device(),
      "Expected two tensors to be on the same device, but "
      "found at least two devices, ",
      input.device(),
      " and ",
      eigvecs.device(),
      "!");
  at::native::checkLinalgCompatibleDtype(
      "torch.linalg.eigh", eigvecs, input, "eigenvectors");

  // eigenvalues are always real-valued here
  ScalarType real_dtype = c10::toRealValueType(input.scalar_type());
  at::native::checkLinalgCompatibleDtype(
      "torch.linalg.eigh", eigvals.scalar_type(), real_dtype, "eigenvalues");

  Tensor eigvals_tmp, eigvecs_tmp;
  std::tie(eigvals_tmp, eigvecs_tmp) =
      at::AtenIpexTypeXPU::linalg_eigh(input, uplo);

  at::native::resize_output(eigvals, eigvals_tmp.sizes());
  eigvals.copy_(eigvals_tmp);
  at::native::resize_output(eigvecs, eigvecs_tmp.sizes());
  eigvecs.copy_(eigvecs_tmp);

  return std::tuple<Tensor&, Tensor&>(eigvals, eigvecs);
}

Tensor linalg_eigvalsh(const Tensor& A, c10::string_view uplo) {
  return std::get<0>(at::AtenIpexTypeXPU::linalg_eigh(A, uplo));
}

Tensor& linalg_eigvalsh_out(
    const Tensor& input,
    c10::string_view uplo,
    Tensor& result) {
  ScalarType real_dtype = c10::toRealValueType(input.scalar_type());
  TORCH_CHECK(
      input.device() == result.device(),
      "Expected two tensors to be on the same device, but "
      "found at least two devices, ",
      input.device(),
      " and ",
      result.device(),
      "!");
  at::native::checkLinalgCompatibleDtype(
      "torch.linalg.eigvalsh", result.scalar_type(), real_dtype);

  at::native::squareCheckInputs(input, "eigvalsh input");
  at::native::checkUplo(uplo);

  auto expected_result_shape =
      IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  bool result_equal_expected_shape =
      result.sizes().equals(expected_result_shape);
  bool expected_result_type = (result.scalar_type() == real_dtype);
  bool copy_needed = !expected_result_type;
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape);
  copy_needed |= (result.numel() != 0 && !result.is_contiguous());

  Tensor vectors = at::empty({0}, input.options());
  Tensor infos = at::zeros(
      {std::max<int64_t>(1, at::native::batchCount(input))},
      input.options().dtype(kInt));

  if (copy_needed) { // we have to allocate a temporary tensor
    Tensor result_tmp =
        at::empty({expected_result_shape}, input.options().dtype(real_dtype));
    linalg_eigh_out_info(
        input,
        result_tmp,
        vectors,
        infos,
        /*compute_eigenvectors=*/false,
        uplo);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    // else use the provided output storage directly
    linalg_eigh_out_info(
        input, result, vectors, infos, /*compute_eigenvectors=*/false, uplo);
  }

  at::_linalg_check_errors(infos, "torch.linalg.eigvalsh", infos.dim() == 2);

  return result;
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

std::tuple<Tensor&, Tensor&, Tensor&> _linalg_det_out(
    const Tensor& A,
    Tensor& result,
    Tensor& LU,
    Tensor& pivots) {
  auto shape = A.sizes();
  auto ndim = shape.size();

  // det
  set_contiguous(result, shape.slice(0, ndim - 2));

  // LU
  auto LU_strides =
      at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_strided(LU, shape, LU_strides);

  // pivots
  set_contiguous(pivots, shape.slice(0, ndim - 1));

  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it
  // copies Use the transpose of if A is contiguous since det(A^T) = det(A) We
  // limit this to real matrices, but it could also be implemented for complex
  // matrices
  at::linalg_lu_factor_ex_out(
      const_cast<Tensor&>(LU),
      const_cast<Tensor&>(pivots),
      const_cast<Tensor&>(info),
      A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  // det = det_P * prod(diag(LU))
  at::mul_out(
      const_cast<Tensor&>(result),
      lu_det_P(pivots),
      at::prod(LU.diagonal(0, -2, -1), /*dim=*/-1));
  return std::tuple<Tensor&, Tensor&, Tensor&>(result, LU, pivots);
}

// In PyTorch1.10, inverse is implemented by MKL api getrf + getri. In
// PyTorch1.13, it used getrf + getrs. getrs solves a system of linear
// equations, getri can be coverd by getrs We keep two implementations and will
// switch to getrf + getrs after verify performance.
std::tuple<Tensor&, Tensor&> linalg_inv_ex_out(
    const Tensor& A,
    bool check_errors,
    Tensor& result,
    Tensor& info) {
  std::vector<int32_t> infos_lu_vec(native::batchCount(result), 0);
  std::vector<int32_t> infos_getri_vec(native::batchCount(result), 0);
  auto input_copy =
      native::cloneBatchedColumnMajor(A); // get column major input tensor
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "apply_inverse_dpcpp_", [&] {
        impl::apply_inverse_dpcpp_<scalar_t>(
            input_copy, result, infos_lu_vec, infos_getri_vec);
      });
  auto expected_info_shape =
      IntArrayRef(A.sizes().cbegin(), A.sizes().cend() - 2);
  info.copy_(at::from_blob(
      (int32_t*)(infos_lu_vec.data()),
      expected_info_shape,
      c10::toRealValueType(info.scalar_type())));

  return std::tuple<Tensor&, Tensor&>(result, info);
}

// // The api design follow PyTorch (getrf + getrs)
// std::tuple<Tensor&, Tensor&> linalg_inv_ex_out(
//     const Tensor& A,
//     bool check_errors,
//     Tensor& result,
//     Tensor& info) {
//   // Fill result with the identity
//   result.zero_();
//   result.diagonal(0, -2, -1).fill_(1.);
//   at::linalg_solve_ex_out(
//       const_cast<Tensor&>(result),
//       const_cast<Tensor&>(info),
//       A,
//       result,
//       /*left*/ true); // (result, info, A, B)
//   if (check_errors) {
//     at::_linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
//   }
//   return std::tuple<Tensor&, Tensor&>(result, info);
// }

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _linalg_solve_ex_out(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool check_errors,
    Tensor& result,
    Tensor& LU,
    Tensor& pivots,
    Tensor& info) {
  TORCH_CHECK(
      A.scalar_type() == B.scalar_type(),
      "linalg.solve: Expected A and B to have the same dtype, but found A of type ",
      A.scalar_type(),
      " and B of type ",
      B.scalar_type(),
      " instead");

  // NumPy compat: Two types of 'B' tensors are supported:
  // - 1D tensor or batch of 1D tensors (vector case)
  // - 2D tensor or batch of 2D tensors (matrix case)
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(A, B);
  auto B_ = vector_case ? B.unsqueeze(-1) : B;

  // matrix shapes
  at::native::checkInputsSolver(A, B_, /*left=*/left, "linalg.solve");

  // Check that B can be broadcasted to the shape of A
  auto B_broad_shape =
      std::get<0>(at::native::_linalg_broadcast_batch_dims(B_, A));
  // We disallow the broadcasting of B as a vector when left=False as, in that
  // case, A.shape = (*, 1, 1)
  TORCH_CHECK(
      left || !vector_case,
      "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. In this case linalg.solve is equivalent to B / A.squeeze(-1)");
  auto result_shape = vector_case
      ? IntArrayRef(B_broad_shape.data(), B_broad_shape.size() - 1)
      : B_broad_shape;
  auto result_strides = at::native::batched_matrix_contiguous_strides(
      result_shape, /*column_major=*/left);

  result.resize_(result_shape);
  result.as_strided_(result_shape, result_strides);
  auto shape = A.sizes();
  auto ndim = shape.size();

  // LU
  auto LU_strides =
      at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_strided(LU, shape, LU_strides);

  // pivots
  set_contiguous(pivots, shape.slice(0, ndim - 1));

  // info
  set_contiguous(info, shape.slice(0, ndim - 2));

  const bool use_A_T = A.is_contiguous() && !A.is_complex();
  at::linalg_lu_factor_ex_out(
      const_cast<Tensor&>(LU),
      const_cast<Tensor&>(pivots),
      const_cast<Tensor&>(info),
      use_A_T ? A.mT() : A);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.solve_ex", A.dim() == 2);
  }

  // [numpy-compat] Handle vectors on the rhs
  const bool vector_case_B = at::native::linalg_solve_is_vector_rhs(LU, B);
  auto result_ = vector_case_B ? result.unsqueeze(-1) : result;
  at::linalg_lu_solve_out(result_, LU, pivots, B_, left, /*adjoint*/ use_A_T);
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(
      result, LU, pivots, info);
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

std::tuple<Tensor&, Tensor&, Tensor&> linalg_lu_factor_ex_out(
    const Tensor& A,
    bool pivot,
    bool check_errors,
    Tensor& LU,
    Tensor& pivots,
    Tensor& info) {
  TORCH_CHECK(
      pivot,
      "linalg.lu_factor: LU without pivoting is not implemented on the XPU");
  if (A.numel() == 0) {
    // zero out the infos as it will have one element if the input is a matrix
    // of size (0, 0)
    info.zero_();
    return std::tuple<Tensor&, Tensor&, Tensor&>(LU, pivots, info);
  }

  if (!LU.is_same(A)) {
    LU.copy_(A);
  }
  // handle the info
  std::vector<int32_t> infos_vec(native::batchCount(A), 0);
  // mkl needs long for pivots, but PT is int
  Tensor pivots_ = at::empty(pivots.sizes(), pivots.options().dtype(kLong));
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_dpcpp", [&] {
    impl::apply_lu_dpcpp_<scalar_t>(LU, pivots_, infos_vec);
  });
  auto expected_info_shape =
      IntArrayRef(LU.sizes().cbegin(), LU.sizes().cend() - 2);

  info.copy_(at::from_blob(
      (int32_t*)(infos_vec.data()),
      expected_info_shape,
      c10::toRealValueType(info.scalar_type())));

  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.lu_factor_ex", A.dim() == 2);
  }
  // Copy to original pivots tensor
  pivots.copy_(pivots_);
  return std::tuple<Tensor&, Tensor&, Tensor&>(LU, pivots, info);
}

} // namespace AtenIpexTypeXPU
} // namespace at
