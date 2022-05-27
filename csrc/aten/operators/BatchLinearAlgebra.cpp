#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>

#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>

#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

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
    DPCPP::queue& queue,
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
    DPCPP::queue& queue,
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
    DPCPP::queue& queue,
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
    DPCPP::queue& queue,
    oneapi::mkl::transpose* trans,
    int64_t* n,
    int64_t* nrhs,
    int64_t* lda,
    int64_t* ldb,
    int64_t group_count,
    int64_t* group_sizes) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<scalar_t>(
      queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}

template <>
int64_t mkl_getrs_scratchpad<c10::complex<double>>(
    DPCPP::queue& queue,
    oneapi::mkl::transpose* trans,
    int64_t* n,
    int64_t* nrhs,
    int64_t* lda,
    int64_t* ldb,
    int64_t group_count,
    int64_t* group_sizes) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<double>>(
      queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}

template <>
int64_t mkl_getrs_scratchpad<c10::complex<float>>(
    DPCPP::queue& queue,
    oneapi::mkl::transpose* trans,
    int64_t* n,
    int64_t* nrhs,
    int64_t* lda,
    int64_t* ldb,
    int64_t group_count,
    int64_t* group_sizes) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<float>>(
      queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}

template <typename scalar_t>
int64_t mkl_getri_scratchpad(
    DPCPP::queue& queue,
    int64_t* n,
    int64_t* lda,
    int64_t group_count,
    int64_t* group_sizes) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<scalar_t>(
      queue, n, lda, group_count, group_sizes);
}

template <>
int64_t mkl_getri_scratchpad<c10::complex<double>>(
    DPCPP::queue& queue,
    int64_t* n,
    int64_t* lda,
    int64_t group_count,
    int64_t* group_sizes) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<double>>(
      queue, n, lda, group_count, group_sizes);
}

template <>
int64_t mkl_getri_scratchpad<c10::complex<float>>(
    DPCPP::queue& queue,
    int64_t* n,
    int64_t* lda,
    int64_t group_count,
    int64_t* group_sizes) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<float>>(
      queue, n, lda, group_count, group_sizes);
}

template <typename scalar_t>
void mkl_getrf(
    DPCPP::queue& queue,
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
    DPCPP::queue& queue,
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
    DPCPP::queue& queue,
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
    DPCPP::queue& queue,
    oneapi::mkl::transpose* trans,
    int64_t* n,
    int64_t* nrhs,
    scalar_t** a,
    int64_t* lda,
    int64_t** ipiv,
    scalar_t** b,
    int64_t* ldb,
    int64_t group_count,
    int64_t* group_sizes,
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
      ipiv,
      b,
      ldb,
      group_count,
      group_sizes,
      scratchpad,
      scratchpad_size);
}

template <>
void mkl_getrs<c10::complex<double>>(
    DPCPP::queue& queue,
    oneapi::mkl::transpose* trans,
    int64_t* n,
    int64_t* nrhs,
    c10::complex<double>** a,
    int64_t* lda,
    int64_t** ipiv,
    c10::complex<double>** b,
    int64_t* ldb,
    int64_t group_count,
    int64_t* group_sizes,
    c10::complex<double>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      reinterpret_cast<std::complex<double>**>(a),
      lda,
      ipiv,
      reinterpret_cast<std::complex<double>**>(b),
      ldb,
      group_count,
      group_sizes,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpad_size);
}

template <>
void mkl_getrs<c10::complex<float>>(
    DPCPP::queue& queue,
    oneapi::mkl::transpose* trans,
    int64_t* n,
    int64_t* nrhs,
    c10::complex<float>** a,
    int64_t* lda,
    int64_t** ipiv,
    c10::complex<float>** b,
    int64_t* ldb,
    int64_t group_count,
    int64_t* group_sizes,
    c10::complex<float>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      reinterpret_cast<std::complex<float>**>(a),
      lda,
      ipiv,
      reinterpret_cast<std::complex<float>**>(b),
      ldb,
      group_count,
      group_sizes,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpad_size);
}

template <typename scalar_t>
void mkl_getri(
    DPCPP::queue& queue,
    int64_t* n,
    scalar_t** a,
    int64_t* lda,
    int64_t** ipiv,
    int64_t group_count,
    int64_t* group_sizes,
    scalar_t* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getri_batch,
      queue,
      n,
      a,
      lda,
      ipiv,
      group_count,
      group_sizes,
      scratchpad,
      scratchpad_size);
}

template <>
void mkl_getri<c10::complex<double>>(
    DPCPP::queue& queue,
    int64_t* n,
    c10::complex<double>** a,
    int64_t* lda,
    int64_t** ipiv,
    int64_t group_count,
    int64_t* group_sizes,
    c10::complex<double>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getri_batch,
      queue,
      n,
      reinterpret_cast<std::complex<double>**>(a),
      lda,
      ipiv,
      group_count,
      group_sizes,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpad_size);
}

template <>
void mkl_getri<c10::complex<float>>(
    DPCPP::queue& queue,
    int64_t* n,
    c10::complex<float>** a,
    int64_t* lda,
    int64_t** ipiv,
    int64_t group_count,
    int64_t* group_sizes,
    c10::complex<float>* scratchpad,
    int64_t scratchpad_size) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getri_batch,
      queue,
      n,
      reinterpret_cast<std::complex<float>**>(a),
      lda,
      ipiv,
      group_count,
      group_sizes,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpad_size);
}

template <typename scalar_t>
int64_t mkl_geqrf_batch_scratchpad_size(
    DPCPP::queue& queue,
    int64_t* m,
    int64_t* n,
    int64_t* lda,
    int64_t group_count,
    int64_t* batch_size) {
  return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<scalar_t>(
      queue, m, n, lda, group_count, batch_size);
}

template <>
int64_t mkl_geqrf_batch_scratchpad_size<c10::complex<float>>(
    DPCPP::queue& queue,
    int64_t* m,
    int64_t* n,
    int64_t* lda,
    int64_t group_count,
    int64_t* batch_size) {
  return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<float>>(
      queue, m, n, lda, group_count, batch_size);
}

template <>
int64_t mkl_geqrf_batch_scratchpad_size<c10::complex<double>>(
    DPCPP::queue& queue,
    int64_t* m,
    int64_t* n,
    int64_t* lda,
    int64_t group_count,
    int64_t* batch_size) {
  return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<double>>(
      queue, m, n, lda, group_count, batch_size);
}

template <typename scalar_t>
void mkl_geqrf_batch(
    DPCPP::queue& queue,
    int64_t* m,
    int64_t* n,
    scalar_t** a,
    int64_t* lda,
    scalar_t** tau,
    int64_t group_count,
    int64_t* group_sizes,
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
      tau,
      group_count,
      group_sizes,
      (scalar_t*)scratchpad,
      scratchpadsize);
}

template <>
void mkl_geqrf_batch<c10::complex<float>>(
    DPCPP::queue& queue,
    int64_t* m,
    int64_t* n,
    c10::complex<float>** a,
    int64_t* lda,
    c10::complex<float>** tau,
    int64_t group_count,
    int64_t* group_sizes,
    c10::complex<float>* scratchpad,
    int64_t scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::geqrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<float>**>(a),
      lda,
      reinterpret_cast<std::complex<float>**>(tau),
      group_count,
      group_sizes,
      reinterpret_cast<std::complex<float>*>(scratchpad),
      scratchpadsize);
}

template <>
void mkl_geqrf_batch<c10::complex<double>>(
    DPCPP::queue& queue,
    int64_t* m,
    int64_t* n,
    c10::complex<double>** a,
    int64_t* lda,
    c10::complex<double>** tau,
    int64_t group_count,
    int64_t* group_sizes,
    c10::complex<double>* scratchpad,
    int64_t scratchpadsize) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::geqrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<double>**>(a),
      lda,
      reinterpret_cast<std::complex<double>**>(tau),
      group_count,
      group_sizes,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpadsize);
}
#endif

#ifdef USE_ONEMKL
void error_handle(
    std::vector<int64_t>& infos,
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
    } catch (DPCPP::exception e) {
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
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
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
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
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
    std::vector<int64_t>& infos_) {
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
    std::vector<int64_t>& infos_,
    TransposeType t) {
#ifdef USE_ONEMKL
  // do nothing if empty input
  if (lu_.numel() == 0)
    return;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t batch_size = native::batchCount(b_);
  int64_t group_count = (batch_size + local_size - 1) / local_size;
  int64_t* group_sizes = new int64_t[group_count];
  for (auto i = 0; i < group_count; i++)
    group_sizes[i] = std::min(local_size, batch_size - i * local_size);

  std::vector<oneapi::mkl::transpose> trans(group_count, to_blas(t));
  std::vector<int64_t> n(group_count, lu_.size(-2));
  std::vector<int64_t> nrhs(group_count, b_.size(-1));
  std::vector<int64_t> lda(group_count, lu_.size(-2));
  std::vector<int64_t> ldb(group_count, b_.size(-2));

  scalar_t* a_ptr = (scalar_t*)(lu_.data_ptr());
  int64_t* ipiv_ptr = (int64_t*)(pivots_.data_ptr());
  scalar_t* b_ptr = (scalar_t*)(b_.data_ptr());
  std::vector<scalar_t*> a;
  std::vector<int64_t*> ipiv;
  std::vector<scalar_t*> b;
  int64_t stride_a = native::matrixStride(lu_);
  int64_t stride_ipiv = pivots_.size(-1);
  int64_t stride_b = native::matrixStride(b_);
  for (auto i = 0; i < batch_size; i++) {
    a.push_back(&a_ptr[i * stride_a]);
    ipiv.push_back(&ipiv_ptr[i * stride_ipiv]);
    b.push_back(&b_ptr[i * stride_b]);
  }

  int64_t scratchpadsize = mkl_getrs_scratchpad<scalar_t>(
      dpcpp_queue,
      trans.data(),
      n.data(),
      nrhs.data(),
      lda.data(),
      ldb.data(),
      group_count,
      group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, b_.options());
  try {
    mkl_getrs<scalar_t>(
        dpcpp_queue,
        trans.data(),
        n.data(),
        nrhs.data(),
        a.data(),
        lda.data(),
        ipiv.data(),
        b.data(),
        ldb.data(),
        group_count,
        group_sizes,
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
static void apply_inverse_dpcpp_(Tensor& self_, std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto req_size = self_.sizes().vec();
  req_size.pop_back();
  Tensor pivots_ = at::empty(req_size, self_.options().dtype(kLong));
  impl::apply_lu_dpcpp_<scalar_t>(self_, pivots_, infos_);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t batch_size = native::batchCount(self_);
  int64_t group_count = (batch_size + local_size - 1) / local_size;
  int64_t* group_sizes = new int64_t[group_count];
  for (auto i = 0; i < group_count; i++)
    group_sizes[i] = std::min(local_size, batch_size - i * local_size);

  std::vector<int64_t> n(group_count, self_.size(-2));
  std::vector<int64_t> lda(group_count, self_.size(-2));

  scalar_t* a_ptr = (scalar_t*)(self_.data_ptr());
  int64_t* ipiv_ptr = (int64_t*)(pivots_.data_ptr());
  std::vector<scalar_t*> a;
  std::vector<int64_t*> ipiv;
  int64_t stride_a = native::matrixStride(self_);
  int64_t stride_ipiv = pivots_.size(-1);
  for (auto i = 0; i < batch_size; i++) {
    a.push_back(&a_ptr[i * stride_a]);
    ipiv.push_back(&ipiv_ptr[i * stride_ipiv]);
  }

  int64_t scratchpadsize = mkl_getri_scratchpad<scalar_t>(
      dpcpp_queue, n.data(), lda.data(), group_count, group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    mkl_getri<scalar_t>(
        dpcpp_queue,
        n.data(),
        a.data(),
        lda.data(),
        ipiv.data(),
        group_count,
        group_sizes,
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
static void apply_geqrf_dpcpp_(
    Tensor& self_,
    Tensor& tau_,
    int64_t m_,
    int64_t n_,
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t batch_size = native::batchCount(self_);
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t group_count = (batch_size + local_size - 1) / local_size;
  int64_t* group_sizes = new int64_t[group_count];
  for (auto i = 0; i < group_count; i++)
    group_sizes[i] = std::min(local_size, batch_size - i * local_size);

  std::vector<int64_t> m(group_count, m_);
  std::vector<int64_t> n(group_count, n_);
  std::vector<int64_t> lda(group_count, self_.size(-2));

  scalar_t* a_ptr = (scalar_t*)(self_.data_ptr());
  scalar_t* tau_ptr = (scalar_t*)(tau_.data_ptr());
  std::vector<scalar_t*> a;
  std::vector<scalar_t*> tau;
  int64_t stride_a = native::matrixStride(self_);
  int64_t stride_tau = tau_.size(-1);
  for (auto i = 0; i < batch_size; i++) {
    a.push_back(&a_ptr[i * stride_a]);
    tau.push_back(&tau_ptr[i * stride_tau]);
  }

  int64_t scratchpadsize = mkl_geqrf_batch_scratchpad_size<scalar_t>(
      dpcpp_queue, m.data(), n.data(), lda.data(), group_count, group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    mkl_geqrf_batch<scalar_t>(
        dpcpp_queue,
        m.data(),
        n.data(),
        a.data(),
        lda.data(),
        tau.data(),
        group_count,
        group_sizes,
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
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t batch_size = native::batchCount(self_);
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t group_count = (batch_size + local_size - 1) / local_size;
  int64_t* group_sizes = new int64_t[group_count];
  for (auto i = 0; i < group_count; i++)
    group_sizes[i] = std::min(local_size, batch_size - i * local_size);

  std::vector<int64_t> m(group_count, m_);
  std::vector<int64_t> n(group_count, n_columns_);
  std::vector<int64_t> k(group_count, k_);
  std::vector<int64_t> lda(group_count, self_.size(-2));

  scalar_t* a_ptr = (scalar_t*)(self_.data_ptr());
  scalar_t* tau_ptr = (scalar_t*)(tau_.data_ptr());
  std::vector<scalar_t*> a;
  std::vector<scalar_t*> tau;
  int64_t stride_a = native::matrixStride(self_);
  int64_t stride_tau = tau_.size(-1);
  for (auto i = 0; i < batch_size; i++) {
    a.push_back(&a_ptr[i * stride_a]);
    tau.push_back(&tau_ptr[i * stride_tau]);
  }

  int64_t scratchpadsize =
      oneapi::mkl::lapack::orgqr_batch_scratchpad_size<scalar_t>(
          dpcpp_queue,
          m.data(),
          n.data(),
          k.data(),
          lda.data(),
          group_count,
          group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::orgqr_batch,
        dpcpp_queue,
        m.data(),
        n.data(),
        k.data(),
        a.data(),
        lda.data(),
        tau.data(),
        group_count,
        group_sizes,
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
  ScalarType dtype = toValueType(typeMetaToScalarType(input.dtype()));
  S_empty = at::empty(sizes, input.options().dtype(dtype));
  return std::tuple<Tensor, Tensor, Tensor>(U_empty, S_empty, VT_empty);
}

template <typename scalar_t, typename value_t>
static void apply_svd(
    DPCPP::queue& dpcpp_queue,
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
    DPCPP::queue& dpcpp_queue,
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
    DPCPP::queue& dpcpp_queue,
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
    std::vector<int64_t>& infos) {
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
    Tensor& b,
    Tensor& A,
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

  auto n = A.size(-2);
  auto nrhs = b.size(-1);
  std::int64_t lda = A.size(-2);
  std::int64_t ldb = b.size(-2);
  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::trtrs_scratchpad_size<scalar_t>(
          dpcpp_queue, uplo, trans, diag, n, nrhs, lda, ldb);
  Tensor scratchpad_at = at::empty({scratchpadsize}, A.options());
  DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::lapack::trtrs,
      dpcpp_queue,
      uplo,
      trans,
      diag,
      n,
      nrhs,
      (scalar_t*)(A.data_ptr()),
      lda,
      (scalar_t*)(b.data_ptr()),
      ldb,
      (scalar_t*)(scratchpad_at.data_ptr()),
      scratchpadsize);
#else
  AT_ERROR("triangular_solve: oneMKL library not found in compilation");
#endif
}

template <>
void apply_triangular_solve<c10::complex<float>>(
    Tensor& b,
    Tensor& A,
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

  auto n = A.size(-2);
  auto nrhs = b.size(-1);
  std::int64_t lda = A.size(-2);
  std::int64_t ldb = b.size(-2);
  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::trtrs_scratchpad_size<std::complex<float>>(
          dpcpp_queue, uplo, trans, diag, n, nrhs, lda, ldb);
  Tensor scratchpad_at = at::empty({scratchpadsize}, A.options());
  DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::lapack::trtrs,
      dpcpp_queue,
      uplo,
      trans,
      diag,
      n,
      nrhs,
      reinterpret_cast<std::complex<float>*>(A.data_ptr()),
      lda,
      reinterpret_cast<std::complex<float>*>(b.data_ptr()),
      ldb,
      reinterpret_cast<std::complex<float>*>(scratchpad_at.data_ptr()),
      scratchpadsize);
#else
  AT_ERROR("triangular_solve: oneMKL library not found in compilation");
#endif
}

template <>
void apply_triangular_solve<c10::complex<double>>(
    Tensor& b,
    Tensor& A,
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

  auto n = A.size(-2);
  auto nrhs = b.size(-1);
  std::int64_t lda = A.size(-2);
  std::int64_t ldb = b.size(-2);
  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::trtrs_scratchpad_size<std::complex<double>>(
          dpcpp_queue, uplo, trans, diag, n, nrhs, lda, ldb);
  Tensor scratchpad_at = at::empty({scratchpadsize}, A.options());
  DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::lapack::trtrs,
      dpcpp_queue,
      uplo,
      trans,
      diag,
      n,
      nrhs,
      reinterpret_cast<std::complex<double>*>(A.data_ptr()),
      lda,
      reinterpret_cast<std::complex<double>*>(b.data_ptr()),
      ldb,
      reinterpret_cast<std::complex<double>*>(scratchpad_at.data_ptr()),
      scratchpadsize);
#else
  AT_ERROR("triangular_solve: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_cholesky_solve_dpcpp_(
    const Tensor& b_,
    const Tensor& A_,
    bool upper_,
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  oneapi::mkl::uplo uplo = upper_ ? oneapi::mkl::uplo::U : oneapi::mkl::uplo::L;
  int64_t batch_size = native::batchCount(b_);
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t group_count = (batch_size + local_size - 1) / local_size;
  int64_t* group_sizes = new int64_t[group_count];
  for (auto i = 0; i < group_count; i++)
    group_sizes[i] = std::min(local_size, batch_size - i * local_size);

  std::vector<oneapi::mkl::transpose> trans(
      group_count, oneapi::mkl::transpose::nontrans);
  std::vector<int64_t> n(group_count, A_.size(-2));
  std::vector<int64_t> nrhs(group_count, b_.size(-1));
  std::vector<int64_t> lda(group_count, A_.size(-2));
  std::vector<int64_t> ldb(group_count, b_.size(-2));

  scalar_t* a_ptr = (scalar_t*)(A_.data_ptr());
  scalar_t* b_ptr = (scalar_t*)(b_.data_ptr());
  std::vector<scalar_t*> a;
  std::vector<scalar_t*> b;
  int64_t stride_a = native::matrixStride(A_);
  int64_t stride_b = native::matrixStride(b_);
  for (auto i = 0; i < batch_size; i++) {
    a.push_back(&a_ptr[i * stride_a]);
    b.push_back(&b_ptr[i * stride_b]);
  }

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrs_batch_scratchpad_size<scalar_t>(
          dpcpp_queue,
          &uplo,
          n.data(),
          nrhs.data(),
          lda.data(),
          ldb.data(),
          group_count,
          group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, b_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrs_batch,
        dpcpp_queue,
        &uplo,
        n.data(),
        nrhs.data(),
        a.data(),
        lda.data(),
        b.data(),
        ldb.data(),
        group_count,
        group_sizes,
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
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo =
      upper_ ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

  auto n = self_.size(-1);

  std::int64_t lda = self_.size(-2);

  scalar_t* a = (scalar_t*)(self_.data_ptr());

  int64_t scratchpadsize = oneapi::mkl::lapack::potrf_scratchpad_size<scalar_t>(
      dpcpp_queue, uplo, n, lda);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrf,
        dpcpp_queue,
        uplo,
        n,
        a,
        lda,
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
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo =
      upper_ ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

  auto n = self_.size(-1);

  std::int64_t lda = self_.size(-2);

  std::complex<float>* a = (std::complex<float>*)(self_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>>(
          dpcpp_queue, uplo, n, lda);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrf,
        dpcpp_queue,
        uplo,
        n,
        a,
        lda,
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
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  oneapi::mkl::uplo uplo =
      upper_ ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

  auto n = self_.size(-1);

  std::int64_t lda = self_.size(-2);

  std::complex<double>* a = (std::complex<double>*)(self_.data_ptr());

  int64_t scratchpadsize =
      oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(
          dpcpp_queue, uplo, n, lda);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potrf,
        dpcpp_queue,
        uplo,
        n,
        a,
        lda,
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
  std::vector<int64_t> infos(native::batchCount(input), 0);
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
  TORCH_CHECK(pivot, "lu without pivoting is not implemented on the DPCPP");
  TORCH_CHECK(
      self.dim() >= 2,
      "expected tensor with 2 or more dimensions, got size: ",
      self.sizes(),
      " instead");
  native::squareCheckInputs(self);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  auto pivots_tensor = at::empty(req_size, self.options().dtype(kLong));
  req_size.pop_back();
  auto infos_tensor =
      at::zeros(req_size, self.options().dtype(kLong).device(DeviceType::CPU));
  std::vector<int64_t> infos(native::batchCount(self), 0);

  Tensor self_working_copy;
  if (self.numel() == 0) {
    self_working_copy = at::empty_like(self);
  } else {
    self_working_copy = native::cloneBatchedColumnMajor(self);
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lu_dpcpp", [&] {
      impl::apply_lu_dpcpp_<scalar_t>(self_working_copy, pivots_tensor, infos);
    });
  }
  if (check_errors) {
    if (self.dim() > 2) {
      native::batchCheckErrors(infos, "lu_dpcpp");
    } else {
      native::singleCheckErrors(infos[0], "lu_dpcpp");
    }
  }
  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int64_t>());
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
    at::native::resize_output(result, result_tmp.sizes());
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
  std::vector<int64_t> infos_vec_1(native::batchCount(input_broadcasted), 0);
  std::vector<int64_t> infos_vec_2(native::batchCount(input_broadcasted), 0);
  // compute the LU factorization of 'input_working_copy'
  auto pivots_shape =
      IntArrayRef(input_broadcasted.sizes().data(), input_broadcasted.dim() - 2)
          .vec(); // input_broadcasted.shape[:-2]
  pivots_shape.push_back(std::min(input.size(-2), input.size(-1)));
  Tensor pivots = at::empty(pivots_shape, input.options().dtype(kInt));
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
      infos.template data_ptr<int64_t>());

  if (input_working_copy.dim() > 2) {
    native::batchCheckErrors(infos_vec_2, "lu_solve_dpcpp");
  } else {
    native::singleCheckErrors(infos_vec_2[0], "lu_solve_dpcpp");
  }

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
  std::vector<int64_t> infos(native::batchCount(self), 0);

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
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "lu_solve_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "lu_solve_dpcpp");
  }
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
    Tensor& out,
    const Tensor& self,
    const Tensor& LU_data,
    const Tensor& LU_pivots) {
  Tensor out_tmp = at::AtenIpexTypeXPU::lu_solve(self, LU_data, LU_pivots);
  out.resize_as_(out_tmp).copy_(out_tmp);
  return out;
}

std::tuple<Tensor, Tensor> _solve_helper(const Tensor& self, const Tensor& A) {
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto A_working_copy = native::cloneBatchedColumnMajor(A);
  auto req_size = A.sizes().vec();
  req_size.pop_back();
  auto pivots_tensor = at::empty(req_size, A.options().dtype(kLong));
  std::vector<int64_t> infos(native::batchCount(self), 0);

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "solve_dpcpp", [&] {
    impl::apply_lu_dpcpp_<scalar_t>(A_working_copy, pivots_tensor, infos);
    impl::apply_lu_solve_dpcpp_<scalar_t>(
        self_working_copy,
        A_working_copy,
        pivots_tensor,
        infos,
        TransposeType::NoTranspose);
  });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "lu_solve_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "lu_solve_dpcpp");
  }
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}

std::tuple<Tensor&, Tensor&> solve_out(
    Tensor& solution,
    Tensor& lu,
    const Tensor& self,
    const Tensor& A) {
  native::checkSameDevice("solve", solution, self, "solution");
  native::checkSameDevice("solve", lu, self, "lu");
  native::checkLinalgCompatibleDtype("solve", solution, self, "solution");
  native::checkLinalgCompatibleDtype("solve", lu, self, "lu");

  Tensor solution_tmp, lu_tmp;
  std::tie(solution_tmp, lu_tmp) = at::AtenIpexTypeXPU::_solve_helper(self, A);
  solution.resize_as_(solution_tmp).copy_(solution_tmp);
  lu.resize_as_(lu_tmp).copy_(lu_tmp);
  return std::tuple<Tensor&, Tensor&>(solution, lu);
}

Tensor _inverse_helper(const Tensor& self) {
  std::vector<int64_t> infos(native::batchCount(self), 0);
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self.scalar_type(), "inverse_dpcpp", [&] {
        impl::apply_inverse_dpcpp_<scalar_t>(self_working_copy, infos);
      });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "inverse_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "inverse_dpcpp");
  }
  return self_working_copy;
}

Tensor inverse(const Tensor& self) {
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  native::squareCheckInputs(self);
  return at::AtenIpexTypeXPU::_inverse_helper(self);
}

Tensor& inverse_out(Tensor& out, const Tensor& self) {
  if (self.size(-1) == 0) {
    out.resize_as_(self);
    return out;
  }
  out.copy_(at::AtenIpexTypeXPU::inverse(self));
  return out;
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

  std::vector<int64_t> infos(native::batchCount(self), 0);
  Tensor self_working_copy = native::cloneBatchedColumnMajor(self);
  Tensor tau_working_copy = at::empty(req_size, self.options());

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "geqrf_dpcpp", [&] {
    impl::apply_geqrf_dpcpp_<scalar_t>(
        self_working_copy, tau_working_copy, m, n, infos);
  });
  return std::tuple<Tensor, Tensor>(self_working_copy, tau_working_copy);
}

std::tuple<Tensor&, Tensor&> geqrf_out(
    Tensor& a,
    Tensor& tau,
    const Tensor& self) {
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

Tensor linalg_householder_product(const Tensor& self, const Tensor& input2) {
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  std::vector<int64_t> infos(native::batchCount(self), 0);
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
    Tensor& out,
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
  std::vector<int64_t> infos(native::batchCount(self), 0);
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
    if (self.dim() > 2) {
      native::batchCheckErrors(infos, "svd_xpu");
    } else {
      native::singleCheckErrors(infos[0], "svd_xpu");
    }

    if (compute_uv) {
      if (some) {
        VT_working_copy = VT_working_copy.narrow(-1, 0, k);
      }
    } else {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  } else {
    U_working_copy.zero_();
    VT_working_copy.zero_();
  }
  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

std::tuple<Tensor, Tensor> _symeig_helper(
    const Tensor& self,
    bool eigenvectors,
    bool upper) {
  std::vector<int64_t> infos(native::batchCount(self), 0);

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

  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "symeig");
  } else {
    native::singleCheckErrors(infos[0], "symeig");
  }
  if (eigenvectors) {
    return std::tuple<Tensor, Tensor>(eigvals, self_working_copy);
  } else {
    return std::tuple<Tensor, Tensor>(eigvals, at::empty({0}, self.options()));
  }
}

std::tuple<Tensor, Tensor> _triangular_solve_helper(
    const Tensor& self,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular) {
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto A_working_copy = native::cloneBatchedColumnMajor(A);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self.scalar_type(), "triangular_solve_xpu", [&] {
        impl::apply_triangular_solve<scalar_t>(
            self_working_copy, A_working_copy, upper, transpose, unitriangular);
      });
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
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
      "b should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(
      A.dim() >= 2,
      "u should have at least 2 dimensions, but has ",
      A.dim(),
      " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) =
      native::_linalg_broadcast_batch_dims(self, A, "triangular_solve");
  return at::AtenIpexTypeXPU::_triangular_solve_helper(
      self_broadcasted, A_broadcasted, upper, transpose, unitriangular);
}

std::tuple<Tensor&, Tensor&> triangular_solve_out(
    Tensor& result,
    Tensor& clone_A,
    const Tensor& self,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular) {
  Tensor result_tmp, clone_A_tmp;
  std::tie(result_tmp, clone_A_tmp) =
      at::AtenIpexTypeXPU::_triangular_solve_helper(
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
  std::vector<int64_t> infos(native::batchCount(self), 0);
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "cholesky_solve_dpcpp", [&] {
    impl::apply_cholesky_solve_dpcpp_<scalar_t>(
        self_working_copy, input2_working_copy, upper, infos);
  });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "cholesky_solve_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "cholesky_solve_dpcpp");
  }
  return self_working_copy;
}

Tensor _cholesky_helper(const Tensor& self, bool upper) {
  std::vector<int64_t> infos(native::batchCount(self), 0);
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self.scalar_type(), "cholesky_dpcpp", [&] {
        impl::apply_cholesky_dpcpp<scalar_t>(self_working_copy, upper, infos);
      });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "cholesky_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "cholesky_dpcpp");
  }
  return self_working_copy;
}

Tensor cholesky(const Tensor& self, bool upper) {
  TORCH_CHECK(
      self.dim() == 2, "input must be 2-d matrix, input shape=", self.sizes());
  if (self.size(-1) == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  native::squareCheckInputs(self);
  auto raw_cholesky_output = at::AtenIpexTypeXPU::_cholesky_helper(self, upper);
  if (upper) {
    return raw_cholesky_output.triu_();
  } else {
    return raw_cholesky_output.tril_();
  }
}

Tensor& cholesky_out(Tensor& out, const Tensor& self, bool upper) {
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
  at::native::resize_output(values, values_tmp.sizes());
  at::native::resize_output(vectors, vectors_tmp.sizes());
  values.copy_(values_tmp);
  vectors.copy_(vectors_tmp);
  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor, Tensor> eig(const Tensor& self, bool eigenvectors) {
  // fall back to CPU
  // 1, mkl doesn't have GPU interface for GEEV routine. and Due to this lack of
  // uniqueness, different hardware and software may compute different
  // eigenvectors.
  // 2, we will try to dep on IPEX oneMKL package as long as if it supports CPU
  // device
  // 3, magma CPU is potential path, as well

  auto self_tmp = self.cpu();
  Tensor e_tmp = at::empty({0}, self_tmp.options());
  Tensor v_tmp = at::empty({0}, self_tmp.options());
  at::eig_out(e_tmp, v_tmp, self_tmp, eigenvectors);
  Tensor e = e_tmp.to(kXPU);
  Tensor v = v_tmp.to(kXPU);
  return std::tuple<Tensor, Tensor>(e, v);
}

Tensor linalg_solve(const Tensor& input, const Tensor& other) {
  return at::native::linalg_solve(input, other);
}

Tensor& linalg_solve_out(
    const Tensor& input,
    const Tensor& other,
    Tensor& result) {
  auto infos =
      at::empty({0}, input.options().dtype(kLong).device(DeviceType::CPU));
  result = linalg_solve_out_info(result, infos, input, other);

  // Now check MKL error codes
  bool vector_case = native::linalg_solve_is_vector_rhs(input, other);
  if (vector_case ? result.dim() > 1 : result.dim() > 2) {
    native::batchCheckErrors(infos.to(ScalarType::Int), "linalg_solve");
  } else {
    native::singleCheckErrors(infos.item<int64_t>(), "linalg_solve");
  }

  return result;
}

Tensor _det_lu_based_helper_backward_helper(
    const Tensor& det_grad,
    const Tensor& det,
    const Tensor& self,
    const Tensor& lu,
    const Tensor& pivs) {
  auto eps = at::native::_get_epsilon(c10::toValueType(self.scalar_type()));
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
  std::vector<int64_t> infos(native::batchCount(d), 0);

  // d is modified in-place and will contain the result
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      d.scalar_type(), "_det_lu_based_helper_backward_helper", [&] {
        impl::apply_lu_solve_dpcpp_<scalar_t>(d, lu_clone, pivs, infos, trans);
      });

  if (d.dim() > 2) {
    native::batchCheckErrors(infos, "_det_lu_based_helper_backward_helper");
  } else {
    native::singleCheckErrors(infos[0], "_det_lu_based_helper_backward_helper");
  }
  return d;
}

} // namespace AtenIpexTypeXPU
} // namespace at
