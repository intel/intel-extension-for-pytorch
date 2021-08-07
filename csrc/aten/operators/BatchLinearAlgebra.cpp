#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <runtime/Utils.h>

#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <utils/oneMKLUtils.h>
#endif

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

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
  IndexType stride0 = (IndexType)self.size(-2);
  IndexType stride1 = (IndexType)self.size(-1);

  scalar_t* result_ptr = (scalar_t*)(result.data_ptr());
  scalar_t* self_ptr = (scalar_t*)(self.data_ptr());

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      for (size_t linearIndex = item.get_global_id(0); linearIndex < (size_t)N;
           linearIndex += item.get_global_range()[0]) {
        IndexType row, col;

        row = (linearIndex % (stride0 * stride1)) / stride1;
        col = (linearIndex % (stride0 * stride1)) % stride1;

        bool mask = upper ? (col - row >= k) : (col - row <= k);
        result_ptr[linearIndex] = mask ? self_ptr[linearIndex] : scalar_t(0);
      }
    };

    // kick off kernel
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <bool upper>
Tensor& triu_tril_dpcpp_template(
    Tensor& result,
    const Tensor& self,
    int64_t k,
    const char* name) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      name,
      [&] {
        if (xpu::dpcpp::detail::canUse32BitIndexMath(self)) {
          apply_triu_tril<scalar_t, int32_t, upper>(result, self, k);
        } else {
          apply_triu_tril<scalar_t, int64_t, upper>(result, self, k);
        }
      });

  return result;
}

Tensor& tril_dpcpp_out(Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }

  return triu_tril_dpcpp_template<false>(result, self, k, "tril");
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
  return triu_tril_dpcpp_template<true>(result, self, k, "triu");
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
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t batch_size = native::batchCount(self_);
  int64_t m = self_.size(-2);
  int64_t n = self_.size(-1);
  int64_t lda = m;
  int64_t stride_a = lda * n;
  int64_t stride_ipiv = (m < n) ? m : n;
  scalar_t* a = (scalar_t*)(self_.data_ptr());
  int64_t* ipiv = (int64_t*)(pivots_.data_ptr());
  int64_t scratchpadsize =
      oneapi::mkl::lapack::getrf_batch_scratchpad_size<scalar_t>(
          dpcpp_queue, m, n, lda, stride_a, stride_ipiv, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::getrf_batch,
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
    Tensor& b_,
    Tensor& lu_,
    Tensor& pivots_,
    std::vector<int64_t>& infos_) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t batch_size = native::batchCount(b_);
  int64_t group_count = (batch_size + local_size - 1) / local_size;
  int64_t* group_sizes = new int64_t[group_count];
  for (auto i = 0; i < group_count; i++)
    group_sizes[i] = std::min(local_size, batch_size - i * local_size);

  std::vector<oneapi::mkl::transpose> trans(
      group_count, oneapi::mkl::transpose::nontrans);
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

  int64_t scratchpadsize =
      oneapi::mkl::lapack::getrs_batch_scratchpad_size<scalar_t>(
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
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::getrs_batch,
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

  int64_t scratchpadsize =
      oneapi::mkl::lapack::getri_batch_scratchpad_size<scalar_t>(
          dpcpp_queue, n.data(), lda.data(), group_count, group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::getri_batch,
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

  int64_t scratchpadsize =
      oneapi::mkl::lapack::geqrf_batch_scratchpad_size<scalar_t>(
          dpcpp_queue,
          m.data(),
          n.data(),
          lda.data(),
          group_count,
          group_sizes);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::geqrf_batch,
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

template <typename scalar_t>
static void apply_svd(
    Tensor& self,
    Tensor& U,
    Tensor& S,
    Tensor& VT,
    char jobz,
    std::vector<int64_t>& infos) {
#ifdef USE_ONEMKL
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  scalar_t* self_data = (scalar_t*)(self.data_ptr());
  scalar_t* U_data = (scalar_t*)U.data_ptr();
  value_t* S_data = (value_t*)S.data_ptr();
  scalar_t* VT_data = (scalar_t*)VT.data_ptr();
  auto m = self.size(-2);
  auto n = self.size(-1);
  std::int64_t lda = m;
  std::int64_t ldu = m;
  std::int64_t ldvt = n;
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
  Tensor scratchpad_at = at::empty({scratchpadsize}, self.options());
  DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::lapack::gesvd,
      dpcpp_queue,
      jobu,
      jobvt,
      m,
      n,
      (scalar_t*)(self.data_ptr()),
      lda,
      S_data,
      U_data,
      ldu,
      VT_data,
      ldvt,
      (scalar_t*)(scratchpad_at.data_ptr()),
      scratchpadsize);
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
  auto jobz = eigenvectors ? oneapi::mkl::job::vec : oneapi::mkl::job::novec;
  auto uplo = upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::syevd_scratchpad_size<scalar_t>(
          dpcpp_queue, jobz, uplo, n, n);

  Tensor scratchpad_at = at::empty({scratchpadsize}, self.options());
  DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::lapack::syevd,
      dpcpp_queue,
      jobz,
      uplo,
      n,
      (scalar_t*)(self.data_ptr()),
      n,
      (scalar_t*)(eigvals.data_ptr()),
      (scalar_t*)(scratchpad_at.data_ptr()),
      scratchpadsize);
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

} // namespace impl

Tensor& triu_out(Tensor& out, const Tensor& self, int64_t diagonal) {
  impl::triu_dpcpp_out(out, self, diagonal);
  return out;
}

Tensor& tril_out(Tensor& out, const Tensor& self, int64_t diagonal) {
  impl::tril_dpcpp_out(out, self, diagonal);
  return out;
}

Tensor& tril_(Tensor& self, int64_t diagonal) {
  return at::AtenIpexTypeXPU::tril_out(self, self, diagonal);
}

Tensor& triu_(Tensor& self, int64_t diagonal) {
  return at::AtenIpexTypeXPU::triu_out(self, self, diagonal);
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
  return std::make_tuple(self_working_copy, pivots_tensor, infos_tensor);
}

Tensor _lu_solve_helper(
    const Tensor& self,
    const Tensor& LU_data,
    const Tensor& LU_pivots) {
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto LU_data_working_copy = native::cloneBatchedColumnMajor(LU_data);
  auto LU_pivots_working_copy =
      LU_pivots.is_contiguous() ? LU_pivots : LU_pivots.contiguous();
  std::vector<int64_t> infos(native::batchCount(self), 0);

  if (self.numel() == 0 || LU_data.numel() == 0) {
    return at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lu_solve_dpcpp", [&] {
    impl::apply_lu_solve_dpcpp_<scalar_t>(
        self_working_copy, LU_data_working_copy, LU_pivots_working_copy, infos);
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

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "solve_dpcpp", [&] {
    impl::apply_lu_dpcpp_<scalar_t>(A_working_copy, pivots_tensor, infos);
    impl::apply_lu_solve_dpcpp_<scalar_t>(
        self_working_copy, A_working_copy, pivots_tensor, infos);
  });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "lu_solve_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "lu_solve_dpcpp");
  }
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}

std::tuple<Tensor, Tensor> solve(const Tensor& self, const Tensor& A) {
  TORCH_CHECK(
      self.dim() >= 2,
      "B should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(
      A.dim() >= 2,
      "A should have at least 2 dimensions, but has ",
      A.dim(),
      " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) =
      native::_linalg_broadcast_batch_dims(self, A, "solve_dpcpp");
  return at::AtenIpexTypeXPU::_solve_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&, Tensor&> solve_out(
    Tensor& solution,
    Tensor& lu,
    const Tensor& self,
    const Tensor& A) {
  Tensor solution_tmp, lu_tmp;
  std::tie(solution_tmp, lu_tmp) = at::AtenIpexTypeXPU::_solve_helper(self, A);
  solution.resize_as_(solution_tmp).copy_(solution_tmp);
  lu.resize_as_(lu_tmp).copy_(lu_tmp);
  return std::tuple<Tensor&, Tensor&>(solution, lu);
}

Tensor _inverse_helper(const Tensor& self) {
  std::vector<int64_t> infos(native::batchCount(self), 0);
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "inverse_dpcpp", [&] {
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
    return out.resize_as_(self);
  }
  out.copy_(at::AtenIpexTypeXPU::inverse(self));
  return out;
}

std::tuple<Tensor, Tensor> _qr_helper(const Tensor& self, bool some) {
  std::vector<int64_t> infos(native::batchCount(self), 0);
  int64_t m = self.size(-2), n = self.size(-1);

  // Prepare inputs for geqrf
  auto self_sizes = self.sizes().vec();
  self_sizes.pop_back();
  self_sizes[self.dim() - 2] = std::min(m, n);
  auto tau_working_copy = at::empty(self_sizes, self.options());
  Tensor q_working_copy;

  std::vector<int64_t> q_sizes, q_strides;
  int64_t n_columns_q;
  Tensor R;
  std::tie(q_sizes, q_strides, n_columns_q) =
      native::_compute_geometry_for_Q(self, some);

  if (self.numel() == 0) {
    q_sizes[self.dim() - 1] = n_columns_q;
    q_working_copy = at::eye(
        q_sizes[self.dim() - 2], q_sizes[self.dim() - 1], self.options());
    q_working_copy = q_working_copy.expand_as(q_working_copy);

    q_sizes[self.dim() - 1] = n;
    q_sizes[self.dim() - 2] = n_columns_q;
    R = at::empty(q_sizes, self.options());
    return std::make_tuple(q_working_copy, R);
  }

  q_working_copy = at::empty_strided(q_sizes, q_strides, self.options());
  q_working_copy.narrow(-1, 0, n).copy_(self);

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "qr_dpcpp", [&] {
    impl::apply_geqrf_dpcpp_<scalar_t>(
        q_working_copy, tau_working_copy, m, n, infos);
  });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "qr_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "qr_dpcpp");
  }
  R = q_working_copy.slice(-2, 0, n_columns_q)
          .slice(-1, 0, n)
          .contiguous()
          .triu();

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "qr_dpcpp", [&] {
    impl::apply_orgqr_dpcpp_<scalar_t>(
        q_working_copy,
        tau_working_copy,
        m,
        n_columns_q,
        std::min(m, n),
        infos);
  });
  if (self.dim() > 2) {
    native::batchCheckErrors(infos, "qr_dpcpp");
  } else {
    native::singleCheckErrors(infos[0], "qr_dpcpp");
  }
  return std::make_tuple(q_working_copy.narrow(-1, 0, n_columns_q), R);
}

std::tuple<Tensor, Tensor> qr(const Tensor& self, bool some) {
  TORCH_CHECK(
      self.dim() >= 2,
      "self should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  return at::AtenIpexTypeXPU::_qr_helper(self, some);
}

std::tuple<Tensor&, Tensor&> qr_out(
    Tensor& Q,
    Tensor& R,
    const Tensor& self,
    bool some) {
  TORCH_CHECK(
      self.dim() >= 2,
      "self should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  Tensor Q_tmp, R_tmp;
  std::tie(Q_tmp, R_tmp) = at::AtenIpexTypeXPU::_qr_helper(self, some);
  Q.resize_as_(Q_tmp).copy_(Q_tmp);
  R.resize_as_(R_tmp).copy_(R_tmp);
  return std::tuple<Tensor&, Tensor&>(Q, R);
}

std::tuple<Tensor, Tensor> geqrf(const Tensor& self) {
  TORCH_CHECK(
      self.dim() >= 2,
      "input should have at least 2 dimensions. but has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(self.numel() != 0, "input must not be empty");

  std::vector<int64_t> infos(native::batchCount(self), 0);
  int64_t m = self.size(-2), n = self.size(-1);
  Tensor self_working_copy = native::cloneBatchedColumnMajor(self);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  req_size[self.dim() - 2] = std::min(m, n);
  Tensor tau_working_copy = at::empty(req_size, self.options());

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "geqrf_dpcpp", [&] {
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

Tensor orgqr(const Tensor& self, const Tensor& input2) {
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  std::vector<int64_t> infos(native::batchCount(self), 0);
  int64_t m = self.size(-2), n_columns_q = self.size(-1), n = input2.size(-1);
  auto q_working_copy = native::cloneBatchedColumnMajor(self);

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "orgqr_dpcpp", [&] {
    impl::apply_orgqr_dpcpp_<scalar_t>(
        q_working_copy, input2, m, n_columns_q, std::min(m, n), infos);
  });
  return q_working_copy;
}

Tensor& orgqr_out(Tensor& out, const Tensor& self, const Tensor& input2) {
  if (self.size(-1) == 0) {
    return out.resize_as_(self);
  }
  out.copy_(at::AtenIpexTypeXPU::orgqr(self, input2));
  return out;
}

Tensor ormqr(
    const Tensor& self,
    const Tensor& input2,
    const Tensor& input3,
    bool left,
    bool transpose) {
  TORCH_CHECK(
      self.dim() == 2, "input must be 2-d matrix, input shape=", self.sizes());
  TORCH_CHECK(
      input3.dim() == 2, "c must be 2-d matrix, c shape=", input3.sizes());
  int64_t infos = 0;
  // int64_t m = self.size(-2), n = self.size(-1), k = input2.size(-1);
  int64_t m = input3.size(0), n = input3.size(1), k = input2.size(-1);
  auto c_working_copy = native::cloneBatchedColumnMajor(input3);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "ormqr_dpcpp", [&] {
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
  if (self.size(-1) == 0) {
    return out.resize_as_(input3);
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
      native::_create_U_S_VT(self, some, compute_uv);

  if (self.numel() > 0) {
    auto self_working_copy = native::cloneBatchedColumnMajor(self);

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "svd_xpu", [&] {
      impl::apply_svd<scalar_t>(
          self_working_copy,
          U_working_copy,
          S_working_copy,
          VT_working_copy,
          jobz,
          infos);
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

std::tuple<Tensor, Tensor, Tensor> svd(
    const Tensor& self,
    bool some,
    bool compute_uv) {
  TORCH_CHECK(
      self.dim() >= 2,
      "self should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  return at::_svd_helper(self, some, compute_uv);
}

std::tuple<Tensor&, Tensor&, Tensor&> svd_out(
    Tensor& U,
    Tensor& S,
    Tensor& VT,
    const Tensor& self,
    bool some,
    bool compute_uv) {
  TORCH_CHECK(
      self.dim() >= 2,
      "self should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  Tensor U_tmp, S_tmp, VT_tmp;
  std::tie(U_tmp, S_tmp, VT_tmp) = at::_svd_helper(self, some, compute_uv);
  U.resize_as_(U_tmp).copy_(U_tmp);
  S.resize_as_(S_tmp).copy_(S_tmp);
  VT.resize_as_(VT_tmp).copy_(VT_tmp);
  return std::tuple<Tensor&, Tensor&, Tensor&>(U, S, VT);
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
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "symeig", [&] {
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
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "triangular_solve_cpu", [&] {
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
  return at::_triangular_solve_helper(
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
      at::_triangular_solve_helper(self, A, upper, transpose, unitriangular);
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
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "cholesky_solve_dpcpp", [&] {
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

Tensor cholesky_solve(const Tensor& self, const Tensor& input2, bool upper) {
  TORCH_CHECK(
      self.dim() >= 2,
      "b should have at least 2 dimensions, but has ",
      self.dim(),
      " dimensions instead");
  TORCH_CHECK(
      input2.dim() >= 2,
      "u should have at least 2 dimensions, but has ",
      input2.dim(),
      " dimensions instead");
  Tensor self_broadcasted, input2_broadcasted;
  std::tie(self_broadcasted, input2_broadcasted) =
      native::_linalg_broadcast_batch_dims(
          self, input2, "cholesky_solve_dpcpp");
  return at::AtenIpexTypeXPU::_cholesky_solve_helper(
      self_broadcasted, input2_broadcasted, upper);
}

Tensor& cholesky_solve_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& input2,
    bool upper) {
  Tensor out_tmp = at::AtenIpexTypeXPU::cholesky_solve(self, input2, upper);
  out.resize_as_(out_tmp).copy_(out_tmp);
  return out;
}

Tensor _cholesky_helper(const Tensor& self, bool upper) {
  std::vector<int64_t> infos(native::batchCount(self), 0);
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "cholesky_dpcpp", [&] {
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
    return raw_cholesky_output.tril_();
  } else {
    return raw_cholesky_output.triu_();
  }
}

Tensor& cholesky_out(Tensor& out, const Tensor& self, bool upper) {
  Tensor out_tmp = at::AtenIpexTypeXPU::cholesky(self, upper);
  out.resize_as_(out_tmp).copy_(out_tmp);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
