#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <core/ApplyUtils.h>
#include <core/Context.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#ifdef USE_ONEMKL
#include <mkl_sycl.hpp>
#include <mkl.h>
#endif

using namespace at::dpcpp;

//DPCPP_DEF_K2(triuTrilSycl, typename scalar_t, typename IndexType, bool upper);

template <typename...> class CopyTriangleSymetric {};

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
void copy_triangle_symmetric_template(Tensor& self, bool upper) {
  auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto row_stride = self.stride(0);
  auto column_stride = self.stride(1);
  auto n = self.size(0);

  size_t work_item_num = n * (n - 1) / 2; // only start the triangle element

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto data_acc = DPCPPAccessor<dpcpp_rw_mode>(__cgh, self.data_ptr());

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto data_ptr = data_acc.template get_pointer<scalar_t>();
      auto linear_id = item_id.get_linear_id();
      float triangle_row_ = (Numerics<float>::sqrt(1 + 8.0 * linear_id) - 1 ) / 2;
      int64_t triangle_row = triangle_row_;
      int64_t triangle_col = linear_id - (triangle_row*(triangle_row+1))/2 ;
      int64_t r;
      int64_t c;

      if (upper) {
        r = triangle_col;
        c = triangle_row + 1;
      }
      else {
        r = triangle_row + 1;
        c = triangle_col;
      }

      auto src_off = r*row_stride + c*column_stride;
      auto dst_off = c*row_stride + r*column_stride;
      data_ptr[dst_off] = data_ptr[src_off];
    };

    __cgh.parallel_for<CopyTriangleSymetric<scalar_t>>(
      DPCPP::range</*dim=*/1>(work_item_num), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}


} // namespace impl

Tensor & cholesky_inverse_out(Tensor & out, const Tensor & self, bool upper) {
#ifdef USE_ONEMKL
  TORCH_CHECK(self.dim() == 2, "input must be 2-d matrix. input shape=", self.sizes());
  TORCH_CHECK(self.size(0) == self.size(1), "input should be square. input shape=", self.sizes());

  int64_t n = self.size(0);
  int64_t lda = n;
  if (n == 0)
    return out;

  out = native::cloneBatchedColumnMajor(self);

  IPEX_DISPATCH_FLOATING_TYPES(out.scalar_type(), "potri_dpcpp_out", [&] {
    auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
    Tensor info = at::empty({1}, out.options().dtype(kLong));
    auto a = make_buffer<scalar_t>(out.data_ptr());
    auto info_ = make_buffer<int64_t>(info.data_ptr());
    auto upper_lower = upper ? (mkl::uplo::upper) : (mkl::uplo::lower);
    mkl::lapack::potri(dpcpp_queue, upper_lower, n, a, lda, info_);
    native::singleCheckErrors(info.item<int64_t>(), "potri_dpcpp");
    impl::copy_triangle_symmetric_template<scalar_t>(out, upper);
  });

  return out;
#else
  AT_ERROR("potri dpcpp: oneMKL library not found in compilation");
#endif
}

Tensor cholesky_inverse(const Tensor & self, bool upper) {
  TORCH_CHECK(self.dim() == 2, "input must be 2-d matrix. input shape=", self.sizes());
  TORCH_CHECK(self.size(0) == self.size(1), "input should be square. input shape=", self.sizes());
  Tensor out;
  return AtenIpexTypeDPCPP::cholesky_inverse_out(out, self, upper);
}

std::tuple<Tensor&, Tensor&> geqrf_out(Tensor &ra, Tensor &tau, const Tensor &a) {
#ifdef USE_ONEMKL
  TORCH_CHECK(a.dim() == 2, "input must be 2-d matrix. input shape=", a.sizes());
  TORCH_CHECK(a.numel() != 0, "input must not be empty");
  auto m = a.size(0); // rows of matrix
  auto n = a.size(1); // columns of matrix

  ra = native::cloneBatchedColumnMajor(a);
  tau.resize_(std::min(m, n));

  IPEX_DISPATCH_FLOATING_TYPES(a.scalar_type(), "geqrf_out", [&] {
    auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

    // create work bufer
    std::int64_t lwork;
    mkl::lapack::geqrf_get_lwork<scalar_t>(dpcpp_queue.get_device(), m, n, m, lwork);
    Tensor work = at::empty({lwork}, a.options());

    Tensor info = at::empty({1}, ra.options().dtype(kLong));

    auto a_ = make_buffer<scalar_t>(ra.data_ptr());
    auto tau_ = make_buffer<scalar_t>(tau.data_ptr());
    auto work_ = make_buffer<scalar_t>(work.data_ptr());
    auto info_ = make_buffer<int64_t>(info.data_ptr());
    mkl::lapack::geqrf(dpcpp_queue, m, n, a_, m, tau_, work_, lwork, info_);
    native::singleCheckErrors(info.item<int64_t>(), "geqrf_out");
  });

  return std::tuple<Tensor&, Tensor&>(ra, tau);
#else
  AT_ERROR("geqrf: oneMKL library not found in compilation");
#endif
}

std::tuple<Tensor,Tensor> geqrf(const Tensor &a) {
  TORCH_CHECK(a.dim() == 2, "input must be 2-d matrix. input shape=", a.sizes());
  TORCH_CHECK(a.numel() != 0, "input must not be empty");
  auto m = a.size(0); // rows of matrix
  auto n = a.size(1); // columns of matrix
  Tensor ra;
  Tensor rtau = at::empty({std::min(m, n)}, a.options());
  AtenIpexTypeDPCPP::geqrf_out(ra, rtau, a);
  return std::tuple<Tensor, Tensor>(ra, rtau);
}

Tensor& ger_out(Tensor & out, const Tensor & self, const Tensor & vec2) {
#ifdef USE_ONEMKL
  TORCH_CHECK(self.dim() == 1, "input must be 1-d vector. input shape=", self.sizes());
  TORCH_CHECK(vec2.dim() == 1, "vec2 must be 1-d vector. vec2 shape=", vec2.sizes());

  int64_t n = self.size(0); // rows of matrix
  int64_t m = vec2.size(0); // columns of matrix
  if (m == 0 || n == 0)
    return out;
  int64_t input_stride = self.stride(0);
  int64_t vec2_stride = vec2.stride(0);

  out.resize_({n, m}).zero_();
  TORCH_CHECK(out.is_contiguous(), "the out is not contiguous");

  IPEX_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ger_out", [&] {
    auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
    auto x = make_buffer<scalar_t>(self.data_ptr());
    auto y = make_buffer<scalar_t>(vec2.data_ptr());
    auto a = make_buffer<scalar_t>(out.data_ptr());
    // The BLAS API is column major. To save the transpose and element move, we switch the two input.
    // The ger documents https://spec.oneapi.com/versions/0.6.0/oneMKL/GUID-BD2E87B3-5FA7-4E0C-88E2-1982AB0773A2.html
    mkl::blas::ger(dpcpp_queue, m, n, (float)1.0, y, vec2_stride, x, input_stride, a, m);
  });

  return out;
#else
  AT_ERROR("ger: oneMKL library not found in compilation");
#endif
}

Tensor ger(const Tensor & self, const Tensor & vec2) {
  TORCH_CHECK(self.dim() == 1, "input must be 1-d vector. input shape=", self.sizes());
  TORCH_CHECK(vec2.dim() == 1, "vec2 must be 1-d vector. vec2 shape=", vec2.sizes());
  auto n = self.size(0); // rows of matrix
  auto m = vec2.size(0); // columns of matrix
  Tensor out = at::zeros({n, m}, self.options());
  return AtenIpexTypeDPCPP::ger_out(out, self, vec2);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
