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


} // namespace AtenIpexTypeDPCPP
} // namespace at
