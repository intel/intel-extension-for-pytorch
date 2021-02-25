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

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
Tensor mv(const Tensor & self, const Tensor & vec) {
#ifdef USE_ONEMKL
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto lda = n;

  Tensor out = at::empty({m}, vec.options());
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();

  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "mv", [&] {
    oneapi::mkl::blas::row_major::gemv(dpcpp_queue, oneapi::mkl::transpose::N, m, n, (scalar_t)1.0, (scalar_t *)self.data_ptr(), lda, (scalar_t *)vec.data_ptr(), 1, (scalar_t)0, (scalar_t *)out.data_ptr(), 1);
  });

  return out;
#else
  AT_ERROR("mv: oneMKL library not found in compilation");
#endif
}

} // namespace AtenIpexTypeXPU
} // namespace at
