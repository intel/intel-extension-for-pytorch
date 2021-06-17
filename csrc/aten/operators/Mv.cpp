#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/ATDispatch.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <utils/oneMKLUtils.h>
#endif


using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

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
    DPCPP_ONEMKL_SUBMIT(
      dpcpp_queue,
      oneapi::mkl::blas::row_major::gemv, dpcpp_queue, oneapi::mkl::transpose::N, m, n, (scalar_t)1.0, (scalar_t *)self.data_ptr(), lda, (scalar_t *)vec.data_ptr(), 1, (scalar_t)0, (scalar_t *)out.data_ptr(), 1);
  });

  return out;
#else
  AT_ERROR("mv: oneMKL library not found in compilation");
#endif
}

} // namespace AtenIpexTypeXPU
} // namespace at
