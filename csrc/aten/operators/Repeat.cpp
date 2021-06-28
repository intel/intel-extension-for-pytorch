#include <ATen/ATen.h>
#include <ATen/native/Repeat.h>

#include <runtime/DPCPPUtils.h>
#include <core/Memory.h>
#include "comm/Helpers.h"


using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

DPCPP_DEF_K1(ComputeDpcppKer);
static void repeat_interleave_dpcpp_kernel(
    int64_t* repeat_ptr,
    int64_t* cumsum_ptr,
    int64_t* result_ptr,
    int64_t size) {
  auto queue = dpcppGetCurrentQueue();
  int64_t rng, grng, tile_size;
  parallel_for_setup(size, tile_size, rng, grng);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto rep_data = repeat_ptr;
    auto cum_data = cumsum_ptr;
    auto res_data = result_ptr;

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto rep_ptr = rep_data;
      auto cum_ptr = cum_data;
      auto res_ptr = res_data;

      for (int64_t i = item.get_global_id(0); i < size;
           i += item.get_global_range()[0]) {
        int64_t end = cum_ptr[i];
        int64_t repeat = rep_ptr[i];
        int64_t start = end - repeat;
        for (int64_t j = start; j < end; j++) {
          res_ptr[j] = i;
        }
      }
    };
    // kick off kernel
    cgh.parallel_for<DPCPP_K(ComputeDpcppKer)>(
        DPCPP::nd_range<1>(DPCPP::range<1>(grng), DPCPP::range<1>(tile_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

// static void repeat_interleave_dpcpp(int64_t *repeat_ptr, int64_t *cumsum_ptr,
// int64_t *result_ptr, int64_t size) {
//   repeat_interleave_dpcpp_kernel(repeat_ptr, cumsum_ptr, result_ptr, size);
// }

} // namespace impl

Tensor repeat_interleave(const Tensor& repeat) {
  return repeat_interleave_common<impl::repeat_interleave_dpcpp_kernel>(repeat);
}

} // namespace AtenIpexTypeXPU
} // namespace at
