#include <ATen/ATen.h>
#include <ATen/native/Repeat.h>

#include <core/DPCPPUtils.h>
#include <core/Memory.h>

using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
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
    auto rep_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, repeat_ptr);
    auto cum_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, cumsum_ptr);
    auto res_acc = DPCPPAccessor<dpcpp_rw_mode>(cgh, result_ptr);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto rep_ptr = rep_acc.template get_pointer<int64_t>();
      auto cum_ptr = cum_acc.template get_pointer<int64_t>();
      auto res_ptr = res_acc.template get_pointer<int64_t>();

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

} // impl

Tensor repeat_interleave(const Tensor& repeat) {
  return repeat_interleave_common<impl::repeat_interleave_dpcpp_kernel>(repeat);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
