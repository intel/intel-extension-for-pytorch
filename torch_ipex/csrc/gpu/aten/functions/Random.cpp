#include <ATen/ATen.h>

#include <core/DPCPP.h>
#include <core/Stream.h>
#include <core/Memory.h>
#include <core/DPCPPUtils.h>
#include <core/Generator.h>
#include "Random.h"


using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
void uniform(Tensor & self, Generator *_generator, double a, double b) {
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);

  auto queue = dpcppGetCurrentQueue();

  auto cgf = DP_Q_CGF(cgh) {
    auto self_data_size = self.nbytes();
    void *self_data_ptr = self.data_ptr<scalar_t>();

    auto acc = DPCPPAccessor<dp_w_mode>(
        cgh, self_data_ptr, self_data_size).get_access();

    int64_t tile_size, range, global_range;
    parallel_for_setup(self.numel(), tile_size, range, global_range);
    auto num_work_items = DP::nd_range<1>(DP::range<1>(
        global_range), DP::range<1>(tile_size));

    if (std::is_same<scalar_t, float>::value) {
      FloatRandomFiller uniform_rnd_filler(acc, range, gen->current_seed(), a, b);
      cgh.parallel_for(num_work_items, uniform_rnd_filler);
    } else if (std::is_same<scalar_t, at::Half>::value) {
      HalfRandomFiller uniform_rnd_filler(acc, range, gen->current_seed(), a, b);
      cgh.parallel_for(num_work_items, uniform_rnd_filler);
    } else {
      DoubleRandomFiller uniform_rnd_filler(acc, range, gen->current_seed(), a, b);
      cgh.parallel_for(num_work_items, uniform_rnd_filler);
    }
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
  queue.wait_and_throw();
}

template <typename scalar_t, typename accreal>
void normal(Tensor & self, double mean, double stdv, Generator *_generator) {
  TORCH_CHECK(stdv > 0, "standard deviation must be strictly positive");

  // Generate uniform number
  uniform<scalar_t>(self, _generator, 0.0, 1.0);

  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);

  auto queue = dpcppGetCurrentQueue();

  auto cgf = DP_Q_CGF(cgh) {
    auto self_data_size = self.nbytes();
    void *self_data_ptr = self.data_ptr<scalar_t>();

    auto acc = DPCPPAccessor<dp_rw_mode>(
        cgh, self_data_ptr, self_data_size).get_access();
    int64_t tile_size, range, global_range;

    bool recompute = ((self.numel() % 2) != 0);
    int64_t compute_num = recompute ? (self.numel() / 2 + 1) : (self.numel() / 2); // We will generate two normal element per time
    parallel_for_setup(compute_num, tile_size, range, global_range);
    auto num_work_items = DP::nd_range<1>(DP::range<1>(global_range), DP::range<1>(tile_size));

    NormalRandomFiller<accreal> normal_rnd_filler(acc, compute_num, stdv, mean);
    cgh.parallel_for(num_work_items, normal_rnd_filler);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
  queue.wait_and_throw();
}

} // namespace impl

Tensor & normal_(Tensor & self, double mean, double std, Generator * generator) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "normal_", [&]() {
    using accreal = typename std::conditional<
        std::is_same<scalar_t, at::Half>::value, float, scalar_t>::type;
    impl::normal<scalar_t, accreal>(self, mean, std, generator);
  });
  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
