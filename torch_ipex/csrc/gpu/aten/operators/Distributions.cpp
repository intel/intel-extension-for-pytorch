#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "Random.h"
#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Generator.h>
#include <core/Memory.h>

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t> class bernoulli_tensor_dpcpp_ker {};

template <typename scalar_t> class bernoulli_scalar_dpcpp_ker {};

template <typename scalar_t>
void bernoulli_scalar_kernel(Tensor &ret, double p_, uint64_t seed) {
  auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto size = ret.numel() * sizeof(scalar_t);
  // First fill self with random number within range [0.0 - 1.0]
  dpcpp_queue.submit([&](DPCPP::handler &cgh) {
    auto acc = DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size)
                   .get_access();
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = DPCPP::nd_range<1>(DPCPP::range<1>(global_range),
                                             DPCPP::range<1>(tile_size));
    FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  });

  // Generate final bernoulli distributions
  dpcpp_queue.submit([&](DPCPP::handler &cgh) {
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto out_acc =
        DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size);
    cgh.parallel_for<bernoulli_scalar_dpcpp_ker<scalar_t>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(global_range),
                           DPCPP::range<1>(tile_size)),
        [=](DPCPP::nd_item<1> item) {
          int64_t id = item.get_global_linear_id();
          auto out_ptr = out_acc.template get_pointer<scalar_t>();
          if (id < range) {
            out_ptr[id] = out_ptr[id] < p_ ? 1.0f : 0.0f;
          }
        });
  });
}

template <typename scalar_t>
void bernoulli_tensor_kernel(Tensor &ret, const Tensor &p, uint64_t seed) {
  auto &queue = getCurrentDPCPPStream().dpcpp_queue();
  auto size = ret.numel() * sizeof(scalar_t);

  // First fill self with random number within range [0.0 - 1.0]
  auto cgf1 = DPCPP_Q_CGF(cgh) {
    auto acc = DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size)
                   .get_access();
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = DPCPP::nd_range<1>(DPCPP::range<1>(global_range),
                                             DPCPP::range<1>(tile_size));
    FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf1);

  // Generate final bernoulli distributions
  auto cgf2 = DPCPP_Q_CGF(cgh) {
    auto in_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, p.data_ptr<scalar_t>(), size);
    auto out_acc =
        DPCPPAccessor<dpcpp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size);
    int64_t tile_size, range, global_range;
    parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      int64_t id = item.get_global_linear_id();
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      if (id < range) {
        out_ptr[id] = out_ptr[id] < in_ptr[id] ? 1.0f : 0.0f;
      }
    };

    cgh.parallel_for<bernoulli_tensor_dpcpp_ker<scalar_t>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(global_range),
                           DPCPP::range<1>(tile_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf2);
}

} // namespace impl

Tensor &bernoulli_(Tensor &self, const Tensor &p_, Generator *_generator) {
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // Call dpcpp kernel to generate bernoulli distribution
  AT_DISPATCH_FLOATING_TYPES(p_.scalar_type(), "bernoulli_tensor_kernel", [&] {
    impl::bernoulli_tensor_kernel<scalar_t>(self, p_, gen->current_seed());
  });
  return self;
}

Tensor &bernoulli_(Tensor &self, double p, Generator *_generator) {
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // Call dpcpp kernel to generate bernoulli distribution
  AT_DISPATCH_FLOATING_TYPES(
      self.scalar_type(), "bernoulli_scalar_kernel", [&] {
        impl::bernoulli_scalar_kernel<scalar_t>(self, p, gen->current_seed());
      });
  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
