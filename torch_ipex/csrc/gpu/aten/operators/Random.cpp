#include <ATen/ATen.h>

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include "Random.h"
#include <utils/ATDispatch.h>

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DPCPP_DEF_K1(uniform_random_filler);

template <typename scalar_t>
void uniform(Tensor& self, Generator* _generator, double a, double b) {
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);

  auto queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_data_size = self.nbytes();
    auto data = get_buffer<dpcpp_w_mode>(cgh, self.data_ptr<scalar_t>());

    int64_t tile_size, range, global_range;
    parallel_for_setup(self.numel(), tile_size, range, global_range);
    auto num_work_items = DPCPP::nd_range<1>(
        DPCPP::range<1>(global_range), DPCPP::range<1>(tile_size));
    auto current_seed = gen->current_seed();

    cgh.parallel_for<DPCPP_K(uniform_random_filler, scalar_t)>(
        num_work_items, [=](DPCPP::nd_item<1> item) {
          auto ptr = get_pointer(data);
          RandomEngine<scalar_t> uniform_rnd_filler(ptr, range, current_seed, a, b);
          uniform_rnd_filler(item);
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

DPCPP_DEF_K1(normal_random_filler);

template <typename scalar_t, typename accreal>
void normal(Tensor& self, double mean, double stdv, Generator* _generator) {
  TORCH_CHECK(stdv > 0, "standard deviation must be strictly positive");

  // Generate uniform number
  uniform<scalar_t>(self, _generator, 0.0, 1.0);

  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      _generator, getDefaultDPCPPGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);

  auto queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_data_size = self.nbytes();

    auto data = get_buffer<dpcpp_rw_mode>(cgh, self.data_ptr<scalar_t>());
    int64_t tile_size, range, global_range;

    bool recompute = ((self.numel() % 2) != 0);
    int64_t compute_num = recompute
        ? (self.numel() / 2 + 1)
        : (self.numel() / 2); // We will generate two normal element per time
    parallel_for_setup(compute_num, tile_size, range, global_range);
    auto num_work_items = DPCPP::nd_range<1>(
        DPCPP::range<1>(global_range), DPCPP::range<1>(tile_size));
    cgh.parallel_for<DPCPP_K(normal_random_filler, accreal, scalar_t)>(
        num_work_items, [=](DPCPP::nd_item<1> item) {
          auto ptr = get_pointer(data);
          NormalRandomFiller<scalar_t, accreal> normal_rnd_filler(
            ptr, compute_num, stdv, mean);
          normal_rnd_filler(item);
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

} // namespace impl

Tensor& normal_(Tensor& self, double mean, double std, Generator* generator) {
  if (self.numel() != 0) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "normal_",
        [&]() {
          using accreal = typename std::conditional<std::is_same<scalar_t, at::Half>::value ||
              std::is_same<scalar_t, at::BFloat16>::value, float, scalar_t>::type;
          impl::normal<scalar_t, accreal>(self, mean, std, generator);
    });
  }
  return self;
}

Tensor& uniform_(Tensor& self, double from, double to, Generator* generator) {
  if (self.numel() != 0) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      self.scalar_type(), "uniform_", [&]() {
        impl::uniform<scalar_t>(self, generator, from, to);
      });
  }
  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
