#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/Memory.h>
#include <core/Utils.h>
#include <core/Context.h>
#include <core/Generator.h>
#include "Random.h"


using namespace at::sycl::detail;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
class bernoulli_tensor_sycl_ker {};

template <typename scalar_t>
class bernoulli_scalar_sycl_ker {};

template <typename scalar_t>
void bernoulli_scalar_kernel(Tensor &ret, double p_, uint64_t seed) {
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  auto size = ret.numel() * sizeof(scalar_t);
  // First fill self with random number within range [0.0 - 1.0]
  sycl_queue.submit([&](DP::handler &cgh){
    auto acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size).get_access();
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = DP::nd_range<1>(DP::range<1>(global_range), DP::range<1>(tile_size));
    FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  });

  // Generate final bernoulli distributions
  sycl_queue.submit([&](DP::handler& cgh) {
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto out_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size);
    cgh.parallel_for<bernoulli_scalar_sycl_ker<scalar_t>>(DP::nd_range<1>(
      DP::range<1>(global_range), DP::range<1>(tile_size)),
      [=](DP::nd_item<1> item) {
        int64_t id = item.get_global_linear_id();
        auto out_ptr = out_acc.template get_pointer<scalar_t>();
        if (id < range) {
          out_ptr[id] = out_ptr[id] < p_ ? 1.0f : 0.0f;
        }
      }
    );
  });
}

template <typename scalar_t>
void bernoulli_tensor_kernel(Tensor &ret, const Tensor& p, uint64_t seed) {
  auto& queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  auto size = ret.numel() * sizeof(scalar_t);

  // First fill self with random number within range [0.0 - 1.0]
  auto cgf1 = DP_Q_CGF(cgh) {
    auto acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size).get_access();
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = DP::nd_range<1>(DP::range<1>(global_range), DP::range<1>(tile_size));
    FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf1);

  // Generate final bernoulli distributions
  auto cgf2 = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, p.data_ptr<scalar_t>(), size);
    auto out_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, ret.data_ptr<scalar_t>(), size);
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      int64_t id = item.get_global_linear_id();
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      if (id < range) {
        out_ptr[id] = out_ptr[id] < in_ptr[id] ? 1.0f : 0.0f;
      }
    };

    cgh.parallel_for<bernoulli_tensor_sycl_ker<scalar_t>>(DP::nd_range<1>(
      DP::range<1>(global_range), DP::range<1>(tile_size)), kfn);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf2);
}

} // namespace impl

Tensor & bernoulli_(Tensor & self, const Tensor & p_, Generator *_generator) {
  auto gen = at::get_generator_or_default<at::SYCLGenerator>(
      _generator, getDefaultSYCLGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // Call sycl kernel to generate bernoulli distribution
  AT_DISPATCH_FLOATING_TYPES(p_.scalar_type(), "bernoulli_tensor_kernel", [&] {
    impl::bernoulli_tensor_kernel<scalar_t>(self, p_, gen->current_seed());
  });
  return self;
}

Tensor & bernoulli_(Tensor &self, double p, Generator *_generator) {
  auto gen = at::get_generator_or_default<at::SYCLGenerator>(
      _generator, getDefaultSYCLGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // Call sycl kernel to generate bernoulli distribution
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "bernoulli_scalar_kernel", [&] {
    impl::bernoulli_scalar_kernel<scalar_t>(self, p, gen->current_seed());
  });
  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at

