#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLContext.h>

#include <legacy/THSYCLGenerator.hpp>
#include <legacy/generic/THSYCLTensorRandomKernel.hpp>


namespace {

template <typename scalar_t>
class bernoulli_tensor_sycl_ker {};

template <typename scalar_t>
class bernoulli_scalar_sycl_ker {};

template <typename scalar_t>
void bernoulli_scalar_sycl_kernel(
  at::Tensor &ret, double p_, uint64_t seed) {
  static const auto write_mode = cl::sycl::access::mode::write;
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  auto size = ret.numel() * sizeof(scalar_t);
  // First fill self with random number within range [0.0 - 1.0]
  sycl_queue.submit([&](cl::sycl::handler &cgh){
    auto acc = c10::sycl::SYCLAccessor<write_mode>(cgh, ret.data_ptr<scalar_t>(), size).get_access();
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = cl::sycl::nd_range<1>(cl::sycl::range<1>(global_range), cl::sycl::range<1>(tile_size));
    FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  });

  // Generate final bernoulli distributions
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, ret.data_ptr<scalar_t>(), size);
    cgh.parallel_for<bernoulli_scalar_sycl_ker<scalar_t>>(cl::sycl::nd_range<1>(
      cl::sycl::range<1>(global_range), cl::sycl::range<1>(tile_size)),
      [=](cl::sycl::nd_item<1> item) {
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
void bernoulli_tensor_sycl_kernel(
  at::Tensor &ret, const at::Tensor& p, uint64_t seed) {
  static const auto write_mode = cl::sycl::access::mode::write;
  static const auto read_mode = cl::sycl::access::mode::read;
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  auto size = ret.numel() * sizeof(scalar_t);
  // First fill self with random number within range [0.0 - 1.0]
  sycl_queue.submit([&](cl::sycl::handler &cgh){
    auto acc = c10::sycl::SYCLAccessor<write_mode>(cgh, ret.data_ptr<scalar_t>(), size).get_access();
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    auto num_work_items = cl::sycl::nd_range<1>(cl::sycl::range<1>(global_range), cl::sycl::range<1>(tile_size));
    FloatRandomFiller uniform_rnd_filler(acc, range, seed, 0.0f, 1.0f);
    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  });

  // Generate final bernoulli distributions
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<read_mode>(cgh, p.data_ptr<scalar_t>(), size);
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, ret.data_ptr<scalar_t>(), size);
    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(ret.numel(), tile_size, range, global_range);
    cgh.parallel_for<bernoulli_tensor_sycl_ker<scalar_t>>(cl::sycl::nd_range<1>(
      cl::sycl::range<1>(global_range), cl::sycl::range<1>(tile_size)),
      [=](cl::sycl::nd_item<1> item) {
        int64_t id = item.get_global_linear_id();
        auto in_ptr = in_acc.template get_pointer<scalar_t>();
        auto out_ptr = out_acc.template get_pointer<scalar_t>();
        if (id < range) {
          out_ptr[id] = out_ptr[id] < in_ptr[id] ? 1.0f : 0.0f;
        }
      }
    );
  });
}

};

THSYCLGenerator* THSYCLRandom_getGenerator();

namespace at {
namespace native {

Tensor& bernoulli_tensor_sycl_(Tensor &self, const Tensor& p_, Generator* gen) {
  THSYCLGenerator *gen_ = THSYCLRandom_getGenerator();
  std::lock_guard<std::mutex> lock(gen_->mutex);
  // Call sycl kernel to generate bernoulli distribution
  AT_DISPATCH_FLOATING_TYPES(p_.scalar_type(), "bernoulli_tensor_sycl_kernel", [&] {
    bernoulli_tensor_sycl_kernel<scalar_t>(self, p_, gen_->state.initial_seed);
  });
  return self;
}

Tensor& bernoulli_scalar_sycl_(Tensor &self, double p, Generator* gen) {
  THSYCLGenerator *gen_ = THSYCLRandom_getGenerator();
  std::lock_guard<std::mutex> lock(gen_->mutex);
  // Call sycl kernel to generate bernoulli distribution
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "bernoulli_scalar_sycl_kernel", [&] {
    bernoulli_scalar_sycl_kernel<scalar_t>(self, p, gen_->state.initial_seed);
  });
  return self;
}

} // end of native
} // end of at

