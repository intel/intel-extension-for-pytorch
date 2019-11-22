#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorRandom.cpp"
#else

#include <c10/dpcpp/SYCLStream.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>

#include "THSYCLTensorRandomKernel.hpp"

THSYCLGenerator* THSYCLRandom_getGenerator(THSYCLState* state);

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)

void THSYCLTensor_(uniform)(struct THSYCLState *state, THSYCLTensor *self, at::Generator *_generator, double a, double b) {
  THSYCLGenerator *gen = THSYCLRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  c10::DeviceIndex device_idx = self->get_device();
  c10::sycl::SYCLStream cur_sycl_stream = c10::sycl::getCurrentSYCLStream(device_idx);
  auto cur_sycl_queue = cur_sycl_stream.sycl_queue();

  cur_sycl_queue.submit([&](cl::sycl::handler &cgh){
    auto self_data_size = self->numel() * (self->dtype().itemsize());
    void *self_data_ptr = self->data();

    auto acc = c10::sycl::SYCLAccessor<cl::sycl::access::mode::write>(cgh, self_data_ptr, self_data_size).get_access();

    int64_t tile_size, range, global_range;
    c10::sycl::parallel_for_setup(self->numel(), tile_size, range, global_range);
    auto num_work_items = cl::sycl::nd_range<1>(cl::sycl::range<1>(global_range), cl::sycl::range<1>(tile_size));

  #if defined(THSYCL_REAL_IS_FLOAT)
    FloatRandomFiller uniform_rnd_filler(acc, range, gen->state.initial_seed, a, b);
  #elif defined(THSYCL_REAL_IS_HALF)
    HalfRandomFiller uniform_rnd_filler(acc, range, gen->state.initial_seed, a, b);
  #else
    DoubleRandomFiller uniform_rnd_filler(acc, range, gen->state.initial_seed, a, b);
  #endif

    cgh.parallel_for(num_work_items, uniform_rnd_filler);
  });

  cur_sycl_queue.wait_and_throw();
}

void THSYCLTensor_(normal)(struct THSYCLState *state, THSYCLTensor *self, at::Generator *_generator, double mean, double stdv) {
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");

  // Generate uniform number
  THSYCLTensor_(uniform)(state, self, _generator, 0.0, 1.0);

  THSYCLGenerator *gen = THSYCLRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  c10::DeviceIndex device_idx = self->get_device();
  c10::sycl::SYCLStream cur_sycl_stream = c10::sycl::getCurrentSYCLStream(device_idx);
  auto cur_sycl_queue = cur_sycl_stream.sycl_queue();

  cur_sycl_queue.submit([&](cl::sycl::handler &cgh) {
    auto self_data_size = self->numel() * (self->dtype().itemsize());
    void *self_data_ptr = self->data();

    auto acc = c10::sycl::SYCLAccessor<cl::sycl::access::mode::read_write>(cgh, self_data_ptr, self_data_size).get_access();
    int64_t tile_size, range, global_range;

    bool recompute = ((self->numel() % 2) != 0);
    int64_t compute_num = recompute ? (self->numel() / 2 + 1) : (self->numel() / 2); // We will generate two normal element per time
    c10::sycl::parallel_for_setup(compute_num, tile_size, range, global_range);
    auto num_work_items = cl::sycl::nd_range<1>(cl::sycl::range<1>(global_range), cl::sycl::range<1>(tile_size));

    NormalRandomFiller<accreal> normal_rnd_filler(acc, compute_num, stdv, mean);
    cgh.parallel_for(num_work_items, normal_rnd_filler);
  });

  cur_sycl_queue.wait_and_throw();
}

#endif

#endif
