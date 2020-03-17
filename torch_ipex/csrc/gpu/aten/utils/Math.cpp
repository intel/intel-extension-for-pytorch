#include <core/Stream.h>
#include <core/Memory.h>

#include "Math.h"


DP_DEF_K1(memory_scale);
DP_DEF_K1(memory_scale1);
DP_DEF_K1(memory_scale2);

namespace c10 {
namespace sycl {

// dst = src * alpha
void syclMemoryScale(void * dst, const void * src, size_t n_elements, const float alpha)
{
  static constexpr auto write_mode = DP::access::mode::discard_write;
  static constexpr auto read_mode = DP::access::mode::read;
  auto &sycl_queue = getCurrentSYCLStream().sycl_queue();
  auto total_threads = sycl_queue.get_device(). template get_info<dp_dev_max_wgroup_size>();

  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<read_mode>(cgh, src);
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, dst);
    cgh.parallel_for<DP_K(memory_scale)>(
            DP::range<1>(total_threads),
            [=](DP::item<1> itemId) {
              auto in_ptr = in_acc.template get_pointer<float>();
              auto out_ptr = out_acc.template get_pointer<float>();
              auto id = itemId.get_id(0);
              for (auto i = id; i < n_elements; i += itemId.get_range()[0])
                out_ptr[i] = in_ptr[i] * alpha;
            });
  };

  //launch kernel
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// dst = src * eps + dst * (1 - eps)
void syclMemoryScale1(void * dst, const void * src, size_t n_elements, const double eps)
{
  static constexpr auto write_mode = DP::access::mode::discard_write;
  static constexpr auto read_mode = DP::access::mode::read;
  auto &sycl_queue = getCurrentSYCLStream().sycl_queue();
  auto total_threads = sycl_queue.get_device(). template get_info<dp_dev_max_wgroup_size>();

  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<read_mode>(cgh, src);
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, dst);
    cgh.parallel_for<DP_K(memory_scale1)>(
            DP::range<1>(total_threads),
            [=](DP::item<1> itemId) {
              auto in_ptr = in_acc.template get_pointer<float>();
              auto out_ptr = out_acc.template get_pointer<float>();
              auto id = itemId.get_id(0);
              for (auto i = id; i < n_elements; i += itemId.get_range()[0])
                out_ptr[i] = in_ptr[i] * eps + out_ptr[i] * (1 - eps);
            });
  };

  //launch kernel
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

// dst = src * alpha * eps + dst * (1 - eps)
void syclMemoryScale2(void * dst, const void * src, size_t n_elements, const float alpha, const double eps)
{
  static constexpr auto write_mode = DP::access::mode::discard_write;
  static constexpr auto read_mode = DP::access::mode::read;
  auto &sycl_queue = getCurrentSYCLStream().sycl_queue();
  auto total_threads = sycl_queue.get_device(). template get_info<dp_dev_max_wgroup_size>();

  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<read_mode>(cgh, src);
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, dst);
    cgh.parallel_for<DP_K(memory_scale2)>(
            DP::range<1>(total_threads),
            [=](DP::item<1> itemId) {
              auto in_ptr = in_acc.template get_pointer<float>();
              auto out_ptr = out_acc.template get_pointer<float>();
              auto id = itemId.get_id(0);
              for (auto i = id; i < n_elements; i += itemId.get_range()[0])
                out_ptr[i] = in_ptr[i] * alpha * eps + out_ptr[i] * (1 - eps);
            });
  };

  //launch kernel
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}


}//sycl
}//c10
