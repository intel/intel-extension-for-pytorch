#pragma once

#include <CL/sycl.hpp>
#include <c10/dpcpp/SYCLUtils.h>
#include <c10/dpcpp/SYCLStream.h>
#include <c10/dpcpp/SYCLMemory.h>

DP_DEF_K1(memory_copy);
#define SyclConvertToActualTypePtr(Scalar, buf_acc) static_cast<Scalar*>(static_cast<void*>(((buf_acc.get_pointer().get()))))
#define SyclConvertToActualOffset(Scalar, offset) offset/sizeof(Scalar)

namespace c10 {
namespace sycl {

using buffer_data_type_t = uint8_t;

enum syclMemcpyKind {
  HostToDevice,
  DeviceToHost,
  DeviceToDevice
};

void syclMemcpy(void *dst, const void *src, size_t n_bytes, syclMemcpyKind kind);
void syclMemcpyAsync(void *dst, const void *src, size_t n_bytes, syclMemcpyKind kind);

void syclMemset(void *data, int value, size_t n_bytes);
void syclMemsetAsync(void *data, int value, size_t n_bytes );
void* syclMalloc(size_t n_bytes);
void syclFree(void* ptr);
void syclFreeAll();

/*
 * When we need to write a sycl kernel, we need to create SYCLAccessor object first, then 
 * use get_pointer() to get the raw pointer in the sycl kernel code. If we need to use
 * cgh handler like .copy(), .fill() which only accept accessor, we can use .get_access()
 * to get accessor. 
 * Example:
 *   Acc = SYCLAccessor(cgh, virtual_ptr, n_bytes);
 *   kernel{
 *     acc_ptr =  Acc.template get_pointer<data_type>();
 *     acc_ptr[i] = ...
 *   }
*/

template <cl::sycl::access::mode AccMode, typename ScalarT = uint8_t>
class SYCLAccessor {
 static constexpr auto global_access = cl::sycl::access::target::global_buffer;
 using Accessor = cl::sycl::accessor<ScalarT, 1, AccMode, global_access>;
 public:
    SYCLAccessor(cl::sycl::handler &cgh, cl::sycl::buffer<ScalarT, 1> &buffer) 
      : accessor_(buffer.template get_access<AccMode, global_access>(cgh)) {}

    SYCLAccessor(cl::sycl::handler &cgh,  const void* virtual_ptr) 
      : offset_(syclGetBufferMap().get_offset(virtual_ptr)),
        accessor_(syclGetBufferMap().template get_buffer<ScalarT>(virtual_ptr).template get_access<AccMode, global_access>(cgh)) {}

    SYCLAccessor(cl::sycl::handler &cgh, const void* virtual_ptr, size_t n_bytes)
      : offset_(syclGetBufferMap().get_offset(virtual_ptr)),
        accessor_(syclGetBufferMap().template get_buffer<ScalarT>(virtual_ptr).template get_access<AccMode, global_access>(cgh, cl::sycl::range<1>(n_bytes), cl::sycl::id<1>(offset_))) {}

   const Accessor& get_access() const {
     return accessor_;
   }

   // get_pointer should only be used in sycl kernel.
   template<typename T>
   typename cl::sycl::global_ptr<T>::pointer_t const get_pointer() const {
     return (typename cl::sycl::global_ptr<T>::pointer_t const)(SyclConvertToActualTypePtr(T, accessor_) + SyclConvertToActualOffset(T, offset_));
   }

 private:
   ptrdiff_t offset_;
   Accessor accessor_;
};

template<typename scalar1, typename scalar2>
void syclMemoryCopyType(scalar1 * dst, const scalar2 * src, size_t n_elements)
{
  static constexpr auto write_mode = DP::access::mode::discard_write;
  static constexpr auto read_mode = DP::access::mode::read;
  auto &sycl_queue = getCurrentSYCLStream().sycl_queue();
  auto total_threads = sycl_queue.get_device(). template get_info<dp_dev_max_wgroup_size>();

  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<read_mode>(cgh, src);
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, dst);
    cgh.parallel_for<DP_K(memory_copy, scalar1, scalar2)>(
            DP::range<1>(total_threads),
            [=](DP::item<1> itemId) {
              auto in_ptr = in_acc.template get_pointer<scalar2>();
              auto out_ptr = out_acc.template get_pointer<scalar1>();
              auto id = itemId.get_id(0);
              for (auto i = id; i < n_elements; i += itemId.get_range()[0])
                out_ptr[i] = (scalar1)in_ptr[i];

            });
  };

  //launch kernel
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}
}// namespace sycl
}// namespace c10
