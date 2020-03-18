#pragma once

#include <CL/sycl.hpp>
#include <core/DPCPPUtils.h>
#include <core/Stream.h>
#include <core/Memory.h>

DP_DEF_K1(memory_copy);
#define SyclConvertToActualTypePtr(Scalar, buf_acc) static_cast<Scalar*>(static_cast<void*>(((buf_acc.get_pointer().get()))))
#define SyclConvertToActualOffset(Scalar, offset) offset/sizeof(Scalar)

namespace at {
namespace dpcpp {

using buffer_data_type_t = uint8_t;

enum dpcppMemcpyKind {
  HostToDevice,
  DeviceToHost,
  DeviceToDevice
};

void dpcppMemcpy(void *dst, const void *src, size_t n_bytes, dpcppMemcpyKind kind);
void dpcppMemcpyAsync(void *dst, const void *src, size_t n_bytes, dpcppMemcpyKind kind);

void dpcppMemset(void *data, int value, size_t n_bytes);
void dpcppMemsetAsync(void *data, int value, size_t n_bytes );
void* dpcppMalloc(size_t n_bytes);
void dpcppFree(void* ptr);
void dpcppFreeAll();

/*
 * When we need to write a dpcpp kernel, we need to create DPCPPAccessor object first, then 
 * use get_pointer() to get the raw pointer in the dpcpp kernel code. If we need to use
 * cgh handler like .copy(), .fill() which only accept accessor, we can use .get_access()
 * to get accessor. 
 * Example:
 *   Acc = DPCPPAccessor(cgh, virtual_ptr, n_bytes);
 *   kernel{
 *     acc_ptr =  Acc.template get_pointer<data_type>();
 *     acc_ptr[i] = ...
 *   }
*/

template <DP::access::mode AccMode, typename ScalarT = uint8_t>
class DPCPPAccessor {
 static constexpr auto global_access = DP::access::target::global_buffer;
 using Accessor = DP::accessor<ScalarT, 1, AccMode, global_access>;
 public:
    DPCPPAccessor(DP::handler &cgh, DP::buffer<ScalarT, 1> &buffer) 
      : accessor_(buffer.template get_access<AccMode, global_access>(cgh)) {}

    DPCPPAccessor(DP::handler &cgh,  const void* virtual_ptr) 
      : offset_(dpcppGetBufferMap().get_offset(virtual_ptr)),
        accessor_(dpcppGetBufferMap().template get_buffer<ScalarT>(virtual_ptr).template get_access<AccMode, global_access>(cgh)) {}

    DPCPPAccessor(DP::handler &cgh, const void* virtual_ptr, size_t n_bytes)
      : offset_(dpcppGetBufferMap().get_offset(virtual_ptr)),
        accessor_(dpcppGetBufferMap().template get_buffer<ScalarT>(virtual_ptr).template get_access<AccMode, global_access>(cgh, DP::range<1>(n_bytes), DP::id<1>(offset_))) {}

   const Accessor& get_access() const {
     return accessor_;
   }

   // get_pointer should only be used in dpcpp kernel.
   template<typename T>
   typename DP::global_ptr<T>::pointer_t const get_pointer() const {
     return (typename DP::global_ptr<T>::pointer_t const)(SyclConvertToActualTypePtr(T, accessor_) + SyclConvertToActualOffset(T, offset_));
   }

 private:
   ptrdiff_t offset_;
   Accessor accessor_;
};

template<typename scalar1, typename scalar2>
void dpcppMemoryCopyType(scalar1 * dst, const scalar2 * src, size_t n_elements)
{
  static constexpr auto write_mode = DP::access::mode::discard_write;
  static constexpr auto read_mode = DP::access::mode::read;
  auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = dpcpp_queue.get_device(). template get_info<dp_dev_max_wgroup_size>();

  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = DPCPPAccessor<read_mode>(cgh, src);
    auto out_acc = DPCPPAccessor<write_mode>(cgh, dst);
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
  DP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}
}// namespace dpcpp
}// namespace c10
