#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <runtime/CachingHostAllocator.h>
#include <mutex>

#include <core/AllocationInfo.h>
#include <core/Stream.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {
class HostAllocator final : public at::Allocator {
 public:
  // Singleton
  static HostAllocator* Instance();

  static void deleter(void* ptr);

  at::DataPtr allocate(size_t size) const override;

  at::DeleterFnPtr raw_deleter() const override;

  void* raw_allocate(size_t size);

  bool isHostPtr(const void* ptr);
  void emptyCache();

  void recordEvent(void* ptr, sycl::event& e);

  void release(void* ptr);

 private:
  CachingHostAllocator* alloc();
};
} // namespace dpcpp
} // namespace xpu
