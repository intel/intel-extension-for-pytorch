#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <runtime/CachingHostAllocator.h>
#include <mutex>

#include <core/AllocationInfo.h>
#include <utils/Macros.h>

namespace torch_ipex::xpu {
namespace dpcpp {
class HostAllocator final : public at::Allocator {
 public:
  // Singleton
  static HostAllocator* Instance();

  static void deleter(void* ptr);

  at::DataPtr allocate(size_t size) override;

  at::DeleterFnPtr raw_deleter() const override;

  void* raw_allocate(size_t size);

  bool isHostPtr(const void* ptr);
  void emptyCache();

  void recordEvent(void* ptr, sycl::event& e);

  void release(void* ptr);

  void copy_data(void* dest, const void* src, std::size_t count)
      const final override;

 private:
  CachingHostAllocator* alloc();
};
} // namespace dpcpp
} // namespace torch_ipex::xpu
