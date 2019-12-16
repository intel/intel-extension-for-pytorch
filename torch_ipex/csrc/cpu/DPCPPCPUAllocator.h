#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>

#include "cpu/MemoryAllocationReporter.h"

namespace torch_ipex {
namespace cpu {

struct DefaultDPCPPCPUAllocator final : at::Allocator {
  DefaultDPCPPCPUAllocator() {}
  ~DefaultDPCPPCPUAllocator() override {}

  at::DataPtr allocate(size_t nbytes) const override;
  at::DeleterFnPtr raw_deleter() const override;

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    getMemoryAllocationReporter().Delete(ptr);
    c10::free_cpu(ptr);
  }

protected:
  static MemoryAllocationReporter& getMemoryAllocationReporter() {
    static MemoryAllocationReporter reporter_;
    return reporter_;
  }
};

} // namespace cpu
} // namespace torch_ipex
