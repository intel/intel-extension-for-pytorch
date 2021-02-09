#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>

#include "cpu/DPCPPCPUAllocator.h"

namespace torch_ipex {
namespace cpu {

at::DataPtr DefaultDPCPPCPUAllocator::allocate(size_t nbytes) const {
  void* data = c10::alloc_cpu(nbytes);
  if (FLAGS_caffe2_report_cpu_memory_usage && nbytes > 0) {
    getMemoryAllocationReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::XPU, 0)};
  }
  return {data, data, &c10::free_cpu, at::Device(at::DeviceType::XPU, 0)};
}

at::DeleterFnPtr DefaultDPCPPCPUAllocator::raw_deleter() const {
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    return &ReportAndDelete;
  }
  return &c10::free_cpu;
}

} // namespace cpu
} // namespace torch_ipex
