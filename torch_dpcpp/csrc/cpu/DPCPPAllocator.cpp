#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>

namespace torch_ipex {
namespace cpu {

class C10_API DPCPPMemoryAllocationReporter {
 public:
  DPCPPMemoryAllocationReporter() : allocated_(0) {}

  void New(void* ptr, size_t nbytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    size_table_[ptr] = nbytes;
    allocated_ += nbytes;
    LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated_
              << " bytes.";
  }

  void Delete(void* ptr) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = size_table_.find(ptr);
    CHECK(it != size_table_.end());
    allocated_ -= it->second;
    LOG(INFO) << "C10 deleted " << it->second << " bytes, total alloc "
              << allocated_ << " bytes.";
    size_table_.erase(it);
  }

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_;
};

struct C10_API DefaultDPCPPAllocator final : at::Allocator {
  DefaultDPCPPAllocator() {}
  ~DefaultDPCPPAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = c10::alloc_cpu(nbytes);
    if (FLAGS_caffe2_report_cpu_memory_usage && nbytes > 0) {
      getMemoryAllocationReporter().New(data, nbytes);
      return {data, data, &ReportAndDelete, at::Device(at::DeviceType::DPCPP, 0)};
    }
    return {data, data, &c10::free_cpu, at::Device(at::DeviceType::DPCPP, 0)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    getMemoryAllocationReporter().Delete(ptr);
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      return &ReportAndDelete;
    }
    return &c10::free_cpu;
  }

 protected:
  static DPCPPMemoryAllocationReporter& getMemoryAllocationReporter() {
    static DPCPPMemoryAllocationReporter reporter_;
    return reporter_;
  }
};

void NoDelete(void*) {}

at::Allocator* GetDPCPPAllocator() {
  return at::GetAllocator(at::DeviceType::DPCPP);
}

void SetDPCPPAllocator(at::Allocator* alloc) {
  SetAllocator(at::DeviceType::DPCPP, alloc);
}

static DefaultDPCPPAllocator g_dpcpp_alloc;

at::Allocator* GetDefaultDPCPPAllocator() {
  return &g_dpcpp_alloc;
}

} // cpu
} // torch_ipex

namespace c10 {

REGISTER_ALLOCATOR(at::DeviceType::DPCPP, &torch_ipex::cpu::g_dpcpp_alloc);

} // c10
