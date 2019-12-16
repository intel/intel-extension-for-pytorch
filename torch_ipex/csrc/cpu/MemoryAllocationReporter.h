#include <mutex>
#include <unordered_map>

namespace torch_ipex {
namespace cpu {

class MemoryAllocationReporter {
 public:
  MemoryAllocationReporter() : allocated_(0) {}

  void New(void* ptr, size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_;
};

} // namespace cpu
} // namespace torch_ipex
