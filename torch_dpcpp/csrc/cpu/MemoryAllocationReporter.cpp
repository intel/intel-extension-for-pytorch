#include "cpu/MemoryAllocationReporter.h"

#include <c10/util/Logging.h>

namespace torch_ipex {
namespace cpu {

void MemoryAllocationReporter::New(void* ptr, size_t nbytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  size_table_[ptr] = nbytes;
  allocated_ += nbytes;
  LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated_ << " bytes.";
}

void MemoryAllocationReporter::Delete(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = size_table_.find(ptr);
  CHECK(it != size_table_.end());
  allocated_ -= it->second;
  LOG(INFO) << "C10 deleted " << it->second << " bytes, total alloc " << allocated_ << " bytes.";
  size_table_.erase(it);
}

} // namespace cpu
} // namespace torch_ipex
