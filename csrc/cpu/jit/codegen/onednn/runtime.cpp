#include "runtime.h"
#include <c10/core/CPUAllocator.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

// Non-default dnnl::graph::allocator needs an allocator.
// We would let it use c10::GetCPUAllocator's allocator,
// which uses posix_memalign with 64 byte alignment-size.
void* pytorch_default_allocator(size_t size, size_t alignment) {
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  return c10_allocator->raw_allocate(size);
}

// Non-default dnnl::graph::allocator needs a deallocator.
// We would let it use c10::GetCPUAllocator's deallocator.
void pytorch_default_deallocator(void* buf) {
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  c10_allocator->raw_deallocate(buf);
}

dnnl::engine& Engine::getEngine() {
  // Even if the default PyTorch CPU allocator would change, we'd still use the
  // stale value. In practice, we don't expect users to change the CPU allocator
  // dynamically anyway, as users preload jemalloc/tcmalloc at runtime, if they
  // would like to. But this behavior might need to be changed, as some models
  // work better with tcmalloc, while others work better with jemalloc, so
  // switching the CPU allocator at runtime can be useful.
  static dnnl::graph::allocator alloc{
      pytorch_default_allocator, pytorch_default_deallocator};
  static dnnl::engine cpu_engine = dnnl::graph::make_engine_with_allocator(
      dnnl::engine::kind::cpu, /* device_id = */ 0, alloc);
  return cpu_engine;
}

dnnl::stream& Stream::getStream() {
  static dnnl::stream cpu_stream{Engine::getEngine()};
  return cpu_stream;
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
