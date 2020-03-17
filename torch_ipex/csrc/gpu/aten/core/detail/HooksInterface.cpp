#include <core/detail/HooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

static SYCLHooksInterface* sycl_hooks = nullptr;

// See getCUDAHooks for some more commentary
const SYCLHooksInterface& getSYCLHooks() {
  static std::once_flag once;
  std::call_once(once, [] {
    sycl_hooks = SYCLHooksRegistry()->Create("SYCLHooks", SYCLHooksArgs{}).release();
    if (!sycl_hooks) {
      sycl_hooks = new SYCLHooksInterface();
    }
  });
  return *sycl_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(SYCLHooksRegistry, SYCLHooksInterface, SYCLHooksArgs)

} // namespace at
