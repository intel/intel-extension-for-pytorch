#include <core/detail/HooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

static DPCPPHooksInterface* dpcpp_hooks = nullptr;

// See getCUDAHooks for some more commentary
const DPCPPHooksInterface& getDPCPPHooks() {
  static std::once_flag once;
  std::call_once(once, [] {
    dpcpp_hooks = DPCPPHooksRegistry()->Create("DPCPPHooks", DPCPPHooksArgs{}).release();
    if (!dpcpp_hooks) {
      dpcpp_hooks = new DPCPPHooksInterface();
    }
  });
  return *dpcpp_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(DPCPPHooksRegistry, DPCPPHooksInterface, DPCPPHooksArgs)

} // namespace at
