#include "oneMKLUtils.h"

namespace xpu {
namespace oneMKL {

oneMKLExpInfo& oneMKLExpInfo::Instance() {
  static thread_local oneMKLExpInfo myInstance;
  return myInstance;
}

bool set_onemkl_verbose(int level) {
#ifdef USE_ONEMKL
  auto status = mkl_verbose(level);
  return status != -1;
#else
  return false;
#endif
}

} // namespace oneMKL
} // namespace xpu
