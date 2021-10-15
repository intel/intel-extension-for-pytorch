#include "oneMKLUtils.h"

namespace xpu {
namespace oneMKL {

oneMKLExpInfo& oneMKLExpInfo::Instance() {
  static thread_local oneMKLExpInfo myInstance;
  return myInstance;
}

} // namespace oneMKL
} // namespace xpu
