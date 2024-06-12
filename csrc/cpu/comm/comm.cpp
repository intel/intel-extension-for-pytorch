#include "comm.h"
#include <ATen/ATen.h>
#include "messager.h"

namespace torch_ipex {
namespace cpu {
int get_rank() {
#ifdef BUILD_CPU_WITH_ONECCL
  return Messenger::getInstance().getRank();
#else
  TORCH_CHECK(false, "BUILD_CPU_WITH_ONECCL is not enabled.");
  return 0;
#endif
}

int get_world_size() {
#ifdef BUILD_CPU_WITH_ONECCL
  return Messenger::getInstance().getSize();
#else
  TORCH_CHECK(false, "BUILD_CPU_WITH_ONECCL is not enabled.");
  return 0;
#endif
}

void barrier() {
#ifdef BUILD_CPU_WITH_ONECCL
  Messenger::getInstance().barrier();
#else
  TORCH_CHECK(false, "BUILD_CPU_WITH_ONECCL is not enabled.");
  return;
#endif
}
} // namespace cpu
} // namespace torch_ipex