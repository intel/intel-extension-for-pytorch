#include "comm.h"
#include <ATen/ATen.h>
#include "messager.h"

namespace torch_ipex {
namespace cpu {
int get_rank() {
#ifdef USE_CCL
  return Messenger::getInstance().getRank();
#else
  TORCH_CHECK(false, "USE_CCL is not enabled.");
  return 0;
#endif
}

int get_world_size() {
#ifdef USE_CCL
  return Messenger::getInstance().getSize();
#else
  TORCH_CHECK(false, "USE_CCL is not enabled.");
  return 0;
#endif
}

void barrier() {
#ifdef USE_CCL
  Messenger::getInstance().barrier();
#else
  TORCH_CHECK(false, "USE_CCL is not enabled.");
  return;
#endif
}
} // namespace cpu
} // namespace torch_ipex