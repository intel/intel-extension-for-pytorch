#include "comm.h"
#include "messager.h"

namespace torch_ipex {
namespace cpu {
int get_rank() {
  return Messenger::getInstance().getRank();
}

int get_world_size() {
  return Messenger::getInstance().getSize();
}

void barrier() {
  Messenger::getInstance().barrier();
}
} // namespace cpu
} // namespace torch_ipex
