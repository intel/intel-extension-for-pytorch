#ifndef _IDEEP_PIN_SINGLETONS_HPP_
#define _IDEEP_PIN_SINGLETONS_HPP_

#include "ideep.hpp"

namespace ideep {

engine& engine::cpu_engine() {
  static engine cpu_engine(kind::cpu, 0);
  return cpu_engine;
}

struct RegisterEngineAllocator {
  RegisterEngineAllocator(
      engine& eng,
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    eng.set_allocator(malloc, free);
  }
};

} // namespace ideep

#endif
