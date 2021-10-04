#include <ATen/ATen.h>
#include <ATen/Config.h>

// needs to be included only once in library.
#include "ideep/ideep_pin_singletons.hpp"

using namespace ideep;

RegisterEngineAllocator cpu_alloc(
  engine::cpu_engine(),
  [](size_t size) {
    return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);
  },
  [](void* p) {
    c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);
  }
);

namespace torch_ipex { namespace cpu { namespace mkldnn {

void clear_computation_cache() {
  // Reset computation_cache for forward convolutions
  // As it also caches max number of OpenMP workers
  ideep::convolution_forward::t_store().clear();
}

}}}
