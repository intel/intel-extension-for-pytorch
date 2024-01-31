#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
//#include "init.h"
#include "timing.h"
#include "utils.h"

namespace torch_ipex {
namespace tpp {

#ifdef _OPENMP
#pragma message "Using OpenMP"
#endif

double ifreq = 1.0 / getFreq();

PassType globalPass = OTH;
REGISTER_SCOPE(other, "other");
REGISTER_SCOPE(w_vnni, "w_vnni");
REGISTER_SCOPE(w_xpose, "w_xpose");
REGISTER_SCOPE(a_xpose, "a_xpose");
REGISTER_SCOPE(a_vnni, "a_vnni");
REGISTER_SCOPE(zero, "zero");
REGISTER_SCOPE(pad_act, "pad_act");
REGISTER_SCOPE(unpad_act, "unpad_act");

int globalScope = 0;

thread_local unsigned int* rng_state = NULL;
thread_local struct drand48_data drng_state; // For non AVX512 version

unsigned int saved_seed = 0;
void xsmm_manual_seed(unsigned int seed) {
  saved_seed = seed;
#ifndef _WIN32
#pragma omp parallel
#else
// TODO: Fix crash on ICX Windows. CMPLRLLVM-55384 ?
//#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
#ifdef __x86_64__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

    if (rng_state) {
      libxsmm_rng_destroy_extstate(rng_state);
      rng_state = NULL;
    }
    rng_state = libxsmm_rng_create_extstate(seed + tid);
    srand48_r(seed + tid, &drng_state);
  }
}

unsigned int* get_rng_state() {
  if (rng_state) {
    return rng_state;
  }
  auto tid = omp_get_thread_num();
  rng_state = libxsmm_rng_create_extstate(saved_seed + tid);
  srand48_r(saved_seed + tid, &drng_state);
  return rng_state;
}

void init_libxsmm() {
  auto max_threads = omp_get_max_threads();
  PCL_ASSERT(
      max_threads <= MAX_THREADS,
      "Maximun %d threads supported, %d threads being used, please compile with increased  MAX_THREADS value\n",
      MAX_THREADS,
      max_threads);
  libxsmm_init();
  xsmm_manual_seed(0);
}

} // namespace tpp
} // namespace torch_ipex

/*static void init_submodules(pybind11::module& m) {
  auto& _submodule_list = get_submodule_list();
  for (auto& p : _submodule_list) {
    auto sm = m.def_submodule(p.first.c_str());
    auto module = py::handle(sm).cast<py::module>();
    p.second(module);
  }
}*/

// PYBIND11_MODULE(TORCH_MODULE_NAME, m) {
/*
PYBIND11_MODULE(_C, m) {
  init_submodules(m);
  m.def("print_debug_timers", &print_debug_timers, "print_debug_timers");
  m.def(
      "print_debug_thread_imbalance",
      &print_debug_thread_imbalance,
      "print_debug_thread_imbalance");
  m.def("reset_debug_timers", &reset_debug_timers, "reset_debug_timers");
};*/
