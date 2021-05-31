#include "thread.h"

extern "C" {
  typedef void* kmp_affinity_mask_t;
  void kmp_create_affinity_mask(kmp_affinity_mask_t*);
  int kmp_set_affinity_mask_proc(int, kmp_affinity_mask_t*);
  int kmp_set_affinity(kmp_affinity_mask_t*);
  void kmp_destroy_affinity_mask(kmp_affinity_mask_t*);
};

namespace toolkit {
  void thread_bind(int socket_id, int cores_per_socket, int core_id, int num_cores) {
    int phy_core_start = socket_id * cores_per_socket + core_id;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(phy_core_start, &mask);
    omp_set_num_threads(num_cores);
    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int phy_core_id = phy_core_start + thread_id;
      kmp_affinity_mask_t mask;
      kmp_create_affinity_mask(&mask);
      kmp_set_affinity_mask_proc(phy_core_id, &mask);
      kmp_set_affinity(&mask);
      kmp_destroy_affinity_mask(&mask);
    }
  }
}
