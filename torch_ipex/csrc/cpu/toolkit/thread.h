#include "omp.h"
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <stdio.h>

namespace toolkit {
  void thread_bind(int socket_id, int cores_per_socket, int core_id, int num_cores);
}
