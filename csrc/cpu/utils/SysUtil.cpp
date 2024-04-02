#include "SysUtil.h"

void* ipex_alloc_aligned(size_t nbytes, size_t alignment) {
#ifdef _WIN32
  return _aligned_malloc(nbytes, alignment);
#else
  void* p_ptr = NULL;
  int err = posix_memalign(&p_ptr, alignment, nbytes);
  return p_ptr;
#endif
}

void ipex_free_aligned(void* data) {
#ifdef _WIN32
  _aligned_free(data);
#else
  free(data);
#endif
}