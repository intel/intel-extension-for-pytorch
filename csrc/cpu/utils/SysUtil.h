#pragma once

#include <stdlib.h>

#ifndef _WIN32
#define IPEX_FORCE_INLINE inline __attribute__((always_inline))
#else
#define IPEX_FORCE_INLINE __forceinline
#endif

void* ipex_alloc_aligned(size_t nbytes, size_t alignment);
void ipex_free_aligned(void* data);