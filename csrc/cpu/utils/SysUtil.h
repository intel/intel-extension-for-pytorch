#pragma once

#ifndef _MSC_VER
#define IPEX_FORCE_INLINE inline __attribute__((always_inline))
#else
#define IPEX_FORCE_INLINE __forceinline
#endif