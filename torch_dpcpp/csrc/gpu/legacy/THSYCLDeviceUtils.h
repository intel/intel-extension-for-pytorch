#ifndef THSYCL_DEVICE_UTILS_INC
#define THSYCL_DEVICE_UTILS_INC

#include <core/SYCLUtils.h>

template <typename T>
DP_BOTH inline T THSYCLMin(T a, T b) { return (a < b) ? a : b; }

template <typename T>
DP_BOTH inline T THSYCLMax(T a, T b) { return (a > b) ? a : b; }

template <typename T>
DP_BOTH inline T THSYCLCeilDiv(T a, T b) { return (a + b - 1) / b; }

/**
 *    Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
 *       multiple of b
 *       */

template <typename T>
DP_BOTH inline T THSYCLRoundUp(T a, T b) { return THSYCLCeilDiv(a, b) * b; }

#endif
