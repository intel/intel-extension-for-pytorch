#ifndef THSYCL_SPIN_INC
#define THSYCL_SPIN_INC

#include <legacy/THSYCLGeneral.h>
#include <time.h>

// enqueues a kernel that spins for the specified number of cycles
THSYCL_API void THSYCL_sleep(THSYCLState* state, int64_t cycles);

#endif
