#include "vec512_bfloat16.h"
#include "vec512_int8.h"
#if defined(CPU_CAPABILITY_AVX512_FP16)
#include "vec512_fp16.h"
#endif
#include "perf_kernel/kernel.h"
