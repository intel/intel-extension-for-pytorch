#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateHalfType.h"
#endif

#include <TH/THHalf.h>

#define scalar_t THHalf
#define accreal float
#define Real Half

#define SYCLReal SyclHalf

#define THSYCL_REAL_IS_HALF
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real

#undef SYCLReal

#undef THSYCL_REAL_IS_HALF

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
