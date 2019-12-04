#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateLongType.h"
#endif
#define scalar_t int64_t
#define accreal int64_t
#define Real Long
#define SYCLReal SyclLong
#define THSYCL_REAL_IS_LONG
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_LONG

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
