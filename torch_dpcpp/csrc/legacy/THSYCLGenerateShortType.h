#ifndef THSYCL_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define scalar_t int16_t
#define accreal int64_t
#define Real Short
#define SYCLReal SyclShort
#define THSYCL_REAL_IS_SHORT
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_SHORT

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
