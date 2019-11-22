#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateIntType.h"
#endif

#define scalar_t int32_t
#define accreal int64_t
#define Real Int
#define SYCLReal SyclInt
#define THSYCL_REAL_IS_INT
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_INT

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
