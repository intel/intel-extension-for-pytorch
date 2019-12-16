#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateDoubleType.h"
#endif

#define scalar_t double
#define accreal double
#define Real Double
#define SYCLReal SyclDouble
#define THSYCL_REAL_IS_DOUBLE
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_DOUBLE

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif

