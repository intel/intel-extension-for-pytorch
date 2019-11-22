#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateFloatType.h"
#endif

#define scalar_t float

#define accreal float
#define Real Float
#define SYCLReal Sycl
#define THSYCL_REAL_IS_FLOAT
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_FLOAT

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
