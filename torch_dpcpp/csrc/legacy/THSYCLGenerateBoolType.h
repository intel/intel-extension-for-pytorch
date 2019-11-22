#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THGenerateBoolType.h"
#endif

#define scalar_t bool
#define ureal bool
#define accreal int64_t
#define Real Bool
#define SYCLReal SyclBool
#define THSYCL_REAL_IS_BOOL
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef ureal
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_BOOL

#ifndef THSYCLGenerateBoolType
#undef THSYCL_GENERIC_FILE
#endif
