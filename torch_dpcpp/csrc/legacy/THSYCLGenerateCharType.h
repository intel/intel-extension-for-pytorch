#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateCharType.h"
#endif

#define scalar_t int8_t
#define accreal int64_t
#define Real Char
#define SYCLReal SyclChar
#define THSYCL_REAL_IS_CHAR
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t 
#undef accreal
#undef Real
#undef SYCLReal
#undef THSYCL_REAL_IS_CHAR

#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
