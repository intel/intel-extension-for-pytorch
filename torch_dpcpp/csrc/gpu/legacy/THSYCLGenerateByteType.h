#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateByteType.h"
#endif

#define scalar_t uint8_t
#define accreal int64_t
#define Real Byte
#define SYCLReal SyclByte
#define THSYCL_REAL_IS_BYTE
#line 1 THSYCL_GENERIC_FILE
#include THSYCL_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef SYCLReal
#ifndef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
