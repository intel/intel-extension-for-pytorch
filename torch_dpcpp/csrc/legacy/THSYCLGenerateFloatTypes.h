#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateFloatTypes.h"
#endif

#ifndef THSYCLGenerateManyTypes
#define THSYCLFloatLocalGenerateManyTypes
#define THSYCLGenerateManyTypes
#endif

#include <THDP/THSYCLGenerateHalfType.h>
#include <THDP/THSYCLGenerateFloatType.h>
#include <THDP/THSYCLGenerateDoubleType.h>

#ifdef THSYCLFloatLocalGenerateManyTypes
#undef THSYCLFloatLocalGenerateManyTypes
#undef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
