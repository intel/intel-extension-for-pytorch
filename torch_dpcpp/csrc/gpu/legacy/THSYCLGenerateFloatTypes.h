#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateFloatTypes.h"
#endif

#ifndef THSYCLGenerateManyTypes
#define THSYCLFloatLocalGenerateManyTypes
#define THSYCLGenerateManyTypes
#endif

#include <legacy/THSYCLGenerateHalfType.h>
#include <legacy/THSYCLGenerateFloatType.h>
#include <legacy/THSYCLGenerateDoubleType.h>

#ifdef THSYCLFloatLocalGenerateManyTypes
#undef THSYCLFloatLocalGenerateManyTypes
#undef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
