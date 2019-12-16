#ifndef THSYCL_GENERIC_FILE
#error "You must define THSYCL_GENERIC_FILE before including THSYCLGenerateAllTypes.h"
#endif

#ifndef THSYCLGenerateManyTypes
#define THSYCLAllLocalGenerateManyTypes
#define THSYCLGenerateManyTypes
#endif

#include <legacy/THSYCLGenerateFloatTypes.h>
#include <legacy/THSYCLGenerateIntTypes.h>

#ifdef THSYCLAllLocalGenerateManyTypes
#undef THSYCLAllLocalGenerateManyTypes
#undef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
