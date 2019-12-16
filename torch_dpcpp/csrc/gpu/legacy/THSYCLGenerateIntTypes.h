#ifndef THSYCL_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntTypes.h"
#endif

#ifndef THSYCLGenerateManyTypes
#define THSYCLIntLocalGenerateManyTypes
#define THSYCLGenerateManyTypes
#endif

#include <legacy/THSYCLGenerateByteType.h>
#include <legacy/THSYCLGenerateCharType.h>
#include <legacy/THSYCLGenerateShortType.h>
#include <legacy/THSYCLGenerateIntType.h>
#include <legacy/THSYCLGenerateLongType.h>

#ifdef THSYCLIntLocalGenerateManyTypes
#undef THSYCLIntLocalGenerateManyTypes
#undef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
