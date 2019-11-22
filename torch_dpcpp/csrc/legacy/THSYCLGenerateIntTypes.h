#ifndef THSYCL_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntTypes.h"
#endif

#ifndef THSYCLGenerateManyTypes
#define THSYCLIntLocalGenerateManyTypes
#define THSYCLGenerateManyTypes
#endif

#include <THDP/THSYCLGenerateByteType.h>
#include <THDP/THSYCLGenerateCharType.h>
#include <THDP/THSYCLGenerateShortType.h>
#include <THDP/THSYCLGenerateIntType.h>
#include <THDP/THSYCLGenerateLongType.h>

#ifdef THSYCLIntLocalGenerateManyTypes
#undef THSYCLIntLocalGenerateManyTypes
#undef THSYCLGenerateManyTypes
#undef THSYCL_GENERIC_FILE
#endif
