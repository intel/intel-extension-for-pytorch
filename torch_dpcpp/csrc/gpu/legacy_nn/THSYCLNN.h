#ifndef THSYCLNN_H
#define THSYCLNN_H

#include <stdbool.h>
#include <THDP/THSYCL.h>

#define THNN_(NAME) THSYCL_CONCAT_3(THNN_, SYCLReal, NAME)

#define THSYCLIndexTensor THSyclLongTensor
#define THSYCLIndexTensor_(NAME) THSyclLongTensor_ ## NAME

#define THSYCLIntegerTensor THSyclIntTensor
#define THSYCLIntegerTensor_(NAME) THSyclIntTensor_ ## NAME

typedef int64_t THSYCLIndex_t;
typedef int32_t THSYCLInteger_t;

#include <THDPNN/generic/THSYCLNN.h>
#include <THSYCLGenerateFloatTypes.h>

#endif
