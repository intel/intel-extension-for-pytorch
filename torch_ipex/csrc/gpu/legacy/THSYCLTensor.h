#pragma once

#include <TH/THTensor.h>
#include <legacy/THSYCLStorage.h>
#include <legacy/THSYCLGeneral.h>

#define THSYCLTensor_(NAME) THSYCL_CONCAT_4(TH, SYCLReal, Tensor_, NAME)

#define THSYCL_DESC_BUFF_LEN 64

typedef struct THSYCL_CLASS THSYCLDescBuff
{
  char str[THSYCL_DESC_BUFF_LEN];
} THSYCLDescBuff;


#include <legacy/generic/THSYCLTensor.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensor.h>
#include <legacy/THSYCLGenerateBoolType.h>
