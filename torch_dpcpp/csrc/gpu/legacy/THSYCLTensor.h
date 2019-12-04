#pragma once

#include <TH/THTensor.h>
#include <THDP/THSYCLStorage.h>
#include <THDP/THSYCLGeneral.h>

#define THSYCLTensor_(NAME) THSYCL_CONCAT_4(TH, SYCLReal, Tensor_, NAME)

#define THSYCL_DESC_BUFF_LEN 64

typedef struct THSYCL_CLASS THSYCLDescBuff
{
  char str[THSYCL_DESC_BUFF_LEN];
} THSYCLDescBuff;


#include <THDP/generic/THSYCLTensor.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensor.h>
#include <THDP/THSYCLGenerateBoolType.h>
