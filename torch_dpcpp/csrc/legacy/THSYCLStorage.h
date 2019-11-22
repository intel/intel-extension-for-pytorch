#pragma once
#include <TH/THStorageFunctions.h>
#include <THDP/THSYCLGeneral.h>

#define THSYCLStorage_(NAME) TH_CONCAT_4(TH,SYCLReal, Stroage_, NAME)

#include <THDP/generic/THSYCLStorage.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLStorage.h>
#include <THDP/THSYCLGenerateBoolType.h>
