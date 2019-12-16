#pragma once
#include <TH/THStorageFunctions.h>
#include <legacy/THSYCLGeneral.h>

#define THSYCLStorage_(NAME) TH_CONCAT_4(TH,SYCLReal, Stroage_, NAME)

#include <legacy/generic/THSYCLStorage.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLStorage.h>
#include <legacy/THSYCLGenerateBoolType.h>
