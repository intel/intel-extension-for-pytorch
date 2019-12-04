#include <THDP/THSYCLTensorMath.h>
#include <THDP/THSYCLGeneral.h>
#include <THDP/THSYCLTensorCopy.h>
#include <THDP/THSYCLTensor.hpp>
#include <THDP/THSYCLStorage.hpp>
#include <THDP/THSYCLTensorMathPointwise.h>
#include <THSYCLTensorInfo.h>

c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl> retainTensorImpl(THSYCLTensor* self) {
  c10::raw::intrusive_ptr::incref(self);
  return c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(self);
}

#include <THDP/generic/THSYCLTensorMathPointwise.cpp>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMathPointwise.cpp>
#include <THDP/THSYCLGenerateBoolType.h>
