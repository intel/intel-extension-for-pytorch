#include <legacy/THSYCLTensorMath.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLTensorCopy.h>
#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLStorage.hpp>
#include <legacy/THSYCLTensorMathPointwise.h>
#include <core/detail/TensorInfo.h>

c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl> retainTensorImpl(THSYCLTensor* self) {
  c10::raw::intrusive_ptr::incref(self);
  return c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(self);
}

#include <legacy/generic/THSYCLTensorMathPointwise.cpp>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathPointwise.cpp>
#include <legacy/THSYCLGenerateBoolType.h>
