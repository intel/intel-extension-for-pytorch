//#include <THC/THCApply.cuh>
#include <TH/THHalf.h>
//#include <THC/THCNumerics.cuh>
#include <legacy/THSYCLTensorCopy.hpp>
#include <legacy/THSYCLTensor.hpp>
#include <type_traits>

// Copy operator for the pointwise apply kernel
template <typename TypeDst, typename TypeSrc>
struct CopyOp {
   inline void operator()(TypeDst* dst, TypeSrc* src) {
     printf("sycl CopyOp not implemented yet\n");
    //*dst = ScalarConvert<TypeSrc, TypeDst>::to(*src);
  }
};

#include <legacy/generic/THSYCLTensorCopy.cpp>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorCopy.cpp>
#include <legacy/THSYCLGenerateBoolType.h>

