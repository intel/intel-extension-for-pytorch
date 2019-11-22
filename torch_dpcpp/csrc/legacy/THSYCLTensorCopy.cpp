//#include <THC/THCApply.cuh>
#include <TH/THHalf.h>
//#include <THC/THCNumerics.cuh>
#include <THDP/THSYCLTensorCopy.hpp>
#include <THDP/THSYCLTensor.hpp>
#include <type_traits>

// Copy operator for the pointwise apply kernel
template <typename TypeDst, typename TypeSrc>
struct CopyOp {
   inline void operator()(TypeDst* dst, TypeSrc* src) {
     printf("sycl CopyOp not implemented yet\n");
    //*dst = ScalarConvert<TypeSrc, TypeDst>::to(*src);
  }
};

#include <THDP/generic/THSYCLTensorCopy.cpp>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorCopy.cpp>
#include <THDP/THSYCLGenerateBoolType.h>

