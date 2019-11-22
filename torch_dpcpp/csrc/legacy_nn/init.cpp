#include <THDP/THSYCL.h>
#include <THDPNN/THSYCLNN.h>

#include <THDP/THSYCLTensor.hpp>
#include <ATen/dpcpp/SYCLApplyUtils.h>
#include <THDPNN/THDPNNInnerProduct.h>
#include <cmath>

#define torch_(NAME) THSYCL_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) THSYCL_CONCAT_3(nn_, Real, NAME)

#define THSYCLNN_CHECK_SHAPE(STATE, I1, I2)                         \
  if (I1 != NULL && I2 != NULL && !THSYCLTensor_(isSameSizeAs)(STATE, I1, I2)) \
    {                                                     \
       THSYCLDescBuff s1 = THSYCLTensor_(sizeDesc)(STATE, I1);        \
       THSYCLDescBuff s2 = THSYCLTensor_(sizeDesc)(STATE, I2);        \
       THError(#I1 " and " #I2 " shapes do not match: "               \
         #I1 " %s, " #I2 " %s", s1.str, s2.str);                      \
    }

#define THSYCLNN_CHECK_SHAPE_INDICES(STATE, I1, I2)                     \
  if (I1 != NULL && I2 != NULL && !I1->sizes().equals(I2->sizes()))     \
    {                                                                   \
      THSYCLDescBuff s1 = THSYCLTensor_(sizeDesc)(STATE, I1);           \
      THSYCLDescBuff s2 = THLongTensor_sizeDesc(STATE, I2);             \
      THError(#I1 " and " #I2 " shapes do not match: "                  \
        #I1 " %s, " #I2 " %s", s1.str, s2.str);                         \
    }

#define THSYCLNN_CHECK_NELEMENT(STATE, I1, I2)                          \
  if (I1 != NULL && I2 != NULL ) {                              \
    ptrdiff_t n1 = THSYCLTensor_(nElement)(STATE, I1);          \
    ptrdiff_t n2 = THSYCLTensor_(nElement)(STATE, I2);                  \
    if (n1 != n2)                                         \
      {                                                 \
  THSYCLDescBuff s1 = THSYCLTensor_(sizeDesc)(STATE, I1);         \
  THSYCLDescBuff s2 = THSYCLTensor_(sizeDesc)(STATE, I2);         \
  THError(#I1 " and " #I2 " have different number of elements: "      \
    #I1 "%s has %ld elements, while "                         \
    #I2 "%s has %ld elements", s1.str, n1, s2.str, n2);             \
      }                                                 \
  }

#define THSYCLNN_CHECK_DIM_SIZE(STATE, T, DIM, DIM_SIZE, SIZE)              \
  if (THSYCLTensor_(nDimensionLegacyNoScalars)(STATE, T) != DIM ||        \
      THSYCLTensor_(sizeLegacyNoScalars)(STATE, T, DIM_SIZE) != SIZE) {     \
      THSYCLDescBuff s1 = THSYCLTensor_(sizeDesc)(STATE, T);            \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"         \
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str);   \
  }

#define THSYCLNN_CHECK_DIM_SIZE_INDICES(STATE, T, DIM, DIM_SIZE, SIZE)      \
  if (THSYCLIndexTensor_(nDimensionLegacyNoScalars)(STATE, T) != DIM ||     \
      THSYCLTensor_sizeLegacyNoScalars(STATE, T, DIM_SIZE) != SIZE) {     \
      THSYCLDescBuff s1 = THSYCLIndexTensor_(sizeDesc)(STATE, T);       \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"         \
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str);     \
  }

#define THSYCLNN_ARGCHECK(STATE, COND, ARG, T, FORMAT)                 \
  if (!(COND)) {                                               \
    THSYCLDescBuff s1 = THSYCLTensor_(sizeDesc)(STATE, T);             \
    THArgCheck(COND, ARG, FORMAT, s1.str);                             \
  }

#include <THDPNN/generic/AbsCriterion.c>
#include <THDP/THSYCLGenerateFloatTypes.h>

#include <THDPNN/generic/ClassNLLCriterion.c>
#include <THDP/THSYCLGenerateFloatTypes.h>

#include <THDPNN/MSECriterion.h>
#include <THDPNN/generic/MSECriterion.c>
#include <THDP/THSYCLGenerateFloatTypes.h>

#include <THDPNN/BCECriterion.h>
#include <THDPNN/generic/BCECriterion.c>
#include <THDP/THSYCLGenerateFloatTypes.h>
