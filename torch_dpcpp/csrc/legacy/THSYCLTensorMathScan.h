#pragma once
#ifndef TH_SYCL_TENSOR_MATH_SCAN_INC
#define TH_SYCL_TENSOR_MATH_SCAN_INC

#include <THDP/THSYCLTensor.h>
#include <THDP/THSYCLGeneral.h>
#include <THDP/THSYCLTensorMath.h>
#include <THDP/THSYCLAlgorithm.h>
#include <THDP/THSYCLTensorMathReduce.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>

template <typename T, typename BinOp>
class scanthrust_sycl_ker {};

template <typename T, typename BinOp>
class scanOuterDim_sycl_kernel {};

template <typename T, typename BinOp>
class scanInnerDim_sycl_kernel {};

#endif
