#pragma once
#ifndef TH_SYCL_TENSOR_MATH_SCAN_INC
#define TH_SYCL_TENSOR_MATH_SCAN_INC

#include <legacy/THSYCLTensor.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLTensorMath.h>
#include <legacy/THSYCLAlgorithm.h>
#include <legacy/THSYCLTensorMathReduce.h>
#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>

template <typename T, typename BinOp>
class scanthrust_sycl_ker {};

template <typename T, typename BinOp>
class scanOuterDim_sycl_kernel {};

template <typename T, typename BinOp>
class scanInnerDim_sycl_kernel {};

#endif
