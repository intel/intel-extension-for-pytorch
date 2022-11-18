#ifndef OPERATION_SYCLKERNEL_H_
#define OPERATION_SYCLKERNEL_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include "tiny_tensor.h"

TinyTensor run_syclkernel_operation_scaledown(
    const TinyTensor& inp,
    sycl::queue* q);

#endif