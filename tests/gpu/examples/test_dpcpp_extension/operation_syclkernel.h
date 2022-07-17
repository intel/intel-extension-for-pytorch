#ifndef OPERATION_SYCLKERNEL_H_
#define OPERATION_SYCLKERNEL_H_

#include <CL/sycl.hpp>
#include "tiny_tensor.h"

TinyTensor run_syclkernel_operation_scaledown(const TinyTensor& inp, sycl::queue *q);

#endif