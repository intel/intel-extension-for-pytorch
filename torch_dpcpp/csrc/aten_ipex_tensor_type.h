// #pragma onece
#ifndef _ATEN_IPEX_TENSOR_TYPE_H_
#define _ATEN_IPEX_TENSOR_TYPE_H_

#include <c10/core/TensorTypeId.h>

namespace at { namespace torch_ipex {

static inline at::TensorTypeId DPCPPTensorId() {
  return at::TensorTypeId::DPCPPTensorId;
}

}}

#endif
