// #pragma onece
#ifndef _ATEN_IPEX_TENSOR_TYPE_H_
#define _ATEN_IPEX_TENSOR_TYPE_H_

#include <c10/core/DispatchKey.h>

namespace at { namespace torch_ipex {

static inline at::DispatchKey DPCPPTensorId() {
  return at::DispatchKey::DPCPPTensorId;
}

}}

#endif
