#pragma onece

#include <c10/core/TensorTypeId.h>

namespace at {

static inline at::TensorTypeId DPCPPTensorId() {
  return at::TensorTypeId::DPCPPTensorId;
}

}
