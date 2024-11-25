#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/Helpers.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace AtenIpexTypeQuantizedXPU {

Tensor _reshape_alias(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  return at::native::_reshape_alias(self, size, stride);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
