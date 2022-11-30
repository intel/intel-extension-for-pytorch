#include <ATen/ATen.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include "comm/RegistrationDeclarations.h"
namespace at {
namespace AtenIpexTypeXPU {
// Onednn does not support double, long, complex etc.
// 1.0f will be returned if the scalar type of 't' is double or complex etc.
double AsignOneDnnQuantizeScale(
    const Tensor& t,
    const double default_scale,
    const int64_t zero_point);

void cat_(const ITensorListRef& container, int64_t dim, Tensor& out);

} // namespace AtenIpexTypeXPU
} // namespace at
