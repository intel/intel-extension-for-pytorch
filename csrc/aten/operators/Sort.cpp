#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Sort.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor&, Tensor&> sort_out(
    Tensor& sorted,
    Tensor& indices,
    const Tensor& input,
    long dim,
    bool order) {
  return sort_out_stable(input, true, dim, order, sorted, indices);
}

std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  Tensor sorted, indices;
  return sort_out_stable(self, true, dim, descending, sorted, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
