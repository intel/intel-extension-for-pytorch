#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Sort.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  Tensor sorted, indices;
  return sort_out_stable(self, false, dim, descending, sorted, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
