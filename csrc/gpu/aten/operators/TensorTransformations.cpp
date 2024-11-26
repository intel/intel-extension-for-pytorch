// clang-format off
#include <ATen/native/TensorTransformations.h>
// clang-format on
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <core/detail/OffsetCalculator.h>
#include "Loops.h"
#include "MemoryAccess.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/Helpers.h>

#include <cstddef>
#include <vector>

using namespace at::native;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) {
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size();

  TORCH_CHECK(
      total_rot_dims == 2,
      "expected total rotation dims == 2, but got dims = ",
      total_rot_dims);

  TORCH_CHECK(
      total_dims >= 2,
      "expected total dims >= 2, but got total dims = ",
      total_dims);

  TORCH_CHECK(
      dims[0] != dims[1] &&
          Numerics<int64_t>::abs(dims[0] - dims[1]) != total_dims,
      "expected rotation dims to be different, but got dim0 = ",
      dims[0],
      " and dim1 = ",
      dims[1]);

  // check range of dims
  TORCH_CHECK(
      dims[0] < total_dims && dims[0] >= -total_dims,
      "Rotation dim0 out of range, dim0 = ",
      dims[0]);

  TORCH_CHECK(
      dims[1] < total_dims && dims[1] >= -total_dims,
      "Rotation dim0 out of range, dim0 = ",
      dims[1]);

  // handle modulo with negative k
  k = (4 + (k % 4)) % 4;

  switch (k) {
    case 1:
      return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
    case 2:
      return self.flip(dims);
    case 3:
      return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
    default:
      return self.clone(MemoryFormat::Contiguous);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
