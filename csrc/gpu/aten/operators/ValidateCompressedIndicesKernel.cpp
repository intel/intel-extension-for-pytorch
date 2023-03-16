#include "comm/RegistrationDeclarations.h"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "ValidateCompressedIndicesCommon.h"

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {
using namespace at::native;

namespace {

template <typename func_t>
struct DPCPPKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    dpcpp_kernel_for_tensor_iter(iter, f);
  }
};

} // namespace

void _validate_compressed_sparse_indices(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  validate_compressed_sparse_indices_kernel<DPCPPKernelLauncher>(
      is_crow, cidx, idx, cdim, dim, nnz);
}

} // namespace AtenIpexTypeXPU
} // namespace at
