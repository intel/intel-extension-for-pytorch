#include "comm/RegistrationDeclarations.h"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "ValidateCompressedIndicesCommon.h"

#include "Loops.h"

namespace at {
namespace native {

namespace {

template <typename func_t>
struct DPCPPKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    at::AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(iter, f);
  }
};

} // namespace

void _validate_compressed_sparse_indices_xpu(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  at::AtenIpexTypeXPU::validate_compressed_sparse_indices_kernel<
      DPCPPKernelLauncher>(is_crow, cidx, idx, cdim, dim, nnz);
}

} // namespace native
} // namespace at
