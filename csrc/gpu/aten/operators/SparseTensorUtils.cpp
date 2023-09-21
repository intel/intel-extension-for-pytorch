#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <c10/util/irange.h>

#include "Loops.h"
#include "comm/FlattenIndicesCommon.h"

using namespace xpu::dpcpp;
using namespace at::sparse;

namespace at {
namespace AtenIpexTypeSparseXPU {
namespace {

template <typename func_t>
struct DPCPPKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(iter, f);
  }
};

Tensor flatten_indices_dpcpp_kernel(const Tensor& indices, IntArrayRef size) {
  return AtenIpexTypeSparseXPU::_flatten_indices<DPCPPKernelLauncher>(
      indices, size);
}

} // anonymous namespace

// NOTE [ Flatten Sparse Indices ]
// This helper function flattens a sparse indices tensor (a Tensor) into a 1D
// indices tensor. E.g.,
//   input = [[2, 4, 0],
//            [3, 1, 10]]
//   full_size = [2, 12]
//   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
//
// In other words, assuming that each `indices[i, :]` is a valid index to a
// tensor `t` of shape `full_size`. This returns the corresponding indices to
// the flattened tensor `t.reshape( prod(full_size[:indices.size(0)]), -1 )`.
// if forceClone is true, the result will forced to be a clone of self.
// if force_clone is true, the result will forced to be a clone of self.
Tensor flatten_indices(
    const Tensor& indices,
    IntArrayRef full_size,
    bool force_clone /*= false*/) {
  int64_t sparse_dim = indices.size(0);
  if (sparse_dim == 1) {
    if (force_clone) {
      return indices.squeeze(0).clone(at::MemoryFormat::Contiguous);
    } else {
      return indices.squeeze(0);
    }
  } else {
    if (!indices.numel()) {
      return at::zeros({indices.size(1)}, indices.options().dtype(kLong));
    }
    return flatten_indices_dpcpp_kernel(
        indices, full_size.slice(0, sparse_dim));
  }
}

} // namespace AtenIpexTypeSparseXPU
} // namespace at
