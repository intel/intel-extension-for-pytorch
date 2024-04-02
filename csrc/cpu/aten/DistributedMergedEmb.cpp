#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "MergedEmbeddingBag.h"

namespace torch_ipex {
namespace cpu {

using namespace at;
IPEX_DEFINE_DISPATCH(mergedemb_distribute_forward_local_kernel_stub);
IPEX_DEFINE_DISPATCH(mergedemb_distribute_forward_merge_kernel_stub);

/**
 * mergedemb_distribute_forward_local_cpu -> sparse_all_to_all ->
 * mergedemb_distribute_forward_merge_cpu. Will serve the
 * row-wise-distributed-merged-embedding-foward-lookup
 * 1. mergedemb_distribute_forward_local_cpu will finish the lookup for all
 * indices in local, the returned value is organized by 3 TensorList: val
 * Tensors, idx Tensors, ofs Tensors. The number of the Tensors in 1 TensorList
 * equal to world size. val[i], idx[i], ofs[i] is the tensors will be transfer
 * to rank i by sparse all to all. It contains the particial lookup for rank i
 *    since 1 ranks results might need to be lookuped in other ranks.
 * 2. mergedemb_distribute_forward_merge_cpu will reduce the val tensors got
 * from other ranks (indicate by idx tensors and ofs tensors)
 */

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
mergedemb_distribute_forward_local_cpu(
    const Tensor& weight,
    const std::vector<int64_t> row_offset,
    const TensorList& indices,
    const TensorList& offset,
    const int64_t rank,
    const int64_t world_size,
    const bool include_last_offsets) {
  RECORD_FUNCTION(
      "ipex::mergedemb_distribute_forward_local_cpu",
      c10::ArrayRef<c10::IValue>({}));
  return mergedemb_distribute_forward_local_kernel_stub(
      kCPU,
      weight,
      row_offset,
      indices,
      offset,
      rank,
      world_size,
      include_last_offsets);
}

void mergedemb_distribute_forward_merge_cpu(
    Tensor& output,
    const TensorList& idx,
    const TensorList& val,
    const TensorList& ofs,
    const int64_t num_emb) {
  // return None
  RECORD_FUNCTION(
      "ipex::mergedemb_distribute_forward_merge_cpu",
      c10::ArrayRef<c10::IValue>({}));
  return mergedemb_distribute_forward_merge_kernel_stub(
      kCPU, output, idx, val, ofs, num_emb);
}

IPEX_DEFINE_DISPATCH(mergedemb_distribute_backward_local_kernel_stub);
IPEX_DEFINE_DISPATCH(mergedemb_distribute_backward_merge_adagrad_update_stub);
/**
 * mergedemb_distribute_backward_local_cpu -> sparse_all_to_all ->
 * mergedemb_distribute_backward_merge_adagrad_update_cpu. Will serve the
 * distributed-merged-embedding-foward-lookup
 * 1. mergedemb_distribute_backward_local_cpu will finish the backward with
 * local grad (shape of [local BS * num_table * emb_dim]), the output grad will
 * be organzied by 3 TensorList: val Tensors, idx Tensors, ofs Tensors. The
 * number of the Tensors in 1 TensorList equal to world size. val[i], idx[i],
 * ofs[i] is the tensors will be transfer to rank i by sparse all to all. It
 * contains the grads for those indices on rank i.
 * 2. mergedemb_distribute_forward_merge_cpu will reduce the val tensors got
 * from other ranks (indicate by idx tensors and ofs tensors)
 */

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
mergedemb_distribute_backward_local_cpu(
    const Tensor& grad,
    const std::vector<int64_t> row_offset,
    const TensorList& indices,
    const TensorList& offset,
    const int64_t rank,
    const int64_t world_size,
    const bool include_last_offsets) {
  RECORD_FUNCTION(
      "ipex::mergedemb_distribute_backward_local_cpu",
      c10::ArrayRef<c10::IValue>({}));
  return mergedemb_distribute_backward_local_kernel_stub(
      kCPU,
      grad,
      row_offset,
      indices,
      offset,
      rank,
      world_size,
      include_last_offsets);
}

void mergedemb_distribute_backward_merge_adagrad_update_cpu(
    const TensorList& idx,
    const TensorList& val,
    const TensorList& ofs,
    Tensor& weight,
    Tensor& weight_trail,
    Tensor& hessian,
    const double lr,
    const double eps) {
  // return None
  RECORD_FUNCTION(
      "ipex::mergedemb_distribute_backward_merge_adagrad_update_cpu",
      c10::ArrayRef<c10::IValue>({}));
  return mergedemb_distribute_backward_merge_adagrad_update_stub(
      kCPU, idx, val, ofs, weight, weight_trail, hessian, lr, eps);
}
} // namespace cpu
} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  // forward
  // forward local
  m.def(
      "mergedemb_distribute_forward_local(Tensor weight, int[] row_offset, Tensor[] indices, Tensor[] offsets, int rank, int world_size, bool include_last) -> (Tensor[], Tensor[], Tensor[])");
  m.impl(
      "mergedemb_distribute_forward_local",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mergedemb_distribute_forward_local_cpu);
  // forward merge
  m.def(
      "mergedemb_distribute_forward_merge(Tensor output, Tensor[] idx, Tensor[] val, Tensor[] ofs, int num_emb) -> ()");
  m.impl(
      "mergedemb_distribute_forward_merge",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mergedemb_distribute_forward_merge_cpu);
  // backward
  // backward local
  m.def(
      "mergedemb_distribute_backward_local(Tensor grad, int[] row_offset, Tensor[] indices, Tensor[] offsets, int rank, int world_size, bool include_last) -> (Tensor[], Tensor[], Tensor[])");
  m.impl(
      "mergedemb_distribute_backward_local",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mergedemb_distribute_backward_local_cpu);

  // backward merge and adagrad update
  m.def(
      "mergedemb_distribute_backward_merge_adagrad_update(Tensor []idx, Tensor []val, Tensor []ofs, Tensor wgt, Tensor trail, Tensor hes, float lr, float eps) -> ()");
  m.impl(
      "mergedemb_distribute_backward_merge_adagrad_update",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::mergedemb_distribute_backward_merge_adagrad_update_cpu);
}
} // namespace
