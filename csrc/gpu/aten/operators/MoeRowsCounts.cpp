#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <sycl/sycl.hpp>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "utils/ComputeEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace MoERowsCountsImpl {

struct MoeRowsCounts {
  static constexpr int GroupWorkItem = 256;

  MoeRowsCounts(
      const int* topk_indices,
      int* rows_for_experts,
      int* offsets,
      const int n_tokens,
      const int experts_offset,
      const int n_experts_local,
      const int n_topk)
      : topk_indices(topk_indices),
        rows_for_experts(rows_for_experts),
        offsets(offsets),
        n_tokens(n_tokens),
        experts_offset(experts_offset),
        n_experts_local(n_experts_local),
        n_topk(n_topk) {}

  static inline sycl::nd_range<1> get_nd_range(
      const int n_tokens,
      const int n_topk) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(
        (n_tokens * n_topk + GroupWorkItem - 1) / GroupWorkItem);
    return sycl::nd_range<1>(local * group, local);
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<1> item) const {
    auto global_id = item.get_global_linear_id();
    auto token_idx = global_id / n_topk;
    auto topk_idx = global_id % n_topk;
    if (token_idx >= n_tokens)
      return;

    int expert_id =
        topk_indices[token_idx * n_topk + topk_idx] - experts_offset;

    if (expert_id < 0 || expert_id >= n_experts_local) {
      offsets[token_idx * n_topk + topk_idx] = -1;
      return;
    }

    auto ref_num_tokens = sycl::atomic_ref<
        int,
        sycl::memory_order_relaxed,
        sycl::memory_scope_device,
        sycl::access::address_space::global_space>(
        *(rows_for_experts + expert_id));
    int old = ref_num_tokens.fetch_add(1);
    offsets[token_idx * n_topk + topk_idx] = old;
  }

  const int* topk_indices;
  int* rows_for_experts;
  int* offsets;
  const int n_tokens;
  const int experts_offset;
  const int n_experts_local;
  const int n_topk;
};

void moe_rows_counts(
    const int* topk_indices,
    int* rows_for_experts,
    int* offsets,
    const int n_tokens,
    const int experts_offset,
    const int n_experts_local,
    const int n_topk) {
  using Kernel = MoeRowsCounts;

  auto range = Kernel::get_nd_range(n_tokens, n_topk);
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        topk_indices,
        rows_for_experts,
        offsets,
        n_tokens,
        experts_offset,
        n_experts_local,
        n_topk);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

}; // namespace MoERowsCountsImpl

static std::tuple<at::Tensor, at::Tensor> moe_rows_counts(
    const Tensor& topk_indices, //[n_tokens, n_topk]
    const int64_t experts_offset,
    const int64_t n_experts_local) {
  auto shape = topk_indices.sizes().vec();
  TORCH_CHECK(
      shape.size() == 2,
      "activation must be 2D tensor, but got ",
      shape.size(),
      "D");
  int n_tokens = shape[0];
  int n_topk = shape[1];

  int n_experts_aligned = (n_experts_local + 7) / 8 * 8; // align to 8

  auto rows_for_experts =
      at::zeros({n_experts_aligned}, at::dtype(at::kInt).device(at::kXPU));
  auto expert_offsets =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  MoERowsCountsImpl::moe_rows_counts(
      reinterpret_cast<int*>(topk_indices.data_ptr()),
      reinterpret_cast<int*>(rows_for_experts.data_ptr()),
      reinterpret_cast<int*>(expert_offsets.data_ptr()),
      n_tokens,
      experts_offset,
      n_experts_local,
      n_topk);

  return std::make_tuple(rows_for_experts, expert_offsets);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("moe_rows_counts.moe", at::AtenIpexTypeXPU::moe_rows_counts);
}
} // namespace
