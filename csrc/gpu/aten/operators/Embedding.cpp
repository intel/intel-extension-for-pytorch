#include <ATen/ATen.h>
#include "comm/AccumulateType.h"

#include <core/Device.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "EmbeddingBackwardKernel.h"
#include "Indexing.h"
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace {

template <typename IdxType>
static inline void indices_count(
    IdxType* indices_cnt,
    IdxType* indices,
    int64_t indices_num) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {
      auto row = indices[item.get_id(0)];
      atomicAdd((dpcpp_global_ptr_pt<IdxType>)(&indices_cnt[row]), 1);
    };
    __cgh.parallel_for(sycl::range<1>(indices_num), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename ValType, typename IdxType>
class EmbBwdOperator {
 public:
  EmbBwdOperator(IdxType* indices_cnt, IdxType padding_idx)
      : indices_cnt_(indices_cnt), padding_idx_(padding_idx) {}

  DPCPP_DEVICE void operator()(
      ValType* dst,
      ValType* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      ValType alpha) const {
    if (padding_idx_ == idx) {
      dst[dst_off] = 0;
      return;
    }

    if (indices_cnt_ != nullptr) {
      atomicAdd(
          (dpcpp_global_ptr_pt<ValType>)(dst + dst_off),
          src[src_off] / indices_cnt_[idx]);
    } else {
      atomicAdd((dpcpp_global_ptr_pt<ValType>)(dst + dst_off), src[src_off]);
    }
  }

  IdxType* indices_cnt_;
  int64_t padding_idx_;
};

template <typename scalar_t, typename index_t>
static inline void embedding_dense_backward_kernel(
    const Tensor& grad_output,
    const Tensor& grad_weight,
    const Tensor& indices,
    const Tensor& indices_cnt,
    int64_t padding_idx) {
  TensorInfo<index_t, int64_t> indices_info =
      getTensorInfo<index_t, int64_t>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, int64_t> src_info =
      getTensorInfo<scalar_t, int64_t>(grad_output);

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(grad_weight);

  auto func = indices_cnt.defined()
      ? EmbBwdOperator<scalar_t, index_t>(
            indices_cnt.data_ptr<index_t>(), padding_idx)
      : EmbBwdOperator<scalar_t, index_t>(nullptr, padding_idx);

  using SrcInfo = TensorInfo<scalar_t, int64_t>;
  using DstInfo = TensorInfo<scalar_t, int64_t>;
  using IdxInfo = TensorInfo<index_t, int64_t>;
  auto cfg = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      EmbBwdOperator<scalar_t, index_t>>::
      make_config(src_info, dst_info, indices_info, 0, 0, true, func);
  launch_index_kernel(cfg);
}

} // namespace

Tensor embedding_dense_backward(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  auto grad_arg = TensorArg(grad_output, "grad", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});
  isOnSameDevice("embedding_backward", grad_arg, indices_arg);

  auto num_indices = indices.numel();
  auto grad_output_cont =
      grad_output.contiguous().view({num_indices, grad_output.size(-1)});
  Tensor grad_weight;

  auto sorted_indices =
      at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor count;
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output_cont.scalar_type(),
      "embedding_backward",
      [&]() {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_backward", [&] {
              auto sorted_begin = sorted_indices.data_ptr<index_t>();
              auto orig_begin = orig_indices.data_ptr<index_t>();
              {
                sorted_indices.copy_(indices);
                xpu::pstl::iota(
                    orig_begin, orig_begin + num_indices, (index_t)0);
                xpu::pstl::sort<index_t, index_t>(
                    indices.data_ptr<index_t>(),
                    sorted_begin,
                    orig_begin,
                    num_indices,
                    false);
              }

              if (scale_grad_by_freq) {
                count = at::empty_like(sorted_indices);
                index_t* count_begin = count.data_ptr<index_t>();
                // Take the maximum of each count per unique key:
                // sorted: 2 5 5 5 7 7 8 9 9
                //  count: 1 3 3 3 2 2 1 2 2
                //
                xpu::pstl::count_by_segment<index_t, index_t, index_t>(
                    sorted_begin,
                    sorted_begin + num_indices,
                    count_begin,
                    [](index_t a, index_t b) {
                      return Numerics<index_t>::eq(a, b);
                    });
              }
              grad_weight = impl::
                  embedding_backward_deterministic_kernel<scalar_t, index_t>(
                      grad_output_cont,
                      orig_indices,
                      sorted_indices,
                      count,
                      num_weights,
                      padding_idx);
            });
      });
  return grad_weight;
}

} // namespace AtenIpexTypeXPU
} // namespace at
