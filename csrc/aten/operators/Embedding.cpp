#include <ATen/ATen.h>
#include "comm/AccumulateType.h"

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include "Indexing.h"
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
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      auto row = indices[item.get_id(0)];
      atomicAdd((dpcpp_global_ptr_pt<IdxType>)(&indices_cnt[row]), 1);
    };
    __cgh.parallel_for(DPCPP::range<1>(indices_num), kfn);
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

template <typename scalar_t, typename index_t>
static inline void embedding_backward_dpcpp_kernel(
    const Tensor& indices,
    const scalar_t* __restrict__ grad_data,
    scalar_t* __restrict__ grad_weight_data,
    int num_indices,
    int64_t stride,
    int padding_idx,
    int numel_weights,
    bool scale_grad_by_freq) {
  auto indices_contig = indices.contiguous();
  auto indices_data = indices_contig.data_ptr<index_t>();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  if (scale_grad_by_freq) {
    auto row_num_weights = numel_weights / stride;
    Tensor idx_counts = at::zeros(
        {row_num_weights * sizeof(uint32_t)},
        indices.options().dtype(at::kByte));
    uint32_t* idx_cnt_ptr = static_cast<uint32_t*>(idx_counts.data_ptr());

    auto cgf_scale = DPCPP_Q_CGF(cgh) {
      auto idx_data = indices_data;
      cgh.parallel_for(DPCPP::range<1>(1), [=](DPCPP::item<1> item) {
        auto idx_ptr = idx_data;
        for (int i = 0; i < num_indices; ++i) {
          idx_cnt_ptr[idx_ptr[i]] += static_cast<uint32_t>(1);
        }
      });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf_scale);

    auto cgf_scatter = DPCPP_Q_CGF(cgh) {
      auto idx_data = indices_data;
      auto g_data = grad_data;
      auto gw_data = grad_weight_data;

      cgh.parallel_for(DPCPP::range<1>(stride), [=](DPCPP::item<1> item) {
        int64_t gid = item.get_linear_id();
        auto idx_ptr = idx_data;
        auto g_ptr = g_data;
        auto gw_ptr = gw_data;
        for (int nidx = 0; nidx < num_indices; nidx++) {
          auto idx = idx_ptr[nidx];
          gw_ptr[gid + idx * stride] += static_cast<scalar_t>(
              g_ptr[gid + nidx * stride] * 1.0 / (scalar_t)idx_cnt_ptr[idx]);
        }
      });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf_scatter);

    if (padding_idx != -1) {
      auto cgf_pad = DPCPP_Q_CGF(cgh) {
        auto gw_data = grad_weight_data;

        cgh.parallel_for(DPCPP::range<1>(stride), [=](DPCPP::item<1> item) {
          int64_t gid = item.get_linear_id();
          auto gw_ptr = gw_data;
          gw_ptr[gid + padding_idx * stride] = static_cast<scalar_t>(0);
        });
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf_pad);
    }

  } else {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto idx_data = indices_data;
      auto g_data = grad_data;
      auto gw_data = grad_weight_data;

      cgh.parallel_for(DPCPP::range<1>(stride), [=](DPCPP::item<1> item) {
        int64_t gid = item.get_linear_id();
        auto idx_ptr = idx_data;
        auto g_ptr = g_data;
        auto gw_ptr = gw_data;
        for (int nidx = 0; nidx < num_indices; nidx++) {
          auto idx = idx_ptr[nidx];
          gw_ptr[gid + idx * stride] +=
              static_cast<scalar_t>(g_ptr[gid + nidx * stride]);
        }
      });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    if (padding_idx != -1) {
      auto cgf_pad = DPCPP_Q_CGF(cgh) {
        auto gw_data = grad_weight_data;

        cgh.parallel_for(DPCPP::range<1>(stride), [=](DPCPP::item<1> item) {
          int64_t gid = item.get_linear_id();
          auto gw_ptr = gw_data;
          gw_ptr[gid + padding_idx * stride] = static_cast<scalar_t>(0);
        });
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf_pad);
    }
  }
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
  IsOnSameDevice("embedding_backward", grad_arg, indices_arg);

  auto num_indices = indices.numel();
  auto grad_output_cont =
      grad_output.contiguous().view({num_indices, grad_output.size(-1)});
  auto grad_weight =
      at::zeros({num_weights, grad_output.size(-1)}, grad_output.options());

  int64_t stride = grad_weight.stride(0);

  // XXX: avoid software atomic_ref::fetch_add (compare_and_swap)
  // in violent contend case, `contend_per_dict > 2`.
  // retrieve the condition, if deterministic impl is done.
  // auto contend_per_dict = num_indices / num_weights;
  // if (contend_per_dict > 2) {
  if (num_weights < 128) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grad_output_cont.scalar_type(),
        "embedding_backward",
        [&]() {
          IPEX_DISPATCH_INDEX_TYPES(
              indices.scalar_type(), "embedding_backward", [&] {
                embedding_backward_dpcpp_kernel<scalar_t, index_t>(
                    indices,
                    grad_output_cont.data_ptr<scalar_t>(),
                    grad_weight.data_ptr<scalar_t>(),
                    static_cast<int>(num_indices),
                    static_cast<int64_t>(stride),
                    static_cast<int>(padding_idx),
                    grad_weight.numel(),
                    scale_grad_by_freq);
              });
        });

    return grad_weight;
  }

  at::Tensor indices_cnt;
  if (scale_grad_by_freq) {
    indices_cnt = at::zeros({num_weights}, indices.options());
    switch (indices.scalar_type()) {
      case at::ScalarType::Long:
        indices_count(
            indices_cnt.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            indices.numel());
        break;
      case at::ScalarType::Int:
        indices_count(
            indices_cnt.data_ptr<int32_t>(),
            indices.data_ptr<int32_t>(),
            indices.numel());
        break;
      default:
        break;
    };
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output_cont.scalar_type(),
      "embedding_backward_non_deterministic",
      [&]() {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_backward_non_deterministic", [&] {
              embedding_dense_backward_kernel<scalar_t, index_t>(
                  grad_output_cont,
                  grad_weight,
                  indices,
                  indices_cnt,
                  padding_idx);
            });
      });

  return grad_weight;
}

} // namespace AtenIpexTypeXPU
} // namespace at
