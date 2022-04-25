#include <ATen/ATen.h>
#include "comm/AccumulateType.h"

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
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
  auto indices_data = indices_contig.data_ptr<int64_t>();
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

Tensor embedding_dense_backward_dpcpp(
    const Tensor& grad_,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  // printf("this is %s\n", __func__);
  auto grad_arg = TensorArg(grad_, "grad", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_backward", indices_arg, kLong);
  IsOnSameDevice("embedding_backward", grad_arg, indices_arg);

  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  int64_t stride = grad_weight.stride(0);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_.scalar_type(),
      "embedding_backward",
      [&]() {
        embedding_backward_dpcpp_kernel<scalar_t>(
            indices,
            grad.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            static_cast<int>(num_indices),
            static_cast<int64_t>(stride),
            static_cast<int>(padding_idx),
            grad_weight.numel(),
            scale_grad_by_freq);
      });

  return grad_weight;
}

} // namespace impl

Tensor embedding_dense_backward(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  return impl::embedding_dense_backward_dpcpp(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

} // namespace AtenIpexTypeXPU
} // namespace at
