#include <ATen/ATen.h>
#include <utils/AccumulateType.h>

#include <core/Context.h>
#include <core/DPCPPTensorUtils.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <utils/ATDispatch.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename T>
class embedding_dense_backeward_dpcpp_ker {};
template <typename T>
class embedding_dense_backeward_dpcpp_idx_cnt_ker {};

template <typename scalar_t>
static inline void embedding_backward_dpcpp_kernel(
    int64_t* indices_data,
    const scalar_t* __restrict__ grad_data,
    scalar_t* __restrict__ grad_weight_data,
    int num_indices,
    int64_t stride,
    int padding_idx,
    int numel_weights,
    bool scale_grad_by_freq) {
  static const auto atomic_rw_mode = DPCPP::access::mode::atomic;
  static const auto read_mode = DPCPP::access::mode::read;
  static const auto write_mode = DPCPP::access::mode::write;
  static const auto rw_mode = DPCPP::access::mode::discard_read_write;
  static const auto gbuffer_target = DPCPP::access::target::global_buffer;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto row_num_weights = numel_weights / stride;
  DPCPP::buffer<uint32_t, 1> idx_cnt(DPCPP::range<1>{(size_t)row_num_weights});

  auto cgf = DPCPP_Q_CGF(cgh) {
    int64_t rng, GRange, tileSize;

    // KER1: calc indices count for each
    auto idx_acc = DPCPPAccessor<read_mode>(cgh, indices_data);

    if (scale_grad_by_freq) {
      auto idx_cnt_acc = idx_cnt.get_access<write_mode>(cgh);
      cgh.template fill(idx_cnt_acc, static_cast<uint32_t>(0));
      DPCPP::accessor<uint32_t, 1, atomic_rw_mode, gbuffer_target> idx_cnt_ptr(
          idx_cnt, cgh, DPCPP::range<1>(row_num_weights), 0);

      parallel_for_setup(num_indices, tileSize, rng, GRange);
      cgh.parallel_for<embedding_dense_backeward_dpcpp_idx_cnt_ker<scalar_t>>(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(GRange), DPCPP::range<1>(tileSize)),
          [=](DPCPP::nd_item<1> item) {
            int64_t gid = item.get_global_linear_id();
            auto idx_ptr = idx_acc.template get_pointer<int64_t>();
            if (gid < num_indices)
              idx_cnt_ptr[idx_ptr[gid]].fetch_add(static_cast<uint32_t>(1));
          });
    }

    // KER2: calc gradient weight
    auto idx_cnt_acc = idx_cnt.get_access<read_mode>(cgh);
    auto g_acc = DPCPPAccessor<read_mode>(cgh, grad_data);
    auto gw_acc = DPCPPAccessor<rw_mode, scalar_t>(cgh, grad_weight_data);

    parallel_for_setup(stride, tileSize, rng, GRange);
    cgh.parallel_for<embedding_dense_backeward_dpcpp_ker<scalar_t>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(GRange), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          int64_t gid = item.get_global_linear_id();
          if (gid < stride) {
            auto idx_ptr = idx_acc.template get_pointer<int64_t>();
            auto g_ptr = g_acc.template get_pointer<scalar_t>();
            auto gw_ptr = gw_acc.template get_pointer<scalar_t>();
            for (int nidx = 0; nidx < num_indices; nidx++) {
              auto idx = idx_ptr[nidx] /* - TH_INDEX_BASE*/;
              // TODO: remove branch to optimize performance ?
              if (scale_grad_by_freq) {
                gw_ptr[gid + idx * stride] += static_cast<scalar_t>(
                    g_ptr[gid + nidx * stride] * 1.0 /
                    (scalar_t)idx_cnt_acc[idx]);
              } else {
                gw_ptr[gid + idx * stride] +=
                    static_cast<scalar_t>(g_ptr[gid + nidx * stride]);
              }
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);

  // auto idx_cnt_host_acc = idx_cnt.get_access<read_mode>();
  // printf("idx_cnt [1]-%d [2]-%d [3]-%d [4]-%d [5]-%d [6]-%d [7]-%d [8]-%d
  // [9]-%d [10]-%d\n",
  //     idx_cnt_host_acc[0], idx_cnt_host_acc[1], idx_cnt_host_acc[2],
  //     idx_cnt_host_acc[3], idx_cnt_host_acc[4], idx_cnt_host_acc[5],
  //     idx_cnt_host_acc[6], idx_cnt_host_acc[7], idx_cnt_host_acc[8],
  //     idx_cnt_host_acc[9]);
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
  checkSameDPCPP("embedding_backward", grad_arg, indices_arg);

  auto indices_contig = indices.contiguous();
  auto indices_data = indices_contig.data_ptr<int64_t>();
  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  int64_t stride = grad_weight.stride(0); // 3

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad_.scalar_type(),
      "embedding_backward",
      [&]() {
        embedding_backward_dpcpp_kernel<scalar_t>(
            indices_data,
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

} // namespace AtenIpexTypeDPCPP
} // namespace at
