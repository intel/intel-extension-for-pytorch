#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>

#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLContext.h>

namespace at { namespace native {

template <typename T>
class embedding_dense_backeward_sycl_ker {};
template <typename T>
class embedding_dense_backeward_sycl_idx_cnt_ker {};

template <typename scalar_t>
static inline void embedding_backward_sycl_kernel(
    int64_t* indices_data, const scalar_t* __restrict__ grad_data,
    scalar_t* __restrict__ grad_weight_data, int num_indices, int64_t stride,
    int padding_idx, int numel_weights, bool scale_grad_by_freq) {
  static const auto atomic_rw_mode = cl::sycl::access::mode::atomic;
  static const auto read_mode = cl::sycl::access::mode::read;
  static const auto write_mode = cl::sycl::access::mode::write;
  static const auto rw_mode = cl::sycl::access::mode::discard_read_write;
  static const auto gbuffer_target = cl::sycl::access::target::global_buffer;
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  auto row_num_weights = numel_weights / stride;
  cl::sycl::buffer<uint32_t, 1> idx_cnt(cl::sycl::range<1>{(size_t)row_num_weights});

  sycl_queue.submit([&](cl::sycl::handler & cgh) {
    int64_t rng, GRange, tileSize;

    // KER1: calc indices count for each
    auto idx_acc = c10::sycl::SYCLAccessor<read_mode>(
        cgh, indices_data, num_indices * sizeof(int64_t));

    if (scale_grad_by_freq) {
      auto idx_cnt_acc = idx_cnt.get_access<write_mode>(cgh);
      cgh.template fill(idx_cnt_acc, static_cast<uint32_t>(0));
      cl::sycl::accessor<uint32_t, 1, atomic_rw_mode, gbuffer_target>
          idx_cnt_ptr(idx_cnt, cgh, cl::sycl::range<1>(row_num_weights), 0);

      c10::sycl::parallel_for_setup(num_indices, tileSize, rng, GRange);
      cgh.parallel_for<embedding_dense_backeward_sycl_idx_cnt_ker<scalar_t>> (
          cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange),
          cl::sycl::range<1>(tileSize)),
          [=](cl::sycl::nd_item<1> item) {
        int64_t gid = item.get_global_linear_id();
        auto idx_ptr = idx_acc.template get_pointer<int64_t>();
        if (gid < num_indices)
          idx_cnt_ptr[idx_ptr[gid]].fetch_add(static_cast<uint32_t>(1));
      });
    }

    // KER2: calc gradient weight
    auto idx_cnt_acc = idx_cnt.get_access<read_mode>(cgh);
    auto g_acc = c10::sycl::SYCLAccessor<read_mode>(
        cgh, grad_data, num_indices * stride * sizeof(scalar_t));
    auto gw_acc = c10::sycl::SYCLAccessor<rw_mode, scalar_t>(
        cgh, grad_weight_data, numel_weights * sizeof(scalar_t));

    c10::sycl::parallel_for_setup(stride, tileSize, rng, GRange);
    cgh.parallel_for<embedding_dense_backeward_sycl_ker<scalar_t>> (
        cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange),
        cl::sycl::range<1>(tileSize)),
        [=](cl::sycl::nd_item<1> item) {
      int64_t gid = item.get_global_linear_id();
      if (gid < stride) {
        auto idx_ptr = idx_acc.template get_pointer<int64_t>();
        auto g_ptr = g_acc.template get_pointer<scalar_t>();
        auto gw_ptr = gw_acc.template get_pointer<scalar_t>();
        for (int nidx = 0; nidx < num_indices; nidx++) {
          auto idx = idx_ptr[nidx]/* - TH_INDEX_BASE*/;
          // TODO: remove branch to optimize performance ?
          if (scale_grad_by_freq) {
            gw_ptr[gid + idx * stride] += static_cast<scalar_t>(
                g_ptr[gid + nidx * stride] * 1.0 / (scalar_t)idx_cnt_acc[idx]);
          } else {
            gw_ptr[gid + idx * stride] += static_cast<scalar_t>(
                g_ptr[gid + nidx * stride]);
          }
        }
      }
    });
  });

  // auto idx_cnt_host_acc = idx_cnt.get_access<read_mode>();
  // printf("idx_cnt [1]-%d [2]-%d [3]-%d [4]-%d [5]-%d [6]-%d [7]-%d [8]-%d [9]-%d [10]-%d\n",
  //     idx_cnt_host_acc[0], idx_cnt_host_acc[1], idx_cnt_host_acc[2],
  //     idx_cnt_host_acc[3], idx_cnt_host_acc[4], idx_cnt_host_acc[5],
  //     idx_cnt_host_acc[6], idx_cnt_host_acc[7], idx_cnt_host_acc[8],
  //     idx_cnt_host_acc[9]);
}

Tensor embedding_dense_backward_sycl(const Tensor & grad_, const Tensor & indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  // printf("this is %s\n", __func__);
  auto grad_arg = TensorArg(grad_, "grad", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_backward", indices_arg, kLong);
  checkSameSYCL("embedding_backward", grad_arg, indices_arg);

  auto indices_contig = indices.contiguous();
  auto indices_data = indices_contig.data_ptr<int64_t>();
  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  int64_t stride = grad_weight.stride(0); // 3

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_.scalar_type(), "embedding_backward", [&]() {
    embedding_backward_sycl_kernel<scalar_t>(
        indices_data,
        grad.data_ptr<scalar_t>(),
        grad_weight.data_ptr<scalar_t>(),
        static_cast<int>(num_indices),
        static_cast<int64_t>(stride),
        static_cast<int>(padding_idx),
        grad_weight.numel(),
        scale_grad_by_freq
    );
  });

  return grad_weight;
}

}}
