#include <ATen/Context.h>
#include <ATen/dpcpp/SYCLContext.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/dpcpp/SYCLApplyUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <THDP/THSYCLDeviceUtils.h>

DP_DEF_K2(triuTrilSycl, typename scalar_t, typename IndexType, bool upper);

namespace at {
namespace native {

template <typename scalar_t, typename IndexType, bool upper>
void triu_tril_sycl_kernel(
    scalar_t *result,
    const scalar_t *src,
    const IndexType stride0,
    const IndexType stride1,
    const int64_t k,
    const int64_t N) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  int64_t group_size = c10::sycl::syclMaxWorkGroupSize(queue);
  auto num_groups    = THSYCLCeilDiv(N, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto result_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, result);
    auto src_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, src);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto result_ptr = result_acc.template get_pointer<scalar_t>();
      auto src_ptr = src_acc.template get_pointer<scalar_t>();

      for (size_t linearIndex = item.get_global_id(0);
           linearIndex < (size_t) N; linearIndex += item.get_global_range()[0]) {
        IndexType row, col;

        if (stride0 > stride1) {
          row = (IndexType) (linearIndex / stride0);
          col = (IndexType) (linearIndex % stride0 / stride1);
        } else {
          row = (IndexType) ((linearIndex % stride1) / stride0);
          col = (IndexType) (linearIndex / stride1);
        }

        bool mask = upper ? (col - row >= k) : (col - row <= k);
        result_ptr[linearIndex] = mask ? src_ptr[linearIndex] : scalar_t(0);
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(triuTrilSycl, scalar_t, IndexType, upper)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);

  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <bool upper>
Tensor& triu_tril_sycl_template(Tensor& result, const Tensor& self, int64_t k, const char* name) {
  int64_t N = self.numel();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), name, [&]{
   if (sycl::detail::canUse32BitIndexMath(self)) {
      auto self_info = sycl::detail::getTensorInfo<scalar_t, int32_t>(self);
      triu_tril_sycl_kernel<scalar_t, int32_t, upper>(
        result.data_ptr<scalar_t>(),
        self.data_ptr<scalar_t>(),
        self_info.strides[0],
        self_info.strides[1],
        k, N);
    } else {
      auto self_info = sycl::detail::getTensorInfo<scalar_t, int64_t>(self);
      triu_tril_sycl_kernel<scalar_t, int64_t, upper>(
        result.data_ptr<scalar_t>(),
        self.data_ptr<scalar_t>(),
        self_info.strides[0],
        self_info.strides[1],
        k, N);
    }
  });

  return result;
}


Tensor& tril_sycl_(Tensor &self, int64_t k) {
  return tril_sycl_out(self, self, k);
}

Tensor& tril_sycl_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }

  return triu_tril_sycl_template<false>(result, self, k, "tril");
}

Tensor& triu_sycl_(Tensor &self, int64_t k) {
  return triu_sycl_out(self, self, k);
}

Tensor& triu_sycl_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  return triu_tril_sycl_template<true>(result, self, k, "triu");
}



} // namespace native
} // namespace at
