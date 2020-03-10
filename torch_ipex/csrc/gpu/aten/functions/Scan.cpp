#include <ATen/ATen.h>

#include <core/SYCL.h>
#include <core/SYCLMemory.h>
#include <utils/MathReduce.h>
#include <utils/General.h>
#include <utils/Numerics.h>


using namespace at::detail;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename T, typename BinOp>
class scanthrust_sycl_ker {};

template <typename T, typename BinOp>
class scanOuterDim_sycl_kernel {};

template <typename T, typename BinOp>
class scanInnerDim_sycl_kernel {};

template <typename scalar_t, class BinaryFunction>
typename std::enable_if<!IS_HALF(scalar_t), void>::type
scanThrust(Tensor & dst, Tensor & src, BinaryFunction binary_op) {
  auto src_data = src.data_ptr<scalar_t>();
  auto src_size = src.nbytes();
  auto dst_data = dst.data_ptr<scalar_t>();
  auto dst_size = dst.nbytes();
  ptrdiff_t size = src.numel();

  auto& queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  auto cgf = DP_Q_CGF(cgh) {
    auto acc_src = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, src_data, src_size);
    auto acc_dst = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, dst_data, dst_size);
    // (TODO) single_task need replaced due to low efficiency
    cgh.single_task<scanthrust_sycl_ker<scalar_t, BinaryFunction>>([=]() {
      auto ptr_dst = acc_dst.template get_pointer<scalar_t>();
      auto ptr_src = acc_src.template get_pointer<scalar_t>();
      sycl_inclusive_scan(ptr_src, ptr_src + size, ptr_dst, binary_op);
    });
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template<typename scalar_t, class BinaryOp>
void scanOuterDim(Tensor & tgt, Tensor & src, int dimension,
    scalar_t init, BinaryOp binary_op) {
  auto totalElements = tgt.numel();
  auto tgt_data = tgt.data_ptr<scalar_t>();
  auto tgt_size = tgt.nbytes();
  auto src_data = src.data_ptr<scalar_t>();
  auto src_size = src.nbytes();
  int64_t n = src.size(dimension);
  int64_t stride = src.stride(dimension);
  int64_t batch = totalElements / (n * stride);

  auto& queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  int64_t rng, GRange, tileSize;
  c10::sycl::parallel_for_setup(totalElements, tileSize, rng, GRange);

  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, src_data, src_size);
    auto tgt_acc = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, tgt_data, tgt_size);

    auto kfn = DP_Q_KFN(DP::nd_item<1> item) {
      auto src_ptr = src_acc.template get_pointer<scalar_t>();
      auto tgt_ptr = tgt_acc.template get_pointer<scalar_t>();
      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
        int64_t base_start = linearIndex % (batch * stride);
        int64_t start = (base_start / stride) * n * stride + base_start % stride;
        scalar_t result = init;
        for (int j = 0; j < n; ++j) {
          result = binary_op(result, src_ptr[start + j * stride]);
          tgt_ptr[start + j * stride] = result;
        }
      }
    };

    cgh.parallel_for<scanOuterDim_sycl_kernel<scalar_t, BinaryOp>>(
        DP::nd_range<1>(DP::range<1>(tileSize), DP::range<1>(tileSize)), kfn);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template<typename scalar_t, class BinaryFunction>
void scanInnermostDim(Tensor & tgt, Tensor & src, scalar_t init, BinaryFunction binary_op) {
  auto& queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  auto totalElements = tgt.numel();
  auto tgt_data = tgt.data_ptr<scalar_t>();
  auto tgt_size = tgt.nbytes();
  auto src_data = src.data_ptr<scalar_t>();
  auto src_size = src.nbytes();
  auto dimension = tgt.dim() - 1;
  int64_t n = src.size(dimension);
  int64_t stride = src.stride(dimension);
  int64_t batch = totalElements / (n * stride);

  int64_t rng, GRange, tileSize;
  c10::sycl::parallel_for_setup(totalElements, tileSize, rng, GRange);

  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, src_data, src_size);
    auto tgt_acc = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, tgt_data, tgt_size);

    auto kfn = DP_Q_KFN(DP::nd_item<1> item) {
      auto src_ptr = src_acc.template get_pointer<scalar_t>();
      auto tgt_ptr = tgt_acc.template get_pointer<scalar_t>();

      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
        int64_t start = linearIndex % batch * n;
        scalar_t result = init;
        for (int64_t j = 0; j < n; ++j) {
          result = binary_op(result, src_ptr[start + j]);
          tgt_ptr[start + j] = result;
        }
      }
    };

    cgh.parallel_for<scanInnerDim_sycl_kernel<scalar_t, BinaryFunction>>(
        DP::nd_range<1>(DP::range<1>(tileSize), DP::range<1>(tileSize)), kfn);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template<typename scalar_t, class BinaryFunction>
typename std::enable_if<!IS_HALF(scalar_t), void>::type
scanDim(Tensor & self_, const Tensor & src_, int dimension,
    scalar_t init, BinaryFunction binary_op) {
  int ndim = src_.dim() == 0 ? 1 : src_.dim();
  TORCH_CHECK(dimension >= 0 && dimension < ndim,
      "dimension ", dimension, " out of range");

  self_.resize_as_(src_);
  auto self = self_.contiguous();
  auto src = src_.contiguous();

  if (ndim == 1) {
    // thrust does not take an "init"
    scanThrust<scalar_t>(self, src, binary_op);
  } else if (dimension == ndim - 1) {
    scanInnermostDim<scalar_t>(self, src, init, binary_op);
  } else {
    scanOuterDim<scalar_t>(self, src, dimension, init, binary_op);
  }

  self_.copy_(self);
}

template<typename scalar_t, class BinaryFunction>
typename std::enable_if<IS_HALF(scalar_t), void>::type
scanDim(Tensor & self_, const Tensor & src_, int dimension,
    scalar_t init, BinaryFunction binary_op) {
  int ndim = src_.dim() == 0 ? 1 : src_.dim();
  TORCH_CHECK(dimension >= 0 && dimension < ndim,
      "dimension ", dimension, " out of range");

  self_.resize_as_(src_);
  auto self = self_.contiguous();
  auto src = src_.contiguous();

  if (dimension == ndim - 1) {
    scanInnermostDim<scalar_t>(self, src, init, binary_op);
  } else {
    scanOuterDim<scalar_t>(self, src, dimension, init, binary_op);
  }

  self_.copy_(self);
}

} // namespace impl

Tensor & _cumsum_out(Tensor & out, const Tensor & self, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "cumsum", [&]() {
    impl::scanDim<scalar_t>(out, self, dim,
        ScalarConvert<float, scalar_t>::to(0.0), AddOp<scalar_t>());
  });
  return out;
}

Tensor _cumsum(const Tensor & self, int64_t dim) {
  auto out = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::_cumsum_out(out, self, dim);
}

Tensor & _cumprod_out(Tensor & out, const Tensor & self, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "cumprod", [&]() {
    impl::scanDim<scalar_t>(out, self, dim,
        ScalarConvert<float, scalar_t>::to(1.0), MulOp<scalar_t>());
  });
  return out;
}

Tensor _cumprod(const Tensor & self, int64_t dim) {
  auto out = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::_cumprod_out(out, self, dim);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
