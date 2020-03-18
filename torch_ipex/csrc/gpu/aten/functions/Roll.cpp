#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorTransformations.h>
#include <ATen/NativeFunctions.h>

#include <core/Memory.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>

#include <cstddef>
#include <vector>


using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
class roll_dpcpp_ker {};

template <typename scalar_t>
void roll_dpcpp_kernel(const Tensor& in_tensor, Tensor& out_tensor, int64_t N,
                      int64_t roll_dim, int64_t start,
                      int64_t size, int64_t stride, int64_t total_dims) {
  static const auto write_mode = DP::access::mode::discard_write;
  static const auto read_mode = DP::access::mode::read;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t rng, GRange, tileSize;
  auto offset = ((size - start) * stride);
  parallel_for_setup(N, tileSize, rng, GRange);
  dpcpp_queue.submit([&](DP::handler& cgh) {
    auto in_acc = DPCPPAccessor<read_mode>(cgh, in_tensor.data_ptr<scalar_t>());
    auto out_acc = DPCPPAccessor<write_mode>(cgh, out_tensor.data_ptr<scalar_t>());
    cgh.parallel_for<roll_dpcpp_ker<scalar_t>>(
        DP::nd_range<1>(DP::range<1>(GRange), DP::range<1>(tileSize)),
        [=](DP::nd_item<1> item) {
          int64_t linear_index = item.get_global_id(0);
          auto in_ptr = in_acc.template get_pointer<scalar_t>();
          auto out_ptr = out_acc.template get_pointer<scalar_t>();
          if (linear_index < N) {
            // roll dim idx is the index of linear_index along the rolling dimension.
            int64_t roll_dim_idx = linear_index % (stride * size) / stride;
            // index into the source data to find appropriate value.
            int64_t source_idx = 0;
            if ( roll_dim_idx >= (size - start) ) {
              source_idx = linear_index - offset;
            } else {
              source_idx = linear_index + (start * stride);
            }
            out_ptr[linear_index] = in_ptr[source_idx];
          }
        });
  });
}

// Roll a tensor along a dimension
Tensor roll_dpcpp(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }

  auto in_tensor = self;
  if (!self.is_contiguous()) {
    in_tensor = self.contiguous();
  }
  auto out_tensor = at::empty_like(in_tensor);
  if (out_tensor.numel() == 0) {
    return out_tensor;
  }
  const int64_t N = in_tensor.numel();
  const int64_t dim = dims[0];
  const int64_t size = in_tensor.size(dim);
  int64_t start = (size - shifts[0]) % size;
  if (start < 0) start += size;

  auto total_dims = in_tensor.dim();
  AT_DISPATCH_FLOATING_TYPES(in_tensor.scalar_type(), "roll_dpcpp", [&] {
    roll_dpcpp_kernel<scalar_t>(in_tensor, out_tensor, N,
        dim, start, size, in_tensor.stride(dim), total_dims);
  });
  return out_tensor;
}

} // namespace impl

Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims){
  return impl::roll_dpcpp(self, shifts, dims);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
