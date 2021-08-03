#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/TensorTransformations.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/Helpers.h>

#include <cstddef>
#include <vector>

using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

constexpr size_t dim_bitset_size = 64;

template <typename scalar_t>
class flip_dpcpp_ker {};
template <typename scalar_t>
class roll_dpcpp_ker {};

template <typename scalar_t>
void flip_dpcpp_kernel(
    const Tensor& in_tensor,
    Tensor& out_tensor,
    const int64_t total_dims,
    const std::vector<int64_t>& stride_contiguous_v,
    const std::bitset<dim_bitset_size>& flip_dims_b) {
  const std::vector<int64_t>& sizes_v = in_tensor.sizes().vec();
  const std::vector<int64_t>& strides_v = in_tensor.strides().vec();

  Tensor stride_contiguous_t = at::empty(
      {static_cast<int64_t>(stride_contiguous_v.size())},
      in_tensor.options().dtype(at::ScalarType::Long));
  Tensor sizes_t = at::empty(
      {static_cast<int64_t>(sizes_v.size())},
      in_tensor.options().dtype(at::ScalarType::Long));
  Tensor strides_t = at::empty(
      {static_cast<int64_t>(strides_v.size())},
      in_tensor.options().dtype(at::ScalarType::Long));

  for (int64_t i = 0; i < stride_contiguous_v.size(); i++)
    stride_contiguous_t[i] = stride_contiguous_v[i];
  for (int64_t i = 0; i < sizes_v.size(); i++)
    sizes_t[i] = sizes_v[i];
  for (int64_t i = 0; i < strides_v.size(); i++)
    strides_t[i] = strides_v[i];

  const int64_t N = in_tensor.numel();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(N, tileSize, rng, GRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_tensor_d = in_tensor.data_ptr<scalar_t>();
    auto out_tensor_d = out_tensor.data_ptr<scalar_t>();
    auto stride_contiguous_d = stride_contiguous_t.data_ptr<int64_t>();
    auto sizes_d = sizes_t.data_ptr<int64_t>();
    auto strides_d = strides_t.data_ptr<int64_t>();

    // auto local_output_data = dpcpp_local_acc_t<scalar_t>(1024, cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto in_tensor_ptr = in_tensor_d;
      auto out_tensor_ptr = out_tensor_d;
      auto stride_contiguous_ptr = stride_contiguous_d;
      auto sizes_ptr = sizes_d;
      auto strides_ptr = strides_d;
      auto linear_index = item.get_global_id(0);

      int64_t cur_indices = linear_index;
      int64_t rem = 0;
      int64_t dst_offset = 0;

      for (int64_t d = 0; d < total_dims; d++) {
        int64_t temp = cur_indices;
        cur_indices = cur_indices / stride_contiguous_ptr[d];
        rem = temp - cur_indices * stride_contiguous_ptr[d];
        dst_offset += flip_dims_b[d]
            ? (sizes_ptr[d] - 1 - cur_indices) * strides_ptr[d]
            : cur_indices * strides_ptr[d];
        cur_indices = rem;
      }
      out_tensor_ptr[linear_index] = in_tensor_ptr[dst_offset];
    };
    cgh.parallel_for<flip_dpcpp_ker<scalar_t>>(
        DPCPP::nd_range<1>(GRange, tileSize), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

Tensor flip_dpcpp(const Tensor& self, IntArrayRef& dims) {
  auto in_tensor = self;
  const int64_t total_dims = in_tensor.dim();
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);
  Tensor out_tensor =
      at::empty_like(in_tensor, in_tensor.contiguous().options());

  // create contiguous strides for input tensor
  auto stride_contiguous_v = std::vector<int64_t>(total_dims);
  for (int64_t i = total_dims - 1; i >= 0; i--) {
    if (i == total_dims - 1)
      stride_contiguous_v[i] = 1;
    else
      stride_contiguous_v[i] =
          Max<int64_t>(in_tensor.size(i + 1), 1) * stride_contiguous_v[i + 1];
  }

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, in_tensor.scalar_type(), "flip_dpcpp", [&] {
        flip_dpcpp_kernel<scalar_t>(
            in_tensor,
            out_tensor,
            total_dims,
            stride_contiguous_v,
            flip_dims_b);
      });

  return out_tensor;
}

template <typename scalar_t>
void roll_dpcpp_kernel(
    const Tensor& in_tensor,
    Tensor& out_tensor,
    int64_t N,
    int64_t roll_dim,
    int64_t start,
    int64_t size,
    int64_t stride,
    int64_t total_dims) {
  static const auto write_mode = DPCPP::access::mode::discard_write;
  static const auto read_mode = DPCPP::access::mode::read;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t rng, GRange, tileSize;
  auto offset = ((size - start) * stride);
  parallel_for_setup(N, tileSize, rng, GRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = in_tensor.data_ptr<scalar_t>();
    auto out_data = out_tensor.data_ptr<scalar_t>();
    cgh.parallel_for<roll_dpcpp_ker<scalar_t>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(GRange), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          int64_t linear_index = item.get_global_id(0);
          auto in_ptr = in_data;
          auto out_ptr = out_data;
          if (linear_index < N) {
            // roll dim idx is the index of linear_index along the rolling
            // dimension.
            int64_t roll_dim_idx = linear_index % (stride * size) / stride;
            // index into the source data to find appropriate value.
            int64_t source_idx = 0;
            if (roll_dim_idx >= (size - start)) {
              source_idx = linear_index - offset;
            } else {
              source_idx = linear_index + (start * stride);
            }
            out_ptr[linear_index] = in_ptr[source_idx];
          }
        });
  };
  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

// Roll a tensor along a dimension
Tensor roll_dpcpp(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }

  auto in_tensor = self.contiguous();
  auto out_tensor = at::empty_like(in_tensor);
  if (out_tensor.numel() == 0) {
    return out_tensor;
  }
  const int64_t N = in_tensor.numel();
  const int64_t dim = dims[0];
  const int64_t size = in_tensor.size(dim);
  int64_t start = (size - shifts[0]) % size;
  if (start < 0)
    start += size;

  auto total_dims = in_tensor.dim();
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, in_tensor.scalar_type(), "roll_dpcpp", [&] {
        roll_dpcpp_kernel<scalar_t>(
            in_tensor,
            out_tensor,
            N,
            dim,
            start,
            size,
            in_tensor.stride(dim),
            total_dims);
      });
  return out_tensor;
}

} // namespace impl

Tensor flip(const Tensor& self, IntArrayRef dims) {
  return impl::flip_dpcpp(self, dims);
}

Tensor roll(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  return impl::roll_dpcpp(self, shifts, dims);
}

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) {
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size();

  TORCH_CHECK(
      total_rot_dims == 2,
      "expected total rotation dims == 2, but got dims = ",
      total_rot_dims);

  TORCH_CHECK(
      total_dims >= 2,
      "expected total dims >= 2, but got total dims = ",
      total_dims);

  TORCH_CHECK(
      dims[0] != dims[1] &&
          Numerics<int64_t>::abs(dims[0] - dims[1]) != total_dims,
      "expected rotation dims to be different, but got dim0 = ",
      dims[0],
      " and dim1 = ",
      dims[1]);

  // check range of dims
  TORCH_CHECK(
      dims[0] < total_dims && dims[0] >= -total_dims,
      "Rotation dim0 out of range, dim0 = ",
      dims[0]);

  TORCH_CHECK(
      dims[1] < total_dims && dims[1] >= -total_dims,
      "Rotation dim0 out of range, dim0 = ",
      dims[1]);

  // handle modulo with negative k
  k = (4 + (k % 4)) % 4;

  switch (k) {
    case 1:
      return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
    case 2:
      return self.flip(dims);
    case 3:
      return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
    default:
      return self.clone(MemoryFormat::Contiguous);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
