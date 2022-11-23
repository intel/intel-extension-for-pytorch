#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/TensorTransformations.h>
#include <core/detail/OffsetCalculator.h>
#include "Loops.h"
#include "MemoryAccess.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

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

template <typename func_t>
void _elementwise_kernel(int total_n_elems, func_t f) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  const auto target_global_size = dpcppMaxWorkItemsPerTile(dev_id);
  int work_group_size =
      total_n_elems > max_wg_size ? max_wg_size : total_n_elems;
  const int max_work_group_num = target_global_size / work_group_size;
  int total_group_num = (total_n_elems + work_group_size - 1) / work_group_size;
  int work_group_num = total_group_num < max_work_group_num
      ? total_group_num
      : max_work_group_num;
  // work item in each work group calculates loops' elements
  int loops = total_group_num / work_group_num + 1;

  int total_work_items = work_group_size * work_group_num;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_work_items), sycl::range<1>(work_group_size)),
        [=](sycl::nd_item<1> itemId) {
          int idx = itemId.get_global_linear_id();

          for (int i = 0; i < loops; ++i) {
            if (idx < total_n_elems) {
              f(idx);
              idx += total_work_items;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
static void _launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());
  _elementwise_kernel<func_t>(total_n_elems, f);
}

template <typename scalar_t>
void flip_kernel_impl(TensorIterator& iter) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      flip_kernel_impl<scalar_t>(sub_iter);
    }
    return;
  }

  char* const __restrict__ out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ in_ptr =
      reinterpret_cast<const char*>(iter.data_ptr(1));

  const auto offset_calc =
      make_offset_calculator<2, /*signed_strides=*/true>(iter);

  auto loop = [=](const int i) {
    const auto offsets = offset_calc.get(i);
    // offsets can be negative here, but it's fine
    scalar_t* const __restrict__ out_data =
        reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    const scalar_t* const __restrict__ in_data =
        reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
    *out_data = *in_data;
  };
  _launch_kernel(iter.numel(), loop);
}

Tensor flip_dpcpp(const Tensor& self, IntArrayRef& dims) {
  const int64_t total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  Tensor out_tensor = at::empty_like(self, MemoryFormat::Preserve);

  // Count dimensions in which we need to do work
  int n = 0;
  auto strides = DimVector(self.strides());
  for (int64_t i = 0; i < total_dims; i++) {
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      n++;
      strides[i] = 0;
    }
  }

  // Nothing to do, we return fast
  if (n == 0 || self.numel() <= 1) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  // create dummy output with 0 strides at flipped dimension, to prevent
  // tensorIterator from coalescing flipped dims
  const auto restrided_self = self.as_strided(self.sizes(), strides);
  auto iter =
      TensorIteratorConfig()
          .set_check_mem_overlap(false)
          .check_all_same_dtype(false)
          .declare_static_dtype_and_device(self.scalar_type(), self.device())
          .add_output(out_tensor)
          .add_input(self)
          .add_input(restrided_self)
          .build();

  auto* data = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto sizes = iter.shape();
  // This is a SmallVector of _signed_ ints
  auto strides_bytes = DimVector(iter.strides(0));
  const auto strides_self = iter.strides(1);
  const auto strides_dummy = iter.strides(2);

  // To understand this transformation, think of a 3D cube.
  //   - The data ptr points to the lower-left most vertex of the cube
  //   - The strides tell us how to move in each dimension,
  //     that is, data + stride[i] advances one element in the dimension i
  // To flip a dimension:
  //   - We move the pointer to the opposite vertex of the cube
  //   - We iterate in the opposite direction (invert the strides)
  for (int i = 0; i < iter.ndim(); i++) {
    // We know that an dimension has a zero stride and self[i] does not, as we
    // defined above Note that it may be the case that strides_dummy[i] = 0
    // not because we set it, but because strides_self[i] == 0. We do not want
    // to do anything there
    if (strides_dummy[i] == 0 && strides_self[i] != 0) {
      data += strides_bytes[i] * (sizes[i] - 1);
      strides_bytes[i] *= -1;
    }
  }
  iter._unsafe_set_arg_strides(0, strides_bytes);
  iter._unsafe_set_arg_data(0, reinterpret_cast<void*>(data));

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "flip_xpu",
      [&] {
        using dtype = typename native::Memory::aligned_element<sizeof(
            scalar_t)>::element_type;
        flip_kernel_impl<dtype>(iter);
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
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto shift = size - start;
  auto offset = shift * stride;
  auto start_offset = start * stride;
  auto total_offset = size * stride;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_range = dpcppMaxWorkGroupSize(dev_id);
  const auto target_global_range =
      dpcppMaxWorkItemsPerTile(dev_id) / local_range * local_range;
  int global_range = (N + local_range - 1) / local_range * local_range;
  auto val_of_work_item =
      (global_range + target_global_range - 1) / target_global_range;
  global_range =
      global_range < target_global_range ? global_range : target_global_range;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = in_tensor.data_ptr<scalar_t>();
    auto out_data = out_tensor.data_ptr<scalar_t>();
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        [=](sycl::nd_item<1> item) {
          int64_t linear_index = item.get_global_id(0);
          for (int i = 0; i < val_of_work_item; i++) {
            if (linear_index < N) {
              // roll dim idx is the index of linear_index along the rolling
              // dimension.
              int64_t roll_dim_idx = linear_index % (total_offset) / stride;
              // index into the source data to find appropriate value.
              int64_t source_idx = 0;
              source_idx = roll_dim_idx >= shift ? linear_index - offset
                                                 : linear_index + start_offset;
              out_data[linear_index] = in_data[source_idx];
              linear_index += global_range;
            }
          }
        });
  };
  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
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
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      in_tensor.scalar_type(),
      "roll_dpcpp",
      [&] {
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
