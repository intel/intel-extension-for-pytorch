#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
namespace at {
namespace AtenIpexTypeXPU {
using namespace torch_ipex::xpu::dpcpp;

#define XPU_DEVICE_GUARD(TENSOR) \
  const OptionalDeviceGuard device_guard(device_of(TENSOR));

Tensor asynchronous_complete_cumsum_xpu(const Tensor& t_in) {
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  TORCH_CHECK(t_in.dim() == 1 || t_in.dim() == 2);
  Tensor t_out;
  if (t_in.dim() == 1) {
    t_out = at::zeros({t_in.numel() + 1}, t_in.options());
    auto r_out = t_out.slice(0, 1);
    at::cumsum_out(r_out, t_in, 0);
  } else {
    t_out = at::zeros({t_in.size(0), t_in.size(1) + 1}, t_in.options());
    auto r_out = t_out.slice(1, 1);
    at::cumsum_out(r_out, t_in, 1);
  }
  return t_out;
}

constexpr size_t kStackArrayMaxDims = 5;

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
struct DenseToJaggedKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto output_values_acc = output_values_;
    const int outer_dense_size = y_0_.size(0);
    const int inner_dense_size = y_0_.size(2);
    const int nnz = x_values_.size(0);

    const int offset_begin =
        item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int offset_stride =
        item.get_global_range(0) * item.get_local_range(1);
    for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
      int offset_temp = offset;
      int jidx = 0;
      bool truncated = false;
      int dim_prod = 1;
#pragma unroll
      for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
        // Binary search the first that is bigger than offset
        int count = x_offsets_sizes_.vals[d] - 1;
        int first = 1;
        while (count > 0) {
          int idx = first;
          int step = count / 2;
          idx += step;
          if (x_offsets_.vals[d][idx] <= offset_temp) {
            first = ++idx;
            count -= step + 1;
          } else {
            count = step;
          }
        }

        --first;
        int coord = offset_temp - x_offsets_.vals[d][first];
        if (coord >= jagged_dims_.vals[d]) {
          truncated = true;
          break;
        }
        jidx += coord * dim_prod;
        dim_prod *= jagged_dims_.vals[d];
        offset_temp = first;
      }

      if (offset_temp >= outer_dense_size) {
        // This can happen when values have more elements than the last element
        // of offset
        truncated = true;
      }
      if (!truncated) {
        const int oidx = offset_temp;
        int iidx;
        for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
             iidx += item.get_local_range(0)) {
          output_values_acc[offset][2 * iidx] =
              f_(x_values_[offset][2 * iidx],
                 y_0_[oidx][jidx][2 * iidx],
                 y_1_[oidx][jidx][2 * iidx]);
          output_values_acc[offset][2 * iidx + 1] =
              f_(x_values_[offset][2 * iidx + 1],
                 y_0_[oidx][jidx][2 * iidx + 1],
                 y_1_[oidx][jidx][2 * iidx + 1]);
        }
        if (iidx * 2 + 1 == inner_dense_size) {
          output_values_acc[offset][2 * iidx] =
              f_(x_values_[offset][2 * iidx],
                 y_0_[oidx][jidx][2 * iidx],
                 y_1_[oidx][jidx][2 * iidx]);
        }
      } else {
        int iidx;
        for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
             iidx += item.get_local_range(0)) {
          output_values_acc[offset][2 * iidx] =
              f_(x_values_[offset][2 * iidx], 0, 0);
          output_values_acc[offset][2 * iidx + 1] =
              f_(x_values_[offset][2 * iidx + 1], 0, 0);
        }
        if (iidx * 2 + 1 == inner_dense_size) {
          output_values_acc[offset][2 * iidx] =
              f_(x_values_[offset][2 * iidx], 0, 0);
        }
      }
    }
  }

  DenseToJaggedKernelFunctor(
      PackedTensorAccessor32<scalar_t, 2> x_values,
      StackArray<index_t*> x_offsets,
      StackArray<int64_t> x_offsets_sizes,
      const PackedTensorAccessor32<const scalar_t, 3> y_0,
      const PackedTensorAccessor32<const scalar_t, 3> y_1,
      PackedTensorAccessor32<scalar_t, 2> output_values,
      StackArray<int64_t> jagged_dims,
      F f)
      : x_values_(x_values),
        x_offsets_(x_offsets),
        x_offsets_sizes_(x_offsets_sizes),
        y_0_(y_0),
        y_1_(y_1),
        output_values_(output_values),
        jagged_dims_(jagged_dims),
        f_(f) {}

 private:
  PackedTensorAccessor32<scalar_t, 2> x_values_;
  StackArray<index_t*> x_offsets_;
  StackArray<int64_t> x_offsets_sizes_;
  const PackedTensorAccessor32<const scalar_t, 3> y_0_;
  const PackedTensorAccessor32<const scalar_t, 3> y_1_;
  PackedTensorAccessor32<scalar_t, 2> output_values_;
  StackArray<int64_t> jagged_dims_;
  F f_;
};

inline std::tuple<int64_t, int64_t, int64_t, StackArray<int64_t>>
check_shape_and_partition_(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_tensor) {
  const int32_t outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(
      outer_dense_size == offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != offsets[0].numel() - 1, ",
      offsets[0].numel() - 1);
  const int32_t inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(
      inner_dense_size == values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != values.size(-1), ",
      values.size(-1));
  const int32_t jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);

  const int32_t sub_group_size = dpcppMaxSubGroupSize();
  const int64_t wg_size_0 = inner_dense_size >= sub_group_size / 2
      ? sub_group_size
      : inner_dense_size;
  const int64_t wg_size_1 = dpcppMaxWorkGroupSize() / sub_group_size;
  const int64_t wg_num =
      CeilDiv(outer_dense_size * jagged_folded_size, (int32_t)wg_size_1);

  StackArray<int64_t> jagged_dims_tensor;
  const int32_t num_jagged_dim = dense_tensor.dim() - 2;
  TORCH_CHECK(num_jagged_dim <= kStackArrayMaxDims);
  jagged_dims_tensor.ndim = num_jagged_dim;
  std::memcpy(
      &(jagged_dims_tensor.vals[0]),
      dense_tensor.sizes().data() + 1,
      num_jagged_dim * sizeof(int64_t));
  return {wg_size_0, wg_size_1, wg_num, jagged_dims_tensor};
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
void jagged_dense_dense_elementwise_jagged_output_kernel_(
    PackedTensorAccessor32<scalar_t, 2> x_values, // output
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    const PackedTensorAccessor32<const scalar_t, 3> y_0, // not used
    const PackedTensorAccessor32<const scalar_t, 3> y_1,
    PackedTensorAccessor32<scalar_t, 2> output_values, // not used
    StackArray<int64_t> jagged_dims,
    F f,
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    DenseToJaggedKernelFunctor<NUM_JAGGED_DIM, index_t, scalar_t, F> kfn(
        x_values,
        x_offsets,
        x_offsets_sizes,
        y_0,
        y_1,
        output_values,
        jagged_dims,
        f);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<2>(
            sycl::range<2>(wg_0 * wg_num, wg_1), sycl::range<2>(wg_0, wg_1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

#define JAGGED_TENSOR_DISPATCH_DIMS()                                         \
  AT_DISPATCH_INDEX_TYPES(x_offsets[0].scalar_type(), "jagged_indices", [=] { \
    switch (num_jagged_dim) {                                                 \
      case 1:                                                                 \
        INVOKE_KERNEL_WITH_DIM(1);                                            \
        break;                                                                \
      case 2:                                                                 \
        INVOKE_KERNEL_WITH_DIM(2);                                            \
        break;                                                                \
      case 3:                                                                 \
        INVOKE_KERNEL_WITH_DIM(3);                                            \
        break;                                                                \
      case 4:                                                                 \
        INVOKE_KERNEL_WITH_DIM(4);                                            \
        break;                                                                \
      case 5:                                                                 \
        INVOKE_KERNEL_WITH_DIM(5);                                            \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false, "unsupported number of jagged dim ", num_jagged_dim);      \
    }                                                                         \
  });

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TORCH_CHECK(x_values.is_xpu(), "value must be a xpu tensor");
  for (auto& x_offset : x_offsets) {
    TORCH_CHECK(x_offset.is_xpu(), "offset must be a xpu tensor");
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                         \
  {                                                                    \
    int64_t wg_0, wg_1, wg_num;                                        \
    StackArray<int64_t> jagged_dims_tensor;                            \
    std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =                 \
        check_shape_and_partition_(x_values, x_offsets, y);            \
    wg_num = CeilDiv(x_values.size(0), wg_1);                          \
    std::vector<Tensor> x_offsets_contig;                              \
    x_offsets_contig.resize(num_jagged_dim);                           \
    StackArray<index_t*> x_offset_ptrs;                                \
    x_offset_ptrs.ndim = num_jagged_dim;                               \
    StackArray<int64_t> x_offset_sizes;                                \
    x_offset_sizes.ndim = num_jagged_dim;                              \
    for (int d = 0; d < num_jagged_dim; ++d) {                         \
      x_offsets_contig[d] = x_offsets[d].contiguous();                 \
      x_offset_ptrs.vals[d] =                                          \
          x_offsets_contig[d].template data_ptr<index_t>();            \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                   \
    }                                                                  \
    jagged_dense_dense_elementwise_jagged_output_kernel_<              \
        NUM_JAGGED_DIM,                                                \
        index_t>(                                                      \
        x_values.packed_accessor32<scalar_t, 2>(),                     \
        x_offset_ptrs,                                                 \
        x_offset_sizes,                                                \
        y_reshaped.packed_accessor32<const scalar_t, 3>(),             \
        y_reshaped.packed_accessor32<const scalar_t, 3>(),             \
        output_values.packed_accessor32<scalar_t, 2>(),                \
        jagged_dims_tensor,                                            \
        [f](scalar_t x, scalar_t y, scalar_t /*unused*/) -> scalar_t { \
          return f(x, y);                                              \
        },                                                             \
        wg_0,                                                          \
        wg_1,                                                          \
        wg_num);                                                       \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
#undef INVOKE_KERNEL_WITH_DIM
}

Tensor dense_to_jagged_forward_xpu(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  at::SymInt total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty_symint({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values); // not used

  XPU_DEVICE_GUARD(dense);

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      values.scalar_type(),
      "dense_to_jagged_gpu_op_forward",
      [&]() {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            values,
            offsets,
            dense,
            output,
            [](scalar_t /*unused*/, scalar_t y) -> scalar_t { return y; });
      });
  return output;
}

template <int NUM_JAGGED_DIM, typename index_t>
bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<index_t*>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets.vals[d][offset];
    const int end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
struct JaggedToDenseKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto output_acc = output_;
    const int outer_dense_size = y_.size(0);
    const int jagged_folded_size = y_.size(1);
    const int inner_dense_size = y_.size(2);

    const int outer_begin =
        item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int outer_stride = item.get_global_range(0) * item.get_local_range(1);
    for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
         outer += outer_stride) {
      const int oidx = outer / jagged_folded_size;
      const int jidx = outer % jagged_folded_size;

      int offset = oidx;
      const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
          offset, jidx, jagged_dims_, x_offsets_);

      if (is_zero) {
        int iidx;
        for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
             iidx += item.get_local_range(0)) {
          output_acc[oidx][jidx][2 * iidx] =
              f_(padding_value_, y_[oidx][jidx][2 * iidx]);
          output_acc[oidx][jidx][2 * iidx + 1] =
              f_(padding_value_, y_[oidx][jidx][2 * iidx + 1]);
        }
        if (iidx * 2 + 1 == inner_dense_size) {
          output_acc[oidx][jidx][2 * iidx] =
              f_(padding_value_, y_[oidx][jidx][2 * iidx]);
        }
      } else {
        int iidx;
        for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
             iidx += item.get_local_range(0)) {
          output_acc[oidx][jidx][2 * iidx] =
              f_(x_values_[offset][2 * iidx], y_[oidx][jidx][2 * iidx]);
          output_acc[oidx][jidx][2 * iidx + 1] =
              f_(x_values_[offset][2 * iidx + 1], y_[oidx][jidx][2 * iidx + 1]);
        }
        if (iidx * 2 + 1 == inner_dense_size) {
          output_acc[oidx][jidx][2 * iidx] =
              f_(x_values_[offset][2 * iidx], y_[oidx][jidx][2 * iidx]);
        }
      }
    }
  }

  JaggedToDenseKernelFunctor(
      const PackedTensorAccessor32<const scalar_t, 2> x_values,
      StackArray<index_t*> x_offsets,
      PackedTensorAccessor32<scalar_t, 3> y,
      PackedTensorAccessor32<scalar_t, 3> output,
      StackArray<int64_t> jagged_dims,
      F f,
      const scalar_t padding_value)
      : x_values_(x_values),
        x_offsets_(x_offsets),
        y_(y),
        output_(output),
        jagged_dims_(jagged_dims),
        f_(f),
        padding_value_(padding_value) {}

 private:
  const PackedTensorAccessor32<const scalar_t, 2> x_values_;
  StackArray<index_t*> x_offsets_;
  PackedTensorAccessor32<scalar_t, 3> y_;
  PackedTensorAccessor32<scalar_t, 3> output_;
  StackArray<int64_t> jagged_dims_;
  F f_;
  const scalar_t padding_value_;
};

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_kernel_(
    const PackedTensorAccessor32<const scalar_t, 2> x_values,
    StackArray<index_t*> x_offsets,
    PackedTensorAccessor32<scalar_t, 3> y,
    PackedTensorAccessor32<scalar_t, 3> output,
    StackArray<int64_t> jagged_dims,
    F f,
    const scalar_t padding_value,
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    JaggedToDenseKernelFunctor<NUM_JAGGED_DIM, index_t, scalar_t, F> kfn(
        x_values, x_offsets, y, output, jagged_dims, f, padding_value);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<2>(
            sycl::range<2>(wg_0 * wg_num, wg_1), sycl::range<2>(wg_0, wg_1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TORCH_CHECK(x_values.is_xpu(), "value must be a xpu tensor");
  for (auto& x_offset : x_offsets) {
    TORCH_CHECK(x_offset.is_xpu(), "offset must be a xpu tensor");
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim ",
      num_jagged_dim);

  if (y.numel() == 0) {
    return;
  }

  int64_t wg_0, wg_1, wg_num;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                              \
  {                                                                         \
    std::vector<Tensor> x_offsets_contig;                                   \
    x_offsets_contig.resize(num_jagged_dim);                                \
    StackArray<index_t*> x_offset_ptrs;                                     \
    x_offset_ptrs.ndim = num_jagged_dim;                                    \
    for (int d = 0; d < num_jagged_dim; ++d) {                              \
      x_offsets_contig[d] = x_offsets[d].contiguous();                      \
      x_offset_ptrs.vals[d] =                                               \
          x_offsets_contig[d].template data_ptr<index_t>();                 \
    }                                                                       \
    jagged_dense_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>( \
        x_values.packed_accessor32<const scalar_t, 2>(),                    \
        x_offset_ptrs,                                                      \
        y_reshaped.packed_accessor32<scalar_t, 3>(),                        \
        output_reshaped.packed_accessor32<scalar_t, 3>(),                   \
        jagged_dims_tensor,                                                 \
        f,                                                                  \
        padding_value,                                                      \
        wg_0,                                                               \
        wg_1,                                                               \
        wg_num);                                                            \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
#undef INVOKE_KERNEL_WITH_DIM

} // namespace AtenIpexTypeXPU

Tensor jagged_to_padded_dense_forward_xpu(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    c10::SymIntArrayRef max_lengths,
    const double padding_value) {
  const size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);
  XPU_DEVICE_GUARD(values);

  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      values.scalar_type(),
      "jagged_to_padded_dense",
      [&] {
        jagged_dense_elementwise_dense_output_<scalar_t>(
            values_canonicalized,
            offsets,
            padded_values_view, // not used
            padded_values_view,
            [](scalar_t x, scalar_t /*unused*/) -> scalar_t { return x; },
            static_cast<scalar_t>(padding_value));
      });

  return padded_values;
}

class DenseToJaggedOp : public torch::autograd::Function<DenseToJaggedOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& dense,
      const std::vector<Tensor>& offsets,
      const std::optional<at::SymInt>& total_L) {
    ctx->save_for_backward(offsets);

    // dims of dense tensor: <batch, [maxlen0, maxlen1, ...], embedding_dim>
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 1)
    // toSymIntVector support is from a recent PR
    // https://github.com/pytorch/pytorch/pull/101056,
    // so protect it under a version guard for compatibility
    ctx->saved_data["dense_shape"] = dense.sym_sizes();
#else
    ctx->saved_data["dense_shape"] = dense.sizes();
#endif

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::dense_to_jagged_forward", "")
            .typed<Tensor(
                const Tensor& dense,
                const std::vector<Tensor>& offsets,
                std::optional<at::SymInt> total_L)>();
    auto output = op.call(dense, offsets, total_L);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    // TODO: backward kernel
    return {
        torch::autograd::Variable(),
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable() // total_L
    };
  }
};

// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>> dense_to_jagged(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
  return {DenseToJaggedOp::apply(dense, offsets, total_L)[0], offsets};
}

class JaggedToPaddedDenseOp
    : public torch::autograd::Function<JaggedToPaddedDenseOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const std::vector<Tensor>& offsets,
      const c10::SymIntArrayRef max_lengths,
      const double padding_value) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["total_L"] = values.sym_size(0);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<at::Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                at::ArrayRef<at::SymInt> max_lengths,
                const double padding_value)>();
    Tensor padded_values = op.call(values, offsets, max_lengths, padding_value);

    return {padded_values};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    // TODO: backward kernel
    return {
        torch::autograd::Variable(),
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // max_lengths
        torch::autograd::Variable(), // padding_value
    };
  }
};

Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const c10::SymIntArrayRef max_lengths,
    const double padding_value) {
  return JaggedToPaddedDenseOp::apply(
      values, offsets, max_lengths, padding_value)[0];
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "asynchronous_complete_cumsum",
      torch::dispatch(
          c10::DispatchKey::XPU,
          TORCH_FN(at::AtenIpexTypeXPU::asynchronous_complete_cumsum_xpu)));
}

// Autograd backend register in fbgemm
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "dense_to_jagged",
      torch::dispatch(
          c10::DispatchKey::XPU,
          TORCH_FN(at::AtenIpexTypeXPU::dense_to_jagged)));
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "dense_to_jagged_forward",
      torch::dispatch(
          c10::DispatchKey::XPU,
          TORCH_FN(at::AtenIpexTypeXPU::dense_to_jagged_forward_xpu)));
}

// Autograd backend register in fbgemm
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "jagged_to_padded_dense",
      torch::dispatch(
          c10::DispatchKey::XPU,
          TORCH_FN(at::AtenIpexTypeXPU::jagged_to_padded_dense)));
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "jagged_to_padded_dense_forward",
      torch::dispatch(
          c10::DispatchKey::XPU,
          TORCH_FN(at::AtenIpexTypeXPU::jagged_to_padded_dense_forward_xpu)));
}
} // namespace
