#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <core/detail/OffsetCalculator.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "ReduceOpsUtils.h"

// Note on naming: it is unconventional.
// grad_in does not mean that it is a gradient wrt to input,
// grad_in/grad_out is just an input/output of unfold_backward kernel.
//
// unfold_backward, the algorithm is described in
// /native/cpu/UnfoldBackwardKernel.cpp

constexpr int n_elems_per_work_item = UNROLLED_ELEM_PER_WORK_ITEM;

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace {

static TensorIterator _make_unfold_backward_iter_over_grad_out(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds

  auto grad_out_dim_size = ensure_nonempty_size(grad_out, dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);
  // dictates the number of elements to iterate over in dimension `dim`
  auto iter_dim_size =
      std::min(grad_out_dim_size, (grad_in_dim_size - 1) * step + size);

  /* prepare grad_out for TensorIterator { */
  auto grad_out_strides = ensure_nonempty_vec(grad_out.strides().vec());
  auto grad_out_sizes = ensure_nonempty_vec(grad_out.sizes().vec());
  grad_out_sizes[dim] = iter_dim_size;
  auto grad_out_restrided =
      grad_out.as_strided(grad_out_sizes, grad_out_strides);
  /* } */

  /* prepare grad_in for TensorIterator { */
  auto grad_in_strides = ensure_nonempty_vec(grad_in.strides().vec());
  auto grad_in_sizes = ensure_nonempty_vec(grad_in.sizes().vec());

  // set strides for dim to 0
  // and size to 1 because this dimension is indexed inside the kernel
  grad_in_strides[dim] = 0;
  grad_in_sizes[dim] = 1;

  grad_in_strides.pop_back();
  grad_in_sizes.pop_back();

  auto grad_in_restrided =
      grad_in.squeeze(-1).as_strided(grad_in_sizes, grad_in_strides);
  /* } */

  // During the TensorIterator iteration we have to know
  // i_dim in grad_out[i_1,...,i_dim,...i_n],
  // idx_dim stores this information
  /* prepare idx_dim for TensorIterator { */
  auto idx_dim =
      at::arange(0, iter_dim_size, grad_in.options().dtype(at::kLong));

  auto grad_out_dim = ensure_nonempty_dim(grad_out.dim());

  auto idx_dim_strides = std::vector<int64_t>(grad_out_dim, 0);
  auto idx_dim_sizes = std::vector<int64_t>(grad_out_dim, 1);

  idx_dim_strides[dim] = 1;
  idx_dim_sizes[dim] = iter_dim_size;

  // idx_dim size will broadcast over determined by grad_out sizes in
  // TensorIterator
  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
  /* } */

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_owned_output(grad_out_restrided)
                  .add_owned_input(grad_in_restrided)
                  .add_owned_input(idx_dim_restrided)
                  .build();

  return iter;
}

static TensorIterator _make_unfold_backward_iter_over_grad_in(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim = ensure_nonempty_dim(grad_in.dim());
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);
  auto grad_in_last_dim_size = ensure_nonempty_size(grad_in, last_dim);

  /* prepare grad_out for TensorIterator { */
  auto grad_out_restrided = grad_out.unsqueeze(-1);

  auto grad_out_strides =
      ensure_nonempty_vec(grad_out_restrided.strides().vec());
  auto grad_out_sizes = ensure_nonempty_vec(grad_out_restrided.sizes().vec());

  grad_out_strides[dim] = 0;
  grad_out_strides[last_dim] = 0;

  grad_out_sizes[dim] = grad_in_dim_size;
  grad_out_sizes[last_dim] = grad_in_last_dim_size;

  grad_out_restrided =
      grad_out_restrided.as_strided(grad_out_sizes, grad_out_strides);
  /* } */

  // for each element grad_out[i_1,...,i_dim,...,i_last_dim]
  // we have to know i_dim and i_last_dim.
  // This information is stored in Tensors
  // idx_dim and idx_last_dim
  /* prepare idx_dim and idx_last_dim for TensorIterator { */
  auto idx_dim =
      at::arange(0, grad_in_dim_size, grad_in.options().dtype(at::kLong));

  auto idx_dim_strides = std::vector<int64_t>(grad_in_dim, 0);
  auto idx_dim_sizes = std::vector<int64_t>(grad_in_dim, 1);

  idx_dim_strides[dim] = 1;
  idx_dim_sizes[dim] = grad_in_dim_size;

  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);

  auto idx_last_dim =
      at::arange(0, grad_in_last_dim_size, grad_in.options().dtype(at::kLong));

  auto idx_last_dim_strides = std::vector<int64_t>(grad_in_dim, 0);
  auto idx_last_dim_sizes = std::vector<int64_t>(grad_in_dim, 1);

  idx_last_dim_strides[last_dim] = 1;
  idx_last_dim_sizes[last_dim] = grad_in_last_dim_size;

  auto idx_last_dim_restrided =
      idx_last_dim.as_strided(idx_last_dim_sizes, idx_last_dim_strides);
  /* } */

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_owned_output(grad_out_restrided)
                  .add_owned_input(grad_in)
                  .add_owned_input(idx_dim_restrided)
                  .add_owned_input(idx_last_dim_restrided)
                  .build();

  return iter;
}

} // namespace

template <int n_elems_per_work_item, typename func_t>
void _unfold_backward_elementwise_kernel(int total_n_elems, func_t f) {
  int total_work_items =
      (total_n_elems + n_elems_per_work_item - 1) / n_elems_per_work_item;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::range<1>(total_work_items), [=](sycl::item<1> itemId) {
          int idx = itemId.get_linear_id();
#pragma unroll
          for (int i = 0; i < n_elems_per_work_item; ++i) {
            if (idx < total_n_elems) {
              f(idx);
              idx += total_work_items;
            }
          }
        });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int n_elems_per_work_item, typename func_t>
static void _launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <=
          Numerics<int32_t>::upper_bound()); // INT_MAX when int32_t

  _unfold_backward_elementwise_kernel<n_elems_per_work_item, func_t>(
      total_n_elems, f);
}

template <typename scalar_t>
void _unfold_backward_internal_kernel(
    TensorIterator& iter,
    int64_t size,
    int64_t step,
    int64_t grad_in_dim_stride,
    int64_t grad_in_last_dim_stride,
    int64_t grad_in_dim_size,
    int64_t grad_out_dim_stride,
    bool is_step_ge_size) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unfold_backward_internal_kernel<scalar_t>(
          sub_iter,
          size,
          step,
          grad_in_dim_stride,
          grad_in_last_dim_stride,
          grad_in_dim_size,
          grad_out_dim_stride,
          is_step_ge_size);
    }
    return;
  }

  char* grad_out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* grad_in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* idx_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  if (is_step_ge_size) {
    char* idx_last_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(3));

    auto offset_calc = make_offset_calculator<4>(iter);

    // this loop simply copies the data
    // from proper places in grad_out to grad_in
    auto loop = [=](int i) {
      auto offsets = offset_calc.get(i);

      auto* grad_out_data =
          reinterpret_cast<scalar_t*>(grad_out_ptr + offsets[0]);
      auto* grad_in_data =
          reinterpret_cast<scalar_t*>(grad_in_ptr + offsets[1]);

      auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr + offsets[2]);
      auto idx_last_dim =
          *reinterpret_cast<int64_t*>(idx_last_dim_ptr + offsets[3]);

      auto grad_out_idx_dim = idx_dim * step + idx_last_dim;
      grad_out_data[grad_out_idx_dim * grad_out_dim_stride] = *grad_in_data;
    };

    _launch_unfold_backward_kernel<n_elems_per_work_item>(iter.numel(), loop);
  } else {
    auto offset_calc = make_offset_calculator<3>(iter);

    // The algorithm is: for each index in grad_out find
    // the elements contributing to it and sum them up.
    // Note: the algorithm does not require any synchronization.
    auto loop = [=](int i) {
      auto offsets = offset_calc.get(i);

      auto* grad_out_data =
          reinterpret_cast<scalar_t*>(grad_out_ptr + offsets[0]);
      auto* grad_in_data =
          reinterpret_cast<scalar_t*>(grad_in_ptr + offsets[1]);

      auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr + offsets[2]);

      // left_fold potentially intersecting with idx_dim
      // is either (idx_dim - size) / step or the next integer.
      int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
      if (!(left_fold_idx * step <= idx_dim &&
            idx_dim < left_fold_idx * step + size)) {
        ++left_fold_idx;
      }

      auto right_fold_idx = idx_dim / step;
      right_fold_idx = (right_fold_idx >= grad_in_dim_size)
          ? (grad_in_dim_size - 1)
          : right_fold_idx;

      for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx;
           ++fold_idx) {
        auto idx_last_dim = idx_dim - fold_idx * step;
        *grad_out_data += grad_in_data
            [fold_idx * grad_in_dim_stride +
             idx_last_dim * grad_in_last_dim_stride];
      }
    };

    _launch_unfold_backward_kernel<n_elems_per_work_item>(iter.numel(), loop);
  }
}

void unfold_backward_dpcpp(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  auto grad_out_dim_stride = ensure_nonempty_stride(grad_out, dim);

  auto is_step_ge_size = (step >= size);

  TensorIterator iter = is_step_ge_size
      ? _make_unfold_backward_iter_over_grad_in(
            grad_out, grad_in, dim, size, step)
      : _make_unfold_backward_iter_over_grad_out(
            grad_out, grad_in, dim, size, step);

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "unfold_backward_dpcpp",
      [&] {
        _unfold_backward_internal_kernel<scalar_t>(
            iter,
            size,
            step,
            grad_in_dim_stride,
            grad_in_last_dim_stride,
            grad_in_dim_size,
            grad_out_dim_stride,
            is_step_ge_size);
      });
}

Tensor unfold_backward(
    const Tensor& grad,
    IntArrayRef input_sizes,
    int64_t dim,
    int64_t size,
    int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad.options());

  unfold_backward_dpcpp(grad_input, grad, dim, size, step);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
