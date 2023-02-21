#include <ATen/ATen.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

enum SegmentReductionType { MAX, MEAN, MIN, SUM, PROD };

template <typename scalar_t, typename index_t>
// TODO: for optimize output 1D future.
static void post_sum_div_kernel(
    scalar_t* output_data,
    const index_t* lengths_data,
    const int64_t segment_count,
    bool is_initial_set,
    scalar_t initial) {
  auto& dpcpp_queue = xpu::dpcpp::dpcppGetCurrentQueue();
  const auto dev_id = xpu::dpcpp::dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = xpu::dpcpp::dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (segment_count + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> itemId) {
          int64_t linear_id = itemId.get_global_linear_id();
          if (linear_id < segment_count) {
            SYCL_KERNEL_ASSERT(lengths_data[index] >= 0);
            if (lengths_data[linear_id] == 0) {
              if (is_initial_set) {
                output_data[linear_id] = initial;
              } else {
                output_data[linear_id] = NAN;
              }
            } else if (!Numerics<scalar_t>::isnan(output_data[linear_id])) {
              output_data[linear_id] =
                  output_data[linear_id] / lengths_data[linear_id];
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename index_t>
void segment_reduce_forward_kernel(
    SegmentReductionType reduction,
    scalar_t* output_data,
    scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    bool is_initial_set,
    scalar_t initial_value_raw,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis) {
  const int64_t size = outer_offset * segment_count * inner_offset;
  const int64_t work_group_size = xpu::dpcpp::dpcppMaxWorkGroupSize();
  const int64_t work_group_num = (size + work_group_size - 1) / work_group_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int64_t idx = item.get_global_linear_id();
      auto initial_value = initial_value_raw;
      if (idx >= size) {
        return;
      }
      int64_t row_id = idx / inner_offset;
      int64_t lane_id = idx % inner_offset; // lane_id is the inner_idx
      int64_t outer_idx = row_id / segment_count;
      int64_t dim_idx = row_id % segment_count;

      int64_t offset_idx =
          outer_idx * lengths_cumsum_stride_axis * (segment_count + 1) +
          dim_idx;
      index_t offset_start = lengths_cumsum_data[offset_idx];
      index_t offset_end = lengths_cumsum_data[offset_idx + 1];

      // ===== step2: apply reduction
      for (index_t j = offset_start; j < offset_end; ++j) {
        int64_t data_index = outer_idx * data_stride_axis * data_size_axis +
            j * data_stride_axis + lane_id;
        const auto data = values_data[data_index];
        // TODO: There is no need to branch with every element
        if (reduction == SegmentReductionType::MAX) {
          initial_value = Numerics<scalar_t>::isnan(data)
              ? data
              : std::max<scalar_t>(initial_value, data);
        } else if (
            reduction == SegmentReductionType::MEAN ||
            reduction == SegmentReductionType::SUM) {
          initial_value = initial_value + data;
        } else if (reduction == SegmentReductionType::MIN) {
          initial_value = Numerics<scalar_t>::isnan(data)
              ? data
              : std::min<scalar_t>(initial_value, data);
        } else if (reduction == SegmentReductionType::PROD) {
          initial_value = initial_value * data;
        }
      }

      // ===== step3: finalize reduction
      int64_t lengths_idx =
          outer_idx * lengths_stride_axis * segment_count + dim_idx;
      SYCL_KERNEL_ASSERT(lengths_data[lengths_idx] >= 0);
      if (lengths_data[lengths_idx] == 0 && !is_initial_set &&
          reduction == SegmentReductionType::MEAN) {
        initial_value = static_cast<scalar_t>(NAN);
      } else if (
          reduction == SegmentReductionType::MEAN &&
          lengths_data[lengths_idx] > 0 &&
          !Numerics<scalar_t>::isnan(initial_value)) {
        initial_value = initial_value / lengths_data[lengths_idx];
      }
      int64_t output_index = outer_idx * output_stride_axis * output_size_axis +
          dim_idx * output_stride_axis + lane_id;
      output_data[output_index] = initial_value;
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * work_group_num),
            sycl::range<1>(work_group_size)),
        kfn);
  };

  auto& dpcpp_queue = xpu::dpcpp::dpcppGetCurrentQueue();
  dpcpp_queue.submit(cgf);
}

template <typename scalar_t, typename index_t>
void segment_reduce_backward_kernel(
    SegmentReductionType reduction,
    scalar_t* grad_input_data,
    scalar_t* grad_data,
    scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    scalar_t initial_prod_value,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis) {
  const int64_t size = outer_offset * segment_count * inner_offset;
  const int64_t work_group_size = xpu::dpcpp::dpcppMaxWorkGroupSize();
  const int64_t work_group_num = (size + work_group_size - 1) / work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int64_t idx = item.get_global_linear_id();
      if (idx >= size) {
        return;
      }
      if (idx >= size) {
        return;
      }
      int64_t row_id = idx / inner_offset;
      int64_t lane_id = idx % inner_offset; // lane_id is the inner_idx
      int64_t outer_idx = row_id / segment_count;
      int64_t dim_idx = row_id % segment_count;

      int64_t lengths_idx =
          outer_idx * lengths_stride_axis * segment_count + dim_idx;
      auto segment_length = lengths_data[lengths_idx];
      if (segment_length == 0) {
        return;
      }

      int64_t offset_idx =
          outer_idx * lengths_cumsum_stride_axis * (segment_count + 1) +
          dim_idx;
      index_t offset_start = lengths_cumsum_data[offset_idx];
      index_t offset_end = lengths_cumsum_data[offset_idx + 1];

      int64_t output_index = outer_idx * output_stride_axis * output_size_axis +
          dim_idx * output_stride_axis + lane_id;

      if (reduction == SegmentReductionType::MAX ||
          reduction == SegmentReductionType::MIN) {
        int64_t counter = 0;
        for (int64_t j = offset_start; j < offset_end; ++j) {
          int64_t data_index = outer_idx * data_stride_axis * data_size_axis +
              j * data_stride_axis + lane_id;
          if (Numerics<scalar_t>::isnan(values_data[data_index]) ||
              values_data[data_index] == output_data[output_index]) {
            grad_input_data[data_index] = grad_data[output_index];
            counter++;
          }
        }
        // Average gradient based on number of maximum elements in the
        // segment
        if (counter < 2) {
          return;
        }
        for (int64_t j = offset_start; j < offset_end; ++j) {
          int64_t data_index = outer_idx * data_stride_axis * data_size_axis +
              j * data_stride_axis + lane_id;
          if (grad_input_data[data_index] > 0) {
            grad_input_data[data_index] = grad_input_data[data_index] / counter;
          }
        }
      } else if (reduction == SegmentReductionType::MEAN) {
        auto grad_val = grad_data[output_index] / segment_length;
        for (int64_t j = offset_start; j < offset_end; ++j) {
          int64_t data_index = outer_idx * data_stride_axis * data_size_axis +
              j * data_stride_axis + lane_id;
          grad_input_data[data_index] = grad_val;
        }
      } else if (reduction == SegmentReductionType::SUM) {
        const auto& grad_val = grad_data[output_index];
        for (int64_t j = offset_start; j < offset_end; ++j) {
          int64_t data_index = outer_idx * data_stride_axis * data_size_axis +
              j * data_stride_axis + lane_id;
          grad_input_data[data_index] = grad_val;
        }
      } else if (reduction == SegmentReductionType::PROD) {
        const auto& grad_val =
            grad_data[output_index] * output_data[output_index];
        for (int64_t j = offset_start; j < offset_end; ++j) {
          int64_t data_index = outer_idx * data_stride_axis * data_size_axis +
              j * data_stride_axis + lane_id;
          if (Numerics<scalar_t>::isnan(values_data[data_index]) ||
              values_data[data_index] == 0) {
            // explicitly compute exclusive prod
            scalar_t exclusive_prod = initial_prod_value;
            int64_t prod_idx;
            for (int64_t k = offset_start; k < offset_end; ++k) {
              if (k != j) {
                prod_idx = outer_idx * data_stride_axis * data_size_axis +
                    k * data_stride_axis + lane_id;
                exclusive_prod *= values_data[prod_idx];
              }
            }
            grad_input_data[data_index] =
                grad_data[output_index] * exclusive_prod;
          } else {
            grad_input_data[data_index] = grad_val / values_data[data_index];
          }
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * work_group_num),
            sycl::range<1>(work_group_size)),
        kfn);
  };

  auto& dpcpp_queue = xpu::dpcpp::dpcppGetCurrentQueue();
  dpcpp_queue.submit(cgf);
}

SegmentReductionType get_reduction_enum(const c10::string_view& reduce) {
  if (reduce == "max") {
    return SegmentReductionType::MAX;
  } else if (reduce == "mean") {
    return SegmentReductionType::MEAN;
  } else if (reduce == "min") {
    return SegmentReductionType::MIN;
  } else if (reduce == "sum") {
    return SegmentReductionType::SUM;
  } else if (reduce == "prod") {
    return SegmentReductionType::PROD;
  } else {
    TORCH_CHECK(false, "unsupported reduction given! ", reduce);
  }
}

Tensor _segment_reduce_lengths_offsets_backward_xpu_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const c10::optional<Scalar>& initial,
    bool is_offsets_like) {
  axis = lengths_or_offsets_contig.dim() - 1;
  int64_t segment_count = is_offsets_like
      ? lengths_or_offsets_contig.size(axis) - 1
      : lengths_or_offsets_contig.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets_contig.stride(axis);
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  auto offsets = lengths_or_offsets_contig;
  auto lengths = lengths_or_offsets_contig;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    // _get_complete_sum only supports 1D
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets =
        at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++) {
    outer_offset *= output_contig.size(d);
  }
  for (int64_t d = axis + 1; d < output_contig.dim(); d++) {
    inner_offset *= output_contig.size(d);
  }

  constexpr int threads_per_block = 256;
  int64_t num_blocks =
      (outer_offset * inner_offset * segment_count + threads_per_block - 1) /
      threads_per_block;

  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data_contig.stride(axis);
  auto data_size_axis = data_contig.size(axis);
  auto output_stride_axis = output_contig.stride(axis);
  auto output_size_axis = output_contig.size(axis);
  auto offsets_stride_axis = offsets.stride(axis);

  IPEX_DISPATCH_INDEX_TYPES(
      lengths_or_offsets_contig.scalar_type(),
      "_segment_reduce_xpu_lengths_offsets_backward_kernel1",
      ([&] {
        const auto* lengths_data = lengths.data_ptr<index_t>();
        auto* offsets_data = offsets.data_ptr<index_t>();

        // TODO: Switch to TensorIterator for better maintainablility and
        // readability
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            kBFloat16,
            kHalf,
            data_contig.scalar_type(),
            "_segment_reduce",
            ([&]() {
              auto* output_data = output_contig.data_ptr<scalar_t>();
              auto* grad_data = grad_contig.data_ptr<scalar_t>();
              auto* grad_input_data = grad_input.data_ptr<scalar_t>();
              const auto* values_data = data_contig.data_ptr<scalar_t>();

              scalar_t initial_prod_value;
              if (initial.has_value()) {
                initial_prod_value = initial.value().to<scalar_t>();
              } else {
                initial_prod_value = 1;
              }

              segment_reduce_backward_kernel<scalar_t>(
                  reduction,
                  grad_input_data,
                  grad_data,
                  output_data,
                  values_data,
                  lengths_data,
                  offsets_data,
                  segment_count,
                  lengths_stride_axis,
                  initial_prod_value,
                  outer_offset,
                  inner_offset,
                  data_stride_axis,
                  data_size_axis,
                  output_stride_axis,
                  output_size_axis,
                  offsets_stride_axis);
            }));
      }));
  return grad_input;
}

Tensor _segment_reduce_lengths_backward_xpu_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_xpu_kernel(
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      lengths_contig,
      axis,
      initial,
      /*is_offsets_like=*/false);
}

Tensor _segment_reduce_offsets_backward_xpu_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_xpu_kernel(
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      offsets_contig,
      axis,
      initial,
      /*is_offsets_like=*/true);
}

Tensor _segment_reduce_lengths_offsets_xpu_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& lengths_or_offsets,
    int64_t axis,
    const c10::optional<Scalar>& initial,
    bool is_offsets_like) {
  // data and lengths_or_offsets should be contiguous from the call to
  // .contiguous in segment_reduce_kernel
  TORCH_CHECK(data.is_contiguous());
  TORCH_CHECK(lengths_or_offsets.is_contiguous());
  axis = lengths_or_offsets.dim() - 1;
  int64_t segment_count = is_offsets_like ? lengths_or_offsets.size(axis) - 1
                                          : lengths_or_offsets.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets.stride(axis);
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  auto offsets = lengths_or_offsets;
  auto lengths = lengths_or_offsets;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    // _get_complete_sum only supports 1D
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets =
        at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++) {
    outer_offset *= output.size(d);
  }
  for (int64_t d = axis + 1; d < output.dim(); d++) {
    inner_offset *= output.size(d);
  }

  constexpr int threads_per_block = 256;
  // segment_count * stride_count is just output.numel() ?
  int64_t num_blocks =
      (output.numel() + threads_per_block - 1) / threads_per_block;

  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data.stride(axis);
  auto data_size_axis = data.size(axis);
  auto output_stride_axis = output.stride(axis);
  auto output_size_axis = output.size(axis);
  auto offsets_stride_axis = offsets.stride(axis);

  IPEX_DISPATCH_INDEX_TYPES(
      lengths_or_offsets.scalar_type(), "_segment_reduce_xpu_kernel1", ([&] {
        auto* offsets_data_ptr = offsets.data_ptr<index_t>();
        auto* lengths_data_ptr = lengths.data_ptr<index_t>();
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            data.scalar_type(),
            "segment_reduce_xpu",
            [&]() {
              auto* data_data_ptr = data.data_ptr<scalar_t>();
              auto* output_data_ptr = output.data_ptr<scalar_t>();

              // initialize starting value
              scalar_t initial_value;
              if (initial.has_value()) {
                initial_value = initial.value().to<scalar_t>();
              } else if (reduction == SegmentReductionType::MAX) {
                initial_value = -std::numeric_limits<scalar_t>::infinity();
              } else if (
                  reduction == SegmentReductionType::MEAN ||
                  reduction == SegmentReductionType::SUM) {
                initial_value = 0;
              } else if (reduction == SegmentReductionType::MIN) {
                initial_value = std::numeric_limits<scalar_t>::infinity();
              } else if (reduction == SegmentReductionType::PROD) {
                initial_value = 1;
              }
              segment_reduce_forward_kernel<scalar_t>(
                  reduction,
                  output_data_ptr,
                  data_data_ptr,
                  lengths_data_ptr,
                  offsets_data_ptr,
                  segment_count,
                  lengths_stride_axis,
                  initial.has_value(),
                  initial_value,
                  outer_offset,
                  inner_offset,
                  data_stride_axis,
                  data_size_axis,
                  output_stride_axis,
                  output_size_axis,
                  offsets_stride_axis);
            });
      }));

  return output;
}

Tensor _segment_reduce_lengths_xpu_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_xpu_kernel(
      reduction, data, lengths, axis, initial, /*is_offsets_like=*/false);
}

Tensor _segment_reduce_offsets_xpu_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_xpu_kernel(
      reduction, data, offsets, axis, initial, /*is_offsets_like=*/true);
}

} // namespace impl

Tensor segment_reduce(
    const Tensor& data,
    c10::string_view reduce,
    const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& indices,
    const c10::optional<Tensor>& offsets,
    int64_t axis,
    bool unsafe,
    const c10::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  TORCH_CHECK(data.numel() > 0);

  // check that one of lengths or offsets is defined
  auto lengths_has_value = lengths.has_value();
  auto offsets_has_value = offsets.has_value();
  TORCH_CHECK(
      !indices.has_value(),
      "segment_reduce(): indices based reduction is not supported yet.");
  TORCH_CHECK(
      lengths_has_value || offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.")

  auto reduction = impl::get_reduction_enum(reduce);
  const auto data_contig = data.contiguous();

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();

    // offsets related checks
    TORCH_CHECK(data.get_device() == offsets_value.get_device());
    TORCH_CHECK(data.dim() >= offsets_value.dim());
    TORCH_CHECK(
        axis == offsets_value.dim() - 1,
        "segment_reduce(): Expected axis to be the last dimension of offsets but got ",
        axis,
        ".");

    // TODO: add checks when !unsafe

    const auto offsets_contig = offsets_value.contiguous();

    return _segment_reduce_offsets_xpu_kernel(
        reduction, data_contig, offsets_contig, axis, initial);

  } else {
    const auto& lengths_value = lengths.value();

    // length related checks
    TORCH_CHECK(data.get_device() == lengths_value.get_device());
    TORCH_CHECK(data.dim() >= lengths_value.dim());
    TORCH_CHECK(
        axis == lengths_value.dim() - 1,
        "segment_reduce(): Expected axis to be the last dimension of lengths but got ",
        axis,
        ".");

    if (!unsafe) {
      auto min_length = lengths_value.min().item<int64_t>();
      TORCH_CHECK((min_length >= 0), "lengths contains negative value!");
      TORCH_CHECK(
          all(lengths_value.sum({-1}) == data.size(axis)).item<bool>(),
          "segment_reduce(): Expected all rows of lengths along axis ",
          "to sum to data.size(lengths.dim()-1) when !unsafe.");
    }

    const auto lengths_contig = lengths_value.contiguous();

    return _segment_reduce_lengths_xpu_kernel(
        reduction, data_contig, lengths_contig, axis, initial);
  }
}

// Currently some computation is being duplicated across forward and backward.
// TODO: Cache indices in forward pass to re-use in backward
Tensor _segment_reduce_backward(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    c10::string_view reduce,
    const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& offsets,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  // check that one of lengths or offsets is defined
  // codegen for derivatives.yaml passes an undefined Tensor for None rather
  // than a c10::optional so checking .has_value() doesn't work unlike in the
  // forward pass
  auto lengths_has_value = lengths.has_value() && lengths.value().defined();
  auto offsets_has_value = offsets.has_value() && offsets.value().defined();
  TORCH_CHECK(
      lengths_has_value || offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.");

  const auto grad_contig = grad.contiguous();
  const auto output_contig = output.contiguous();
  const auto data_contig = data.contiguous();
  auto reduction = impl::get_reduction_enum(reduce);

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();
    const auto offsets_contig = offsets_value.contiguous();
    return _segment_reduce_offsets_backward_xpu_kernel(
        grad_contig,
        output_contig,
        data_contig,
        reduction,
        offsets_contig,
        axis,
        initial);
  } else {
    const auto& lengths_value = lengths.value();
    const auto lengths_contig = lengths_value.contiguous();
    return _segment_reduce_lengths_backward_xpu_kernel(
        grad_contig,
        output_contig,
        data_contig,
        reduction,
        lengths_contig,
        axis,
        initial);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
