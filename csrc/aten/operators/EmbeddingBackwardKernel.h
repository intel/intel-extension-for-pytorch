#pragma once
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "BitonicMergeSort.h"
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

constexpr int64_t NROWS_PER_THREAD = 64;

template <typename index_t>
void krn_partials_per_segment(
    index_t* ret,
    const index_t* segment_offsets,
    index_t num_of_segments,
    int64_t numel) {
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto ret_data = ret;
    auto offsets_data = segment_offsets;
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      auto ret_ptr = ret_data;
      auto offsets_ptr = offsets_data;
      auto id = item.get_linear_id();
      if (id < num_of_segments) {
        const index_t idx_start = offsets_ptr[id];
        const index_t idx_end = (id == num_of_segments - 1)
            ? static_cast<index_t>(numel)
            : offsets_ptr[id + 1];
        const index_t size = idx_end - idx_start;
        ret_ptr[id] = CeilDiv(size, static_cast<index_t>(NROWS_PER_THREAD));
      }
    };

    // kick off kernel
    cgh.parallel_for(DPCPP::range<1>(num_of_segments), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename index_t>
void krn_partial_segment_offset(
    index_t* ret,
    const index_t* partials_per_segment,
    const index_t* partials_per_segment_offset,
    const index_t* segment_offsets,
    index_t num_of_segments) {
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto ret_data = ret;
    auto partials_per_segment_data = partials_per_segment;
    auto partials_per_segment_offset_data = partials_per_segment_offset;
    auto segment_offsets_data = segment_offsets;
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      auto ret_ptr = ret_data;
      auto partials_per_segment_ptr = partials_per_segment_data;
      auto partials_per_segment_offset_ptr = partials_per_segment_offset_data;
      auto segment_offsets_ptr = segment_offsets_data;

      auto id = item.get_linear_id();
      if (id < num_of_segments) {
        index_t idx = partials_per_segment_offset_ptr[id];
        const index_t num_partials = partials_per_segment_ptr[id];
        const index_t segment_offset = segment_offsets_ptr[id];
        for (index_t i = 0; i < num_partials; ++i) {
          ret_ptr[idx++] = segment_offset + i * NROWS_PER_THREAD;
        }
      }
    };

    // kick off kernel
    cgh.parallel_for(DPCPP::range<1>(num_of_segments), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_t>
void compute_grad_weight_bags(
    const Tensor& indices,
    const Tensor& gradOutput,
    const Tensor& offset2bag,
    const Tensor& count,
    int64_t numel,
    int64_t stride,
    int mode_mean,
    const Tensor& bag_size,
    const Tensor& per_sample_weights,
    const Tensor& segment_offsets,
    int64_t num_of_segments,
    const Tensor& grad_weight_per_segment) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  int64_t work_group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t stride_warped =
      CeilDiv(stride, SYCL_MAX_SUB_GROUP_SIZE) * SYCL_MAX_SUB_GROUP_SIZE;
  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = CeilDiv(num_of_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;

  bool per_sample_weight_defined = per_sample_weights.defined();
  bool count_defined = count.defined();
  int64_t per_sample_weights_stride =
      per_sample_weights.defined() ? per_sample_weights.stride(0) : 0;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_weight_per_segment_data =
        grad_weight_per_segment.template data_ptr<acc_type<scalar_t>>();
    auto indices_data = indices.template data_ptr<index_t>();
    auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
    auto offset2bag_data = offset2bag.data_ptr<index_t>();
    auto count_data = count_defined
        ? count.data_ptr<index_t>()
        : offset2bag_data; // use the offset2bag_data handler as the dummy
                           // buffer.
    auto bag_size_data = bag_size.data_ptr<index_t>();
    auto per_sample_weights_data = per_sample_weight_defined
        ? per_sample_weights.data_ptr<scalar_t>()
        : gradOutput_data; // ise the gradOutput_data handler as the dummy
                           // buffer.
    auto segment_offsets_data = segment_offsets.data_ptr<index_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto grad_weight_per_segment_ptr = grad_weight_per_segment_data;
      auto indices_ptr = indices_data;
      auto gradOutput_ptr = gradOutput_data;
      auto offset2bag_ptr = offset2bag_data;
      auto count_ptr = count_defined ? count_data : NULL;
      auto bag_size_ptr = bag_size_data;
      auto per_sample_weights_ptr =
          per_sample_weight_defined ? per_sample_weights_data : NULL;
      auto segment_offsets_ptr = segment_offsets_data;

      const int gid = item.get_global_linear_id();
      const int id = gid / stride_warped;
      const int startFeature = gid % stride_warped;
      if (startFeature >= stride) {
        return;
      }
      if (id >= num_of_segments) {
        return;
      }

      const int idx_begin = segment_offsets_ptr[id];
      const int idx_end =
          (id == num_of_segments - 1) ? numel : segment_offsets_ptr[id + 1];

      acc_type<scalar_t> weight = 0;
      for (int idx = idx_begin; idx < idx_end; ++idx) {
        const int orig_row = indices_ptr[idx];
        const int seq_number = offset2bag_ptr[orig_row];
        const int grad_output_row = seq_number * stride;

        acc_type<scalar_t> scale = count_ptr ? 1.0 / count_ptr[idx] : 1.0;
        if (per_sample_weight_defined) {
          scale *= per_sample_weights_ptr[idx * per_sample_weights_stride];
        }

        acc_type<scalar_t> gradient =
            gradOutput_ptr[grad_output_row + startFeature];
        if (mode_mean) {
          gradient /= bag_size_ptr[seq_number];
        }
        weight += gradient * scale;
      }
      grad_weight_per_segment_ptr[id * stride + startFeature] = weight;
    };

    // kick off kernel
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_t>
void compute_grad_weight(
    const Tensor& indices,
    const Tensor& grad_output,
    const Tensor& count,
    ptrdiff_t numel,
    int64_t stride,
    const Tensor& segment_offsets,
    int64_t num_of_segments,
    const Tensor& grad_weight_per_segment) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  int64_t work_group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t stride_warped =
      CeilDiv(stride, SYCL_MAX_SUB_GROUP_SIZE) * SYCL_MAX_SUB_GROUP_SIZE;
  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = CeilDiv(num_of_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;

  bool count_defined = count.defined();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_weight_per_segment_data =
        grad_weight_per_segment.data_ptr<acc_type<scalar_t>>();
    auto indices_data = indices.data_ptr<index_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto count_data = count_defined
        ? count.data_ptr<index_t>()
        : indices_data; // use the indices_data handler as the dummy buffer.
    auto segment_offsets_data = segment_offsets.data_ptr<index_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto grad_weight_per_segment_ptr = grad_weight_per_segment_data;
      auto indices_ptr = indices_data;
      auto grad_output_ptr = grad_output_data;
      auto count_ptr = count_defined ? count_data : NULL;
      auto segment_offsets_ptr = segment_offsets_data;

      const int gid = item.get_global_linear_id();
      const int id = gid / stride_warped;
      const int startFeature = gid % stride_warped;
      if (startFeature >= stride) {
        return;
      }
      if (id >= num_of_segments) {
        return;
      }
      const int idx_begin = segment_offsets_ptr[id];
      const int idx_end =
          (id == num_of_segments - 1) ? numel : segment_offsets_ptr[id + 1];

      acc_type<scalar_t> weight = 0;
      for (int idx = idx_begin; idx < idx_end; idx++) {
        const index_t target_row = indices_ptr[idx];

        const acc_type<scalar_t> scale = count_ptr ? 1.0 / count_ptr[idx] : 1.0;
        weight += grad_output_ptr[target_row * stride + startFeature] * scale;
      }
      grad_weight_per_segment_ptr[id * stride + startFeature] = weight;
    };

    // kick off kernel
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_t>
void sum_and_scatter(
    const Tensor& input,
    const Tensor& grad_weight,
    int64_t stride,
    const Tensor& segment_offsets,
    int64_t num_of_segments,
    const Tensor& grad_weight_per_segment,
    const Tensor& segment_sizes_offsets,
    int64_t num_of_partial_segments,
    const int64_t padding_idx) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  int64_t work_group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t stride_warped = CeilDiv(stride, work_group_size) * work_group_size;
  int64_t group_size = std::min(stride_warped, dpcppMaxWorkGroupSize(dev_id));
  auto num_groups = CeilDiv(num_of_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_weight_data = grad_weight.data_ptr<scalar_t>();
    auto input_data = input.data_ptr<index_t>();
    auto segment_offsets_data = segment_offsets.data_ptr<index_t>();
    auto grad_weight_per_segment_data =
        grad_weight_per_segment.data_ptr<acc_type<scalar_t>>();
    auto segment_sizes_offsets_data = segment_sizes_offsets.data_ptr<index_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto grad_weight_ptr = grad_weight_data;
      auto input_ptr = input_data;
      auto segment_offsets_ptr = segment_offsets_data;
      auto grad_weight_per_segment_ptr = grad_weight_per_segment_data;
      auto segment_sizes_offsets_ptr = segment_sizes_offsets_data;

      const int gid = item.get_global_linear_id();
      const int id = gid / stride_warped;
      const int startFeature = gid % stride_warped;
      if (startFeature >= stride) {
        return;
      }
      if (id >= num_of_segments) {
        return;
      }

      const int idx_begin = segment_sizes_offsets_ptr[id];
      const int idx_end = (id == num_of_segments - 1)
          ? num_of_partial_segments
          : segment_sizes_offsets_ptr[id + 1];
      acc_type<scalar_t> weight = 0;
      for (int idx = idx_begin; idx < idx_end; idx++) {
        weight += grad_weight_per_segment_ptr[idx * stride + startFeature];
      }

      int64_t target_row = input_ptr[segment_offsets_ptr[id]];
      if (target_row != padding_idx) {
        grad_weight_ptr[target_row * stride + startFeature] = weight;
      }
    };

    // kick off kernel
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_t>
Tensor embedding_backward_deterministic_kernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    const Tensor& count,
    int64_t num_weights,
    int64_t padding_idx = -1,
    bool mode_mean = false,
    const Tensor& offset2bag = Tensor(),
    const Tensor& bag_size = Tensor(),
    const Tensor& per_sample_weights = Tensor()) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const int64_t numel = sorted_indices.numel();
  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);

  auto segment_offsets = at::empty({numel}, orig_indices.options());
  index_t num_of_segments;
  {
    // sorted:          2 5 5 5 7 7 8 9 9
    // dummy:           1 1 0 0 1 0 1 1 0
    // segment_offsets: 0 1 - - 4 - 6 7 -
    auto sorted_indices_begin = sorted_indices.data_ptr<index_t>();
    auto dummy = at::empty_like(sorted_indices);
    auto dummy_begin = dummy.data_ptr<index_t>();
    auto idx_tensor = at::empty_like(sorted_indices);
    auto idx_begin = idx_tensor.data_ptr<index_t>();
    xpu::pstl::adjacent_difference<index_t>(
        sorted_indices_begin,
        sorted_indices_begin + numel,
        dummy_begin,
        [](auto lhs, auto rhs) -> bool {
          if (lhs != rhs) {
            return true;
          }
          return false;
        });

    // For algorithm adjacent difference, for output, its first element is
    // always equal to source first element. We need to set it as 1 manually.
    Tensor count_tensor =
        at::empty({numel}, at::TensorOptions().device(kXPU).dtype(kLong));
    auto count_begin = count_tensor.data_ptr<int64_t>();
    xpu::pstl::iota(count_begin, count_begin + numel, (int64_t)0);
    auto segment_offsets_begin = segment_offsets.data_ptr<index_t>();
    xpu::pstl::transform_first_true<index_t>(
        dummy_begin,
        dummy_begin + numel,
        count_begin,
        idx_begin,
        [](auto d, auto idx) { return d ? idx : -1; });
    auto ends = xpu::pstl::copy_if<index_t>(
        idx_begin, idx_begin + numel, segment_offsets_begin, [](auto x) {
          return x != -1;
        });
    num_of_segments = std::distance(segment_offsets_begin, ends);
  }

  auto partials_per_segment =
      at::empty({num_of_segments}, orig_indices.options());

  krn_partials_per_segment<index_t>(
      partials_per_segment.template data_ptr<index_t>(),
      segment_offsets.data_ptr<index_t>(),
      num_of_segments,
      numel);

  // In order to compute `partial_segment_offset`, which is the start index
  // of each partial-segment in `sorted_indices`, we need to compute the
  // start position of each _segment_ in `partial_segment_offset`.
  // Unit: index in `partial_segment_offset`
  auto partials_per_segment_offset =
      at::empty({num_of_segments}, orig_indices.options());
  xpu::pstl::exclusive_scan(
      partials_per_segment.template data_ptr<index_t>(),
      partials_per_segment.template data_ptr<index_t>() + num_of_segments,
      partials_per_segment_offset.template data_ptr<index_t>(),
      (index_t)0);

  // The total number of partial-segments is the sum of
  // `partials_per_segment_offset`
  auto num_of_partial_segments =
      partials_per_segment[num_of_segments - 1].template item<index_t>() +
      partials_per_segment_offset[num_of_segments - 1].template item<index_t>();

  auto partial_segment_offset =
      at::empty({num_of_partial_segments}, orig_indices.options());
  krn_partial_segment_offset<index_t>(
      partial_segment_offset.template data_ptr<index_t>(),
      partials_per_segment.template data_ptr<index_t>(),
      partials_per_segment_offset.template data_ptr<index_t>(),
      segment_offsets.data_ptr<index_t>(),
      num_of_segments);

  TensorOptions op;
  if (grad.dtype() == at::kBFloat16 || grad.dtype() == at::kHalf) {
    op = grad.options().dtype(at::kFloat);
  } else {
    op = grad.options();
  }
  auto grad_weight_per_segment =
      at::empty({num_of_partial_segments, stride}, op);
  // Compute the sum of each partial-segment and handle bags
  if (offset2bag.defined()) {
    compute_grad_weight_bags<scalar_t, index_t>(
        orig_indices,
        grad,
        offset2bag,
        count,
        numel,
        stride,
        mode_mean,
        bag_size,
        per_sample_weights,
        partial_segment_offset,
        num_of_partial_segments,
        grad_weight_per_segment);
  } else {
    compute_grad_weight<scalar_t, index_t>(
        orig_indices,
        grad,
        count,
        numel,
        stride,
        partial_segment_offset,
        num_of_partial_segments,
        grad_weight_per_segment);
  }

  sum_and_scatter<scalar_t, index_t>(
      sorted_indices,
      grad_weight,
      stride,
      segment_offsets,
      num_of_segments,
      grad_weight_per_segment,
      partials_per_segment_offset,
      num_of_partial_segments,
      padding_idx);

  return grad_weight;
}

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at