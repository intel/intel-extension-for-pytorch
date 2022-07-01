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
#include "comm/Atomics.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

constexpr int64_t NROWS_PER_THREAD = 10;
constexpr int64_t WARP_SIZE = 64;

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

template <typename index_t>
void krn_partials_per_segment(
    index_t* ret,
    const index_t* segment_offsets,
    index_t num_of_segments,
    int64_t numel) {
  auto& queue = dpcppGetCurrentQueue();
  auto group_size = 32;
  auto num_groups = CeilDiv(num_of_segments, static_cast<index_t>(group_size));
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto ret_data = ret;
    auto offsets_data = segment_offsets;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto ret_ptr = ret_data;
      auto offsets_ptr = offsets_data;
      auto id = item.get_global_id(0);
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
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
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
  auto group_size = 32;
  auto num_groups = CeilDiv(num_of_segments, static_cast<index_t>(group_size));
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto ret_data = ret;
    auto partials_per_segment_data = partials_per_segment;
    auto partials_per_segment_offset_data = partials_per_segment_offset;
    auto segment_offsets_data = segment_offsets;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto ret_ptr = ret_data;
      auto partials_per_segment_ptr = partials_per_segment_data;
      auto partials_per_segment_offset_ptr = partials_per_segment_offset_data;
      auto segment_offsets_ptr = segment_offsets_data;

      auto id = item.get_global_id(0);
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
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
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
  int64_t stride_warped = CeilDiv(stride, work_group_size) * work_group_size;
  int64_t group_size = std::min(stride_warped, dpcppMaxWorkGroupSize(dev_id));
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
  int64_t stride_warped = CeilDiv(stride, work_group_size) * work_group_size;
  int64_t group_size = std::min(stride_warped, dpcppMaxWorkGroupSize(dev_id));
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
Tensor embedding_bag_backward_dpcpp_kernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    const Tensor& count,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool mode_mean,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& per_sample_weights) {
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
    at::AtenIpexTypeXPU::adjacent_difference<index_t>(
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
    dummy[0] = 1;
    Tensor count_tensor =
        at::empty({numel}, at::TensorOptions().device(kXPU).dtype(kLong));
    auto count_begin = count_tensor.data_ptr<int64_t>();
    at::AtenIpexTypeXPU::iota(count_begin, count_begin + numel, (int64_t)0);
    auto segment_offsets_begin = segment_offsets.data_ptr<index_t>();
    at::AtenIpexTypeXPU::transform<index_t>(
        dummy_begin,
        dummy_begin + numel,
        count_begin,
        idx_begin,
        [](auto d, auto idx) { return d ? idx : -1; });
    auto ends = at::AtenIpexTypeXPU::copy_if<index_t>(
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
  at::AtenIpexTypeXPU::exclusive_scan(
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

template <
    int vec_size,
    typename vec_t,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
void vec_chunk_kernel_embeddingbag(
    const int64_t mode,
    index_t* input,
    index_t* offset,
    scalar_t* weight,
    scalar_t* output,
    index_t* offset2bag,
    index_t* bag_size,
    bool per_sample_weights_defined,
    scalar_t* per_sample_weights,
    int64_t per_sample_weights_stride,
    index_t* max_indices,
    int64_t WGNumber,
    int64_t numBags,
    int64_t weight_total_elem,
    int64_t chunk_size,
    int64_t bag_chunk_num,
    int64_t bag_wi_num,
    int64_t bagsPerLoop,
    int64_t input_length,
    int64_t weight_stride0,
    int64_t weight_stride1,
    const bool include_last_offset,
    const index_t padding_idx,
    const bool if_align_vector,
    DPCPP::nd_item<1> item) {
  auto globalId = item.get_global_linear_id();

  // global chunk id
  auto globalChunkId = globalId / chunk_size;

  // which initial bag this work item is in
  auto bagId = globalChunkId / bag_chunk_num;

  // work item id inside one bag
  auto insideBagId = globalId % bag_wi_num;

  constexpr int align_bytes = alignof(vec_t);

  // outer bag loop
  for (auto bag = bagId; bag < numBags; bag += bagsPerLoop) {
    auto begin = offset[bag];

    // TODO: Here need a check for begin and end that end must >= begin.
    auto end = (bag < (numBags - 1))
        ? (offset[bag + 1])
        : (include_last_offset ? offset[bag + 1] : input_length);

    // for mean mode's backward
    index_t bag_size_ = 0;

    // In single_bag situation, embeddingbag is like embedding, no
    // per_sample_weight, mode is not max and not padding entry and 2D weight,
    // pure vec copy is used to achieve most memory bandwidth.
    auto single_bag = bool(
        (end == (begin + 1)) && (!per_sample_weights_defined) &&
        (mode != MODE_MAX) && (input[begin] != padding_idx));

    if (single_bag) {
      auto input_single_elem = input[begin];

      // for checking alignment with vector
      auto shift = ((uint64_t)(weight + input_single_elem * weight_stride0)) %
          align_bytes / sizeof(scalar_t);

      // here the shift elements need to be individually dealed with
      for (auto mis_idx = 0; mis_idx < shift; ++mis_idx) {
        if (insideBagId == 0) {
          if (mis_idx < weight_stride0) {
            output[bag * weight_stride0 + mis_idx] = weight
                [input_single_elem * weight_stride0 + mis_idx * weight_stride1];
          }
        }
      }

      if (((shift + input_single_elem * weight_stride0) < weight_total_elem) &&
          (shift < weight_stride0)) {
        vec_t* weight_vec = reinterpret_cast<vec_t*>(
            shift + weight + input_single_elem * weight_stride0);
        // vector load
        auto weightSingleValue = weight_vec[insideBagId];
        vec_t* output_vec =
            reinterpret_cast<vec_t*>(shift + output + bag * weight_stride0);
#pragma unroll
        for (auto id = 0; id < vec_size; id++) {
          if ((shift + insideBagId * vec_size + id) < weight_stride0) {
            output_vec[insideBagId][id] =
                weightSingleValue[id * weight_stride1];
          }
        }
      }

      if (insideBagId == 0) {
        offset2bag[begin] = bag;
        bag_size[bag] = static_cast<index_t>(1);
      }
    } else {
      // not single bag mode
      index_t maxWord[vec_size];
      accscalar_t weightFeatSum[vec_size];
      scalar_t weightFeatMax[vec_size];

#pragma unroll
      for (auto id = 0; id < vec_size; id++) {
        maxWord[id] = -1;
        weightFeatSum[id] = static_cast<accscalar_t>(0.0);
        weightFeatMax[id] = static_cast<scalar_t>(0.0);
      }

      // alignment with vector load
      if (if_align_vector) {
        for (auto emb = begin; emb < end; emb++) {
          auto input_elem = input[emb];

          // if this bag copes with multi embeddings and one of these embeddings
          // is padding_idx, this embedding is ignored for reduction because
          // embedding vector at padding_idx is excluded from the reduction
          bool pad = (input_elem == padding_idx);

          // vector process remaining
          vec_t* weight_vec =
              reinterpret_cast<vec_t*>(weight + input_elem * weight_stride0);
          auto weightValue = weight_vec[insideBagId];

#pragma unroll
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_stride0) {
              if (mode == MODE_MAX) {
                // static_cast to scalar_t is used because vec_t contains
                // uint dtype
                auto val = weightValue[id];
                auto max_val = weightFeatMax[id];
                // bag_size_ == 0 means it first come
                if (bag_size_ == 0 || val > max_val) {
                  // padded entry will not be included output
                  weightFeatMax[id] = pad ? weightFeatMax[id] : weightValue[id];
                  maxWord[id] = pad ? maxWord[id] : input_elem;
                }
              } else {
                // 1. for scalar type fma/add, accscalar_t is needed to keep
                // accurate. Vec is stored uint value, whose size is same
                // as sizeof(scalar_t), when computing, uint value should
                // be casted to floating value, after computation,
                // write-back needs casting to uint value.
                // 2. if this entry is padded, 0 value is prepared for
                // reduce(sum/mean)
                auto val = pad ? static_cast<scalar_t>(0.0) : weightValue[id];
                auto acc_val = static_cast<accscalar_t>(val);
                auto acc_sum = weightFeatSum[id];
                if (per_sample_weights_defined) {
                  auto scaleWeightBy = static_cast<accscalar_t>(
                      per_sample_weights[emb * per_sample_weights_stride]);
                  acc_sum += acc_val * scaleWeightBy;
                } else {
                  acc_sum += acc_val;
                }
                weightFeatSum[id] = acc_sum;
              }
            }
          }

          // if this entry is padded, it will not contribute to bag size
          bag_size_ += pad ? 0 : 1;

          // avoid compete write in and padded entry also needs to be recorded
          // to offset2bag
          if (insideBagId == 0) {
            offset2bag[emb] = bag;
          }
        }
      } else {
        // exist misalignment, back to single point processing
        for (auto emb = begin; emb < end; emb++) {
          auto input_elem = input[emb];
          // if this bag copes with multi embeddings and one of these embeddings
          // is padding_idx, this embedding is ignored for reduction because
          // embedding vector at padding_idx is excluded from the reduction
          bool pad = (input_elem == padding_idx);

#pragma unroll
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_stride0) {
              auto weight_idx = input_elem * weight_stride0 +
                  insideBagId * vec_size + id * weight_stride1;
              if (mode == MODE_MAX) {
                // static_cast to scalar_t is used because vec_t contains
                // uint dtype
                auto val = weight[weight_idx];
                auto max_val = weightFeatMax[id];
                // bag_size_ == 0 means it first come
                if (bag_size_ == 0 || val > max_val) {
                  // padded entry will not be included output
                  weightFeatMax[id] = pad ? weightFeatMax[id] : val;
                  maxWord[id] = pad ? maxWord[id] : input_elem;
                }
              } else {
                // 1. for scalar type fma/add, accscalar_t is needed to keep
                // accurate. Vec is stored uint value, whose size is same
                // as sizeof(scalar_t), when computing, uint value should
                // be casted to floating value, after computation,
                // write-back needs casting to uint value.
                // 2. if this entry is padded, 0 value is prepared for
                // reduce(sum/mean)
                auto val =
                    pad ? static_cast<scalar_t>(0.0) : weight[weight_idx];
                auto acc_val = static_cast<accscalar_t>(val);
                if (per_sample_weights_defined) {
                  auto scaleWeightBy = static_cast<accscalar_t>(
                      per_sample_weights[emb * per_sample_weights_stride]);
                  weightFeatSum[id] += acc_val * scaleWeightBy;
                } else {
                  weightFeatSum[id] += acc_val;
                }
              }
            }
          }

          // if this entry is padded, it will not contribute to bag size
          bag_size_ += pad ? 0 : 1;

          // avoid compete write in and padded entry also needs to be recorded
          // to offset2bag
          if (insideBagId == 0) {
            offset2bag[emb] = bag;
          }
        }
      }

      // calculate average for mean mode
      if (mode == MODE_MEAN) {
#pragma unroll
        for (auto id = 0; id < vec_size; id++) {
          if ((insideBagId * vec_size + id) < weight_stride0) {
            auto acc_sum = weightFeatSum[id];
            if (bag_size_ != 0) {
              acc_sum /= static_cast<accscalar_t>(bag_size_);
            }
            weightFeatSum[id] = acc_sum;
          }
        }
      }

      // output
#pragma unroll
      for (auto id = 0; id < vec_size; id++) {
        if ((insideBagId * vec_size + id) < weight_stride0) {
          auto output_idx = bag * weight_stride0 + insideBagId * vec_size +
              id * weight_stride1;
          if (mode == MODE_MEAN || mode == MODE_SUM) {
            output[output_idx] = static_cast<scalar_t>(weightFeatSum[id]);
          } else if (mode == MODE_MAX) {
            output[output_idx] = weightFeatMax[id];
            max_indices[output_idx] = maxWord[id];
          }
        }
      }

      if (insideBagId == 0) {
        bag_size[bag] = static_cast<index_t>(bag_size_);
      }
    }
  }
}

/*
  The kernel EmbeddingBag is optimized for memory coleascing and thread
  efficiency. Vec design and chunk design are deployed for this kernel. In
  additional, single bag is specifically considered.(for example, when
  offset_data=0,1,2,3,4,5,...).
  Thought:
  0. Principle: One or multi chunks work for one Bag. One loop at least solves
  one bag.
  1. Implementation: Use vec<scalar_t, vec_size> to achieve higher bandwidth
  both in ATS and PVC, because it is a memory bound kernel. Use chunk design,
  chunk splitted from different WG to reach high occupancy especially when bag
  dim is much larger. The vec size is determined by device. The chunk size is
  determined by workload amounts and device resource.
  2. If it is single bag specific situation, pure copy is done for kernel.
  Single bag means offset is linear increase by 1.
  3. Passing vec size as template to kernel.

  Shortcoming:
  1. Chunk design may cause some resource waste when work items is handling
  the tail of last bag in one loop.
*/
template <typename scalar_t, typename index_t>
void EmbeddingBag_updateOutputKernel(
    const int64_t mode,
    index_t* input_data,
    index_t* offset_data,
    scalar_t* weight_data,
    scalar_t* output_data,
    index_t* offset2bag_data,
    int64_t weight_total_elem,
    int64_t input_length,
    int64_t numBags,
    int64_t weight_stride0,
    int64_t weight_stride1,
    index_t* bag_size_data,
    index_t* max_indices_data,
    scalar_t* per_sample_weights_data,
    int64_t per_sample_weights_stride,
    const bool include_last_offset,
    const index_t padding_idx) {
  using accscalar_t = acc_type<scalar_t>;

  // vector size, query it according to machine, scalar_t and weight_data
  auto& queue = dpcppGetCurrentQueue();
  auto vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(weight_data));

  // determine per sample weights should be in calculation or not
  bool per_sample_weights_defined = per_sample_weights_data ? true : false;

  auto maxWGSize =
      queue.get_device().template get_info<dpcpp_dev_max_work_group_size>();

  auto maxComputeUnit =
      queue.get_device().template get_info<dpcpp_dev_max_compute_units>();

  // how many work items serve for one bag in vector sight
  auto bag_wi_num = (weight_stride0 % vec_size == 0)
      ? (weight_stride0 / vec_size)
      : (weight_stride0 / vec_size + 1);

  auto chunk_size = 32;

  // how many chunks serve for one bag
  auto bag_chunk_num = (bag_wi_num % chunk_size == 0)
      ? (bag_wi_num / chunk_size)
      : (bag_wi_num / chunk_size + 1);

  // how many work items serve for one bag in chunk sight
  bag_wi_num = bag_chunk_num * chunk_size;

  // how many chunks serve for all bag
  auto all_chunk_num = numBags * bag_chunk_num;

  // how many wi serve for all bag
  auto all_wi_num = all_chunk_num * chunk_size;

  // For huge bags number, limited wg number is set to avoid overhead of
  // groups over scheduling. WGNumber default in single tile in one time =
  // Max compute unit * 8 threads * SIMD32 per thread / max WG size * 512.
  auto WGNumber = maxComputeUnit * 8 * 32 / maxWGSize * 512;

  // one or multi chunks for one bag.
  // all_wi_num <= maxWGSize: one wg is enough to finish all bags
  // bag_wi_num > (maxWGSize * WGNumber): all wg is not enough to finish one
  // bag. To avoid the inner-bag loop, all needed wg are launched
  // else: one wg is not enough to finish all bags, but all wg can finish at
  // least one bag
  auto local_range = maxWGSize;
  if (all_wi_num <= maxWGSize) {
    local_range = all_wi_num;
    WGNumber = 1;
  } else if (bag_wi_num > (maxWGSize * WGNumber)) {
    local_range = maxWGSize;
    // at least, one loop finish one bag
    WGNumber = (bag_wi_num + maxWGSize - 1) / maxWGSize;
  } else {
    for (auto factor = 0; (((maxWGSize - factor * 8) >= 8)); ++factor) {
      auto infactor = maxWGSize - factor * 8;
      if (all_wi_num % infactor == 0) {
        if ((all_wi_num / infactor) > WGNumber) {
          local_range = infactor;
        } else {
          WGNumber = all_wi_num / infactor;
          local_range = infactor;
        }
        break;
      }
    }
  }

  // for outer bag loop, how many bag finish in one loop
  auto bagsPerLoop = WGNumber * local_range / chunk_size / bag_chunk_num;

  // total work item size
  auto global_range = WGNumber * local_range;

  bool if_align_vector = ((weight_stride0 % 2 == 0) || (sizeof(scalar_t) != 2));

// launch vec kernel for embeddingbag, code pass according to vec size
#define VEC_EMBBAG_KERNEL(vec_size)                                           \
  {                                                                           \
    auto cgf = DPCPP_Q_CGF(cgh) {                                             \
      auto input = input_data;                                                \
      auto offset = offset_data;                                              \
      auto weight = weight_data;                                              \
      auto output = output_data;                                              \
      auto offset2bag = offset2bag_data;                                      \
      auto bag_size = bag_size_data;                                          \
      auto per_sample_weights =                                               \
          per_sample_weights_defined ? per_sample_weights_data : weight_data; \
      auto max_indices = mode == MODE_MAX ? max_indices_data : nullptr;       \
      using vec_t =                                                           \
          at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;        \
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {                        \
        vec_chunk_kernel_embeddingbag<                                        \
            vec_size,                                                         \
            vec_t,                                                            \
            scalar_t,                                                         \
            accscalar_t,                                                      \
            index_t>(                                                         \
            mode,                                                             \
            input,                                                            \
            offset,                                                           \
            weight,                                                           \
            output,                                                           \
            offset2bag,                                                       \
            bag_size,                                                         \
            per_sample_weights_defined,                                       \
            per_sample_weights,                                               \
            per_sample_weights_stride,                                        \
            max_indices,                                                      \
            WGNumber,                                                         \
            numBags,                                                          \
            weight_total_elem,                                                \
            chunk_size,                                                       \
            bag_chunk_num,                                                    \
            bag_wi_num,                                                       \
            bagsPerLoop,                                                      \
            input_length,                                                     \
            weight_stride0,                                                   \
            weight_stride1,                                                   \
            include_last_offset,                                              \
            padding_idx,                                                      \
            if_align_vector,                                                  \
            item);                                                            \
      };                                                                      \
      cgh.parallel_for(                                                       \
          DPCPP::nd_range<1>(                                                 \
              DPCPP::range<1>(global_range), DPCPP::range<1>(local_range)),   \
          kfn);                                                               \
    };                                                                        \
    DPCPP_Q_SUBMIT(queue, cgf);                                               \
  }

  switch (vec_size) {
    case 16: {
      VEC_EMBBAG_KERNEL(16);
      break;
    }
    case 8: {
      VEC_EMBBAG_KERNEL(8);
      break;
    }
    case 4: {
      VEC_EMBBAG_KERNEL(4);
      break;
    }
    case 2: {
      VEC_EMBBAG_KERNEL(2);
      break;
    }
    case 1: {
      VEC_EMBBAG_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for EmbeddingBag. vec size ",
          vec_size);
  }
#undef VEC_EMBBAG_KERNEL
} // namespace AtenIpexTypeXPU

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_dpcpp_sum_avg(
    const Tensor& grad,
    const Tensor& indices_t,
    const Tensor& offset2bag_t,
    const Tensor& bag_size_t,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_t,
    int64_t padding_idx) {
  auto indices = indices_t.contiguous();
  auto offset2bag = offset2bag_t.contiguous();
  auto bag_size = bag_size_t.contiguous();
  auto per_sample_weights = per_sample_weights_t.contiguous();

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  ptrdiff_t numel = indices.numel();

  if (numel == 0) {
    // return empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  int64_t stride = grad_weight.stride(0);

  auto sorted_indices = at::empty_like(indices);
  auto sorted_begin = sorted_indices.data_ptr<index_t>();
  auto orig_indices = at::empty_like(indices);
  auto orig_begin = orig_indices.data_ptr<index_t>();

  // directly
  {
    sorted_indices.copy_(indices);
    at::AtenIpexTypeXPU::iota(orig_begin, orig_begin + numel, (index_t)0);
    at::AtenIpexTypeXPU::bitonic_merge_sort_kernel<index_t, index_t>(
        sorted_begin,
        orig_begin,
        sorted_indices.size(0), // prb_size
        1, // batch_size
        sorted_indices.stride(0), // stride
        Numerics<index_t>::upper_bound(), // padding
        [](index_t a, index_t b) { return Numerics<index_t>::lt(a, b); });
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(sorted_indices);
    index_t* count_begin = count.data_ptr<index_t>();
    // Take the maximum of each count per unique key:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    //
    at::AtenIpexTypeXPU::count_by_segment<index_t, index_t, index_t>(
        sorted_begin,
        sorted_begin + numel,
        count_begin,
        [](index_t a, index_t b) { return Numerics<index_t>::eq(a, b); });
  }

  return embedding_bag_backward_dpcpp_kernel<scalar_t, index_t>(
      grad,
      orig_indices,
      sorted_indices,
      count,
      num_weights,
      padding_idx,
      scale_grad_by_freq,
      mode == MODE_MEAN,
      offset2bag,
      bag_size,
      per_sample_weights);
}

template <typename scalar_t, typename index_t>
void EmbeddingBag_accGradParametersKernel_max(
    index_t* max_indices,
    scalar_t* gradOutput,
    scalar_t* gradWeight,
    int64_t stride,
    int64_t numBags) {
  auto& queue = dpcppGetCurrentQueue();
  auto chunksPerBag = CeilDiv(stride, (int64_t)64);
  auto numChunks = numBags * chunksPerBag;
  auto kernel_range = 1024 * 64;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto max_indices_data = max_indices;
    auto gradOutput_data = gradOutput;
    auto gradWeight_data = gradWeight;

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item) {
      auto max_indices_ptr = max_indices_data;
      auto gradOutput_ptr = gradOutput_data;
      auto gradWeight_ptr = gradWeight_data;

      auto chunkOffset = item.get_group()[0] * item.get_local_range()[1] +
          item.get_local_id()[1];

      for (auto chunk = chunkOffset; chunk < numChunks;
           chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
        auto featureDim = (chunk % chunksPerBag) * item.get_local_range(0) +
            item.get_local_id(0);
        if (featureDim < stride) {
          auto bag = chunk / chunksPerBag;

          auto word_idx = max_indices_ptr[bag * stride + featureDim];
          if (word_idx >= 0) {
            // If bag is empty, we have max_indices[idx] set to -1 in forward.
            atomicAdd(
                (dpcpp_global_ptr_pt<scalar_t>)&(
                    gradWeight_ptr[word_idx * stride + featureDim]),
                gradOutput_ptr[bag * stride + featureDim]);
          }
        }
      }
    };

    // kick off kernel
    cgh.parallel_for(
        DPCPP::nd_range<2>(
            DPCPP::range<2>(kernel_range, 4), DPCPP::range<2>(64, 4)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_dpcpp_max(
    const Tensor& grad,
    const Tensor& max_indices_t,
    int64_t num_weights,
    int64_t padding_idx) {
  auto max_indices = max_indices_t.contiguous();
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());
  int64_t stride = grad_weight.stride(0);
  int64_t numBags = grad.size(0);

  EmbeddingBag_accGradParametersKernel_max<scalar_t>(
      max_indices.data_ptr<index_t>(),
      grad.data_ptr<scalar_t>(),
      grad_weight.data_ptr<scalar_t>(),
      stride,
      numBags);

  return grad_weight;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_dpcpp(
    const Tensor& weight_t,
    const Tensor& indices_t,
    const Tensor& offsets_t,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights_t,
    bool include_last_offset,
    int64_t padding_idx) {
  auto weight = weight_t.contiguous();
  auto indices_original = indices_t.contiguous();
  auto offsets_original = offsets_t.contiguous();
  auto per_sample_weights = per_sample_weights_t.contiguous();

  Tensor indices, offsets;
  std::tie(indices, offsets) =
      promoteIndicesAndOffsets(indices_original, offsets_original);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag_dpcpp", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_dpcpp", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_dpcpp", indices_arg, offsets_arg);
  IsOnSameDevice("embedding_bag_dpcpp", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  IsOnSameDevice("embedding_bag_dpcpp", weight_arg, indices_arg);
  IsOnSameDevice("embedding_bag_dpcpp", weight_arg, offsets_arg);

  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);

  // include last offset = True, means the last element of offsets will be set
  // equal to the length of input. Default it is False.
  if (include_last_offset) {
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  int64_t weight_total_elem = weight.numel();

  auto bag_size = at::empty(numBags, indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options());
  auto output = at::empty({numBags, weight.size(1)}, weight.options());

  Tensor max_indices = at::empty({numBags, weight.size(1)}, indices.options());

  if (mode == MODE_MAX) {
    max_indices.zero_();
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_bag_dpcpp",
      [&] {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dpcpp", [&] {
              EmbeddingBag_updateOutputKernel<scalar_t, index_t>(
                  mode,
                  indices.data_ptr<index_t>(),
                  offsets.data_ptr<index_t>(),
                  weight.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>(),
                  offset2bag.data_ptr<index_t>(),
                  weight_total_elem,
                  numIndices,
                  numBags,
                  weight.stride(0),
                  weight.stride(1),
                  bag_size.data_ptr<index_t>(),
                  mode == MODE_MAX ? max_indices.data_ptr<index_t>() : NULL,
                  per_sample_weights.defined()
                      ? per_sample_weights.data_ptr<scalar_t>()
                      : NULL,
                  per_sample_weights.defined() ? per_sample_weights.stride(0)
                                               : 0,
                  include_last_offset,
                  padding_idx);
            });
      });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_dpcpp(
    const Tensor& grad_t,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights,
    int64_t padding_idx) {
  Tensor grad = grad_t.contiguous();
  Tensor result;

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_dense_backward_dpcpp",
      [&] {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dense_backward_dpcpp", [&] {
              switch (mode) {
                case MODE_SUM:
                case MODE_MEAN:
                  if (mode == MODE_MEAN) {
                    TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  }
                  result =
                      embedding_bag_backward_dpcpp_sum_avg<scalar_t, index_t>(
                          grad,
                          indices,
                          offset2bag,
                          bag_size,
                          num_weights,
                          scale_grad_by_freq,
                          mode,
                          per_sample_weights,
                          padding_idx);
                  return result;
                case MODE_MAX:
                  TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  result = embedding_bag_backward_dpcpp_max<scalar_t, index_t>(
                      grad, max_indices, num_weights, padding_idx);
                  return result;
                default:
                  TORCH_CHECK(
                      0,
                      "Unknown mode for embedding_bag_backward_dpcpp ",
                      mode);
              }
            });
      });
  return result;
}

} // namespace impl

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return impl::_embedding_bag_dpcpp(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

Tensor _embedding_bag_dense_backward(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return impl::_embedding_bag_dense_backward_dpcpp(
      grad,
      indices,
      offset2bag,
      bag_size,
      maximum_indices,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights,
      padding_idx);
}

} // namespace AtenIpexTypeXPU
} // namespace at
