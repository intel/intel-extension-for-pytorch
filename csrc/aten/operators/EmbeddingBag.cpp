#include <ATen/ATen.h>
#include <torch/torch.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "BitonicMergeSort.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/PSTLFunctions.h"

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

void krn_partials_per_segment(
    int64_t* ret,
    const int64_t* segment_offsets,
    int64_t num_of_segments,
    int64_t numel) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t group_size = 32;
  auto num_groups = CeilDiv(num_of_segments, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto ret_data = ret;
    auto offsets_data = segment_offsets;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto ret_ptr = ret_data;
      auto offsets_ptr = offsets_data;
      int64_t id = item.get_global_id(0);
      if (id < num_of_segments) {
        const int64_t idx_start = offsets_ptr[id];
        const int64_t idx_end =
            (id == num_of_segments - 1) ? numel : offsets_ptr[id + 1];
        const int64_t size = idx_end - idx_start;
        ret_ptr[id] = CeilDiv(size, NROWS_PER_THREAD);
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

void krn_partial_segment_offset(
    int64_t* ret,
    const int64_t* partials_per_segment,
    const int64_t* partials_per_segment_offset,
    const int64_t* segment_offsets,
    int64_t num_of_segments) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t group_size = 32;
  auto num_groups = CeilDiv(num_of_segments, group_size);
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

      int64_t id = item.get_global_id(0);
      if (id < num_of_segments) {
        int64_t idx = partials_per_segment_offset_ptr[id];
        const int64_t num_partials = partials_per_segment_ptr[id];
        const int64_t segment_offset = segment_offsets_ptr[id];
        for (int64_t i = 0; i < num_partials; ++i) {
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

template <typename scalar_t>
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
        grad_weight_per_segment.data_ptr<acc_type<scalar_t>>();
    auto indices_data = indices.data_ptr<int64_t>();
    auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
    auto offset2bag_data = offset2bag.data_ptr<int64_t>();
    auto count_data = count_defined
        ? count.data_ptr<int64_t>()
        : offset2bag_data; // use the offset2bag_data handler as the dummy
                           // buffer.
    auto bag_size_data = bag_size.data_ptr<int64_t>();
    auto per_sample_weights_data = per_sample_weight_defined
        ? per_sample_weights.data_ptr<scalar_t>()
        : gradOutput_data; // ise the gradOutput_data handler as the dummy
                           // buffer.
    auto segment_offsets_data = segment_offsets.data_ptr<int64_t>();

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

template <typename scalar_t>
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
    auto indices_data = indices.data_ptr<int64_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto count_data = count_defined
        ? count.data_ptr<int64_t>()
        : indices_data; // use the indices_data handler as the dummy buffer.
    auto segment_offsets_data = segment_offsets.data_ptr<int64_t>();

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
        const int64_t target_row = indices_ptr[idx];
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

template <typename scalar_t>
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
    auto input_data = input.data_ptr<int64_t>();
    auto segment_offsets_data = segment_offsets.data_ptr<int64_t>();
    auto grad_weight_per_segment_data =
        grad_weight_per_segment.data_ptr<acc_type<scalar_t>>();
    auto segment_sizes_offsets_data = segment_sizes_offsets.data_ptr<int64_t>();

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

Tensor embedding_bag_backward_dpcpp_kernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    const Tensor& count,
    int64_t num_weights,
    int padding_idx,
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
  int64_t num_of_segments;
  {
    // sorted:          2 5 5 5 7 7 8 9 9
    // dummy:           1 1 0 0 1 0 1 1 0
    // segment_offsets: 0 1 - - 4 - 6 7 -
    auto sorted_indices_begin = sorted_indices.data_ptr<int64_t>();
    auto dummy = at::empty_like(sorted_indices);
    auto dummy_begin = dummy.data_ptr<int64_t>();
    auto idx_tensor = at::empty_like(sorted_indices);
    auto idx_begin = idx_tensor.data_ptr<int64_t>();
    at::AtenIpexTypeXPU::adjacent_difference<int64_t>(
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
    auto segment_offsets_begin = segment_offsets.data_ptr<int64_t>();
    at::AtenIpexTypeXPU::transform<int64_t>(
        dummy_begin,
        dummy_begin + numel,
        count_begin,
        idx_begin,
        [](auto d, auto idx) { return d ? idx : -1; });
    auto ends = at::AtenIpexTypeXPU::copy_if<int64_t>(
        idx_begin, idx_begin + numel, segment_offsets_begin, [](auto x) {
          return x != -1;
        });
    num_of_segments = std::distance(segment_offsets_begin, ends);
  }

  auto partials_per_segment =
      at::empty({num_of_segments}, orig_indices.options());

  krn_partials_per_segment(
      partials_per_segment.data_ptr<int64_t>(),
      segment_offsets.data_ptr<int64_t>(),
      num_of_segments,
      numel);

  // In order to compute `partial_segment_offset`, which is the start index
  // of each partial-segment in `sorted_indices`, we need to compute the
  // start position of each _segment_ in `partial_segment_offset`.
  // Unit: index in `partial_segment_offset`
  auto partials_per_segment_offset =
      at::empty({num_of_segments}, orig_indices.options());
  at::AtenIpexTypeXPU::exclusive_scan(
      partials_per_segment.data_ptr<int64_t>(),
      partials_per_segment.data_ptr<int64_t>() + num_of_segments,
      partials_per_segment_offset.data_ptr<int64_t>(),
      (int64_t)0);

  // The total number of partial-segments is the sum of
  // `partials_per_segment_offset`
  const int num_of_partial_segments =
      partials_per_segment[num_of_segments - 1].item<int64_t>() +
      partials_per_segment_offset[num_of_segments - 1].item<int64_t>();

  auto partial_segment_offset =
      at::empty({num_of_partial_segments}, orig_indices.options());
  krn_partial_segment_offset(
      partial_segment_offset.data_ptr<int64_t>(),
      partials_per_segment.data_ptr<int64_t>(),
      partials_per_segment_offset.data_ptr<int64_t>(),
      segment_offsets.data_ptr<int64_t>(),
      num_of_segments);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_backward_dpcpp_compute_grad_weight",
      [&] {
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
          compute_grad_weight_bags<scalar_t>(
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
          compute_grad_weight<scalar_t>(
              orig_indices,
              grad,
              count,
              numel,
              stride,
              partial_segment_offset,
              num_of_partial_segments,
              grad_weight_per_segment);
        }

        sum_and_scatter<scalar_t>(
            sorted_indices,
            grad_weight,
            stride,
            segment_offsets,
            num_of_segments,
            grad_weight_per_segment,
            partials_per_segment_offset,
            num_of_partial_segments,
            padding_idx);
      });

  return grad_weight;
}

template <
    int vec_size,
    typename vec_t,
    typename elem_t,
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
    bool feature_full_divide,
    int64_t chunk_size,
    int64_t bag_chunk_num,
    int64_t bag_wi_num,
    int64_t bagsPerLoop,
    int64_t input_length,
    int64_t weight_feature_size,
    int64_t weight_stride0,
    int64_t weight_stride1,
    DPCPP::nd_item<1> item) {
  auto globalId = item.get_global_linear_id();

  // global chunk id
  auto globalChunkId = globalId / chunk_size;

  // which initial bag this work item is in
  auto bagId = globalChunkId / bag_chunk_num;

  // work item id inside one bag
  auto insideBagId = globalId % bag_wi_num;

  // outer bag loop
  for (auto bag = bagId; bag < numBags; bag += bagsPerLoop) {
    auto begin = offset[bag];
    auto end = (bag < numBags - 1) ? (offset[bag + 1]) : input_length;
    // In single_bag situation, embeddingbag is like embedding, no
    // per_sample_weight, mode is sum, vec copy is used to achieve
    // higher bandwidth.
    auto single_bag = bool(
        (end == (begin + 1)) && (!per_sample_weights_defined) &&
        (mode == MODE_SUM));

    if (single_bag) {
      if (feature_full_divide) {
        vec_t* weight_vec = reinterpret_cast<vec_t*>(weight);
        // for single bag and feature full divide, fully copy is done with no
        // tail situation
        auto weightOffset = input[begin] * bag_wi_num;
        auto weightOffset_inbag = weightOffset + insideBagId;
        auto weightValue = weight_vec[weightOffset_inbag];
        vec_t* output_vec = reinterpret_cast<vec_t*>(output);
        auto output_offset = bag * bag_wi_num + insideBagId;
        output_vec[output_offset] = weightValue;
      } else {
        // for tail, element-wise addressing is needed
        vec_t* weight_vec = reinterpret_cast<vec_t*>(
            weight + input[begin] * weight_feature_size);
        vec_t* output_vec =
            reinterpret_cast<vec_t*>(output + bag * weight_feature_size);
        for (auto id = 0; id < vec_size; id++) {
          // kick off tail worker id
          if ((insideBagId * vec_size + id) < weight_feature_size) {
            output_vec[insideBagId][id] = weight_vec[insideBagId][id];
          }
        }
      }
      // avoid compete write in
      if (insideBagId == 0) {
        offset2bag[begin] = bag;
      }
    } else {
      vec_t weightFeatSum;
      // initial to 0 for accumulating
      for (auto id = 0; id < vec_size; id++) {
        weightFeatSum[id] =
            at::native::Memory::detail::bitwise_cast<elem_t>(scalar_t{0});
      }

      vec_t weightFeatMax;
      index_t bag_size_ = 0;
      // watch out register spill out when vec_size is large
      index_t maxWord[vec_size] = {0};

      for (int64_t emb = begin; emb < end; emb++) {
        vec_t* weight_vec =
            reinterpret_cast<vec_t*>(weight + input[emb] * weight_feature_size);
        auto weightValue = weight_vec[insideBagId];

        for (auto id = 0; id < vec_size; id++) {
          if ((insideBagId * vec_size + id) < weight_feature_size) {
            if (mode == MODE_MAX) {
              // static_cast to scalar_t is used because vec_t contains
              // uint dtype
              auto val = at::native::Memory::detail::bitwise_cast<scalar_t>(
                  weightValue[id]);
              auto max_val = at::native::Memory::detail::bitwise_cast<scalar_t>(
                  weightFeatMax[id]);
              if (emb == begin || val > max_val) {
                weightFeatMax[id] = weightValue[id];
                maxWord[id] = input[emb];
              }
            } else {
              // for scalar type fma/add, accscalar_t is needed to keep
              // accurate. Vec is stored uint value, whose size is same
              // as sizeof(scalar_t), when computing, uint value should
              // be casted to floating value, after computation,
              // write-back needs casting to uint value.
              auto val = at::native::Memory::detail::bitwise_cast<scalar_t>(
                  weightValue[id]);
              auto acc_val = static_cast<accscalar_t>(val);
              auto sum = at::native::Memory::detail::bitwise_cast<scalar_t>(
                  weightFeatSum[id]);
              auto acc_sum = static_cast<accscalar_t>(sum);
              if (per_sample_weights_defined) {
                auto scaleWeightBy = static_cast<accscalar_t>(
                    per_sample_weights[emb * per_sample_weights_stride]);
                acc_sum += acc_val * scaleWeightBy;
              } else {
                acc_sum += acc_val;
              }
              auto _res = static_cast<scalar_t>(acc_sum);
              weightFeatSum[id] =
                  at::native::Memory::detail::bitwise_cast<elem_t>(_res);
            }
          }
        }
        bag_size_++;
        // avoid compete write in
        if (insideBagId == 0) {
          offset2bag[emb] = bag;
        }
      }

      if (mode == MODE_MEAN) {
        if (end == begin) {
          bag_size[bag] = static_cast<index_t>(0);
        } else {
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_feature_size) {
              auto sum = at::native::Memory::detail::bitwise_cast<scalar_t>(
                  weightFeatSum[id]);
              auto acc_sum = static_cast<accscalar_t>(sum);
              acc_sum /= static_cast<accscalar_t>(bag_size_);
              auto _res = static_cast<scalar_t>(acc_sum);
              weightFeatSum[id] =
                  at::native::Memory::detail::bitwise_cast<elem_t>(_res);
              bag_size[bag] = static_cast<index_t>(bag_size_);
            }
          }
        }
      }

      vec_t* output_vec =
          reinterpret_cast<vec_t*>(output + bag * weight_feature_size);
      for (auto id = 0; id < vec_size; id++) {
        if ((insideBagId * vec_size + id) < weight_feature_size) {
          if (mode == MODE_MEAN || mode == MODE_SUM) {
            output_vec[insideBagId][id] = weightFeatSum[id];
          } else if (mode == MODE_MAX) {
            if (end == begin) {
              weightFeatMax[id] =
                  at::native::Memory::detail::bitwise_cast<elem_t>(scalar_t{0});
            }
            output_vec[insideBagId][id] = weightFeatMax[id];
            max_indices
                [bag * weight_feature_size + insideBagId * vec_size + id] =
                    maxWord[id];
          }
        }
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
  1. Chunk design may cause some resource waste when work items is handling the
  tail of last bag in one loop.
*/
template <typename scalar_t, typename index_t>
void EmbeddingBag_updateOutputKernel(
    const int64_t mode,
    index_t* input_data,
    index_t* offset_data,
    scalar_t* weight_data,
    scalar_t* output_data,
    index_t* offset2bag_data,
    int64_t input_length,
    int64_t numBags,
    int64_t weight_feature_size,
    int64_t weight_stride0,
    int64_t weight_stride1,
    index_t* bag_size_data,
    index_t* max_indices_data,
    scalar_t* per_sample_weights_data,
    int64_t per_sample_weights_stride) {
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
  auto bag_wi_num = (weight_feature_size % vec_size == 0)
      ? (weight_feature_size / vec_size)
      : (weight_feature_size / vec_size + 1);

  // chunk size, up to the bag_wi_num
  // candidate size are 8, 16 and 32.
  // Thought: avoid most waste work items
  // TODO: candidate size may need to be considered, however chunks size had
  // TODO: better to be the factor of the wg size to avoid the divergence cross
  // TODO: work group
  auto chunk_size = 8;
  if (bag_wi_num % 32 == 0) {
    chunk_size = 32;
  } else if (bag_wi_num % 16 == 0) {
    chunk_size = 16;
  } else if (bag_wi_num % 8 == 0) {
    chunk_size = 8;
  } else if (
      ((bag_wi_num % 8) == (bag_wi_num % 16)) &&
      ((bag_wi_num % 16) == (bag_wi_num % 32))) {
    chunk_size = 8;
  } else if ((bag_wi_num % 16) == (bag_wi_num % 32)) {
    chunk_size = 16;
  } else if ((bag_wi_num % 8) == (bag_wi_num % 16)) {
    chunk_size = 8;
  } else {
    chunk_size = 32;
  }

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

  // on host-side, the weight feature size can be fully divided by chunk size
  // and vec size or not will affect the performance. Under full divided
  // circumstance, kernel has highest efficiency and bandwidth
  auto feature_full_divide = bool(
      ((weight_feature_size % vec_size) == 0) &&
      (((weight_feature_size / vec_size) % chunk_size) == 0));

  // For huge bags number, limited wg number is set to avoid overhead of
  // group scheduling. WGNumber default in single tile in one time =
  // Max compute unit * 8 threads * SIMD32 per thread / max WG size * 512.
  // TODO: 512 is an empirical value, may need tune
  // FIXME: maxComputeUnit may have issue now.
  // FIXME: Jira link: https://jira.devtools.intel.com/browse/XDEPS-3272
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
      auto max_indices =                                                      \
          mode == MODE_MAX ? max_indices_data : offset2bag_data;              \
      using vec_t = typename at::native::Memory::                             \
          aligned_vector<scalar_t, vec_size>::type;                           \
      using elem_t = typename at::native::Memory::                            \
          aligned_vector<scalar_t, vec_size>::element_type;                   \
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {                        \
        vec_chunk_kernel_embeddingbag<                                        \
            vec_size,                                                         \
            vec_t,                                                            \
            elem_t,                                                           \
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
            feature_full_divide,                                              \
            chunk_size,                                                       \
            bag_chunk_num,                                                    \
            bag_wi_num,                                                       \
            bagsPerLoop,                                                      \
            input_length,                                                     \
            weight_feature_size,                                              \
            weight_stride0,                                                   \
            weight_stride1,                                                   \
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

Tensor embedding_bag_backward_dpcpp_sum_avg(
    const Tensor& grad,
    const Tensor& indices,
    // const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights) {
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  ptrdiff_t numel = indices.numel();

  if (numel == 0) {
    // all empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  int64_t stride = grad_weight.stride(0);

  auto sorted_indices = at::empty_like(indices);
  auto sorted_begin = sorted_indices.data_ptr<int64_t>();
  auto orig_indices = at::empty_like(indices);
  auto orig_begin = orig_indices.data_ptr<int64_t>();

  // directly
  {
    sorted_indices.copy_(indices);
    at::AtenIpexTypeXPU::iota(orig_begin, orig_begin + numel, (int64_t)0);
    at::AtenIpexTypeXPU::bitonic_merge_sort_kernel<int64_t, int64_t>(
        sorted_begin,
        orig_begin,
        sorted_indices.size(0), // prb_size
        1, // batch_size
        sorted_indices.stride(0), // stride
        Numerics<int64_t>::upper_bound(), // padding
        [](int64_t a, int64_t b) { return Numerics<int64_t>::lt(a, b); });
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(sorted_indices);
    int64_t* count_begin = count.data_ptr<int64_t>();
    // Take the maximum of each count per unique key:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    //
    at::AtenIpexTypeXPU::count_by_segment<int64_t, int64_t, int64_t>(
        sorted_begin,
        sorted_begin + numel,
        count_begin,
        [](int64_t a, int64_t b) { return Numerics<int64_t>::eq(a, b); });
  }

  return embedding_bag_backward_dpcpp_kernel(
      grad,
      orig_indices,
      sorted_indices,
      count,
      num_weights,
      /* padding_idx= */ -1,
      scale_grad_by_freq,
      mode == MODE_MEAN,
      offset2bag,
      bag_size,
      per_sample_weights);
}

template <typename scalar_t>
void EmbeddingBag_accGradParametersKernel_max(
    int64_t* max_indices,
    scalar_t* gradOutput,
    scalar_t* gradWeight,
    int64_t stride,
    int64_t numBags) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t chunksPerBag = CeilDiv(stride, (int64_t)64);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t kernel_range = 1024 * 64;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto max_indices_data = max_indices;
    auto gradOutput_data = gradOutput;
    auto gradWeight_data = gradWeight;

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item) {
      auto max_indices_ptr = max_indices_data;
      auto gradOutput_ptr = gradOutput_data;
      auto gradWeight_ptr = gradWeight_data;

      int64_t chunkOffset = item.get_group()[0] * item.get_local_range()[1] +
          item.get_local_id()[1];

      for (int64_t chunk = chunkOffset; chunk < numChunks;
           chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
        int64_t featureDim = (chunk % chunksPerBag) * item.get_local_range(0) +
            item.get_local_id(0);
        if (featureDim < stride) {
          int64_t bag = chunk / chunksPerBag;

          int64_t word_idx = max_indices_ptr[bag * stride + featureDim];
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

Tensor embedding_bag_backward_dpcpp_max(
    const Tensor& grad,
    const Tensor& max_indices,
    int64_t num_weights) {
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  int64_t stride = grad_weight.stride(0);

  int64_t numBags = grad.size(0);

  // for atomicAdd, only support float datatype.
  EmbeddingBag_accGradParametersKernel_max<float>(
      max_indices.data_ptr<int64_t>(),
      grad.data_ptr<float>(),
      grad_weight.data_ptr<float>(),
      stride,
      numBags);

  return grad_weight;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_dpcpp(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights) {
  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
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
  int64_t featureSize = weight.size(1);

  auto bag_size = at::empty(offsets.sizes(), indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options());
  auto output = at::empty({offsets.size(0), weight.size(1)}, weight.options());

  Tensor max_indices =
      at::empty({offsets.size(0), weight.size(1)}, indices.options());

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
                  numIndices,
                  numBags,
                  featureSize,
                  weight.stride(0),
                  weight.stride(1),
                  bag_size.data_ptr<index_t>(),
                  mode == MODE_MAX ? max_indices.data_ptr<index_t>() : NULL,
                  per_sample_weights.defined()
                      ? per_sample_weights.data_ptr<scalar_t>()
                      : NULL,
                  per_sample_weights.defined() ? per_sample_weights.stride(0)
                                               : 0);
            });
      });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_dpcpp(
    const Tensor& grad_,
    const Tensor& indices,
    // const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    const Tensor& max_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights) {
  Tensor grad = grad_.contiguous();

  switch (mode) {
    case MODE_SUM:
    case MODE_MEAN:
      if (mode == MODE_MEAN)
        TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_dpcpp_sum_avg(
          grad,
          indices,
          // offsets,
          offset2bag,
          bag_size_,
          num_weights,
          scale_grad_by_freq,
          mode,
          per_sample_weights);

    case MODE_MAX:
      TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_dpcpp_max(grad, max_indices, num_weights);

    default:
      TORCH_CHECK(0, "Unknown mode for embedding_bag_backward_dpcpp ", mode);
  }
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
  // TODO: include_last_offset
  // TODO: padding_idx
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
      per_sample_weights);
}

Tensor _embedding_bag_dense_backward(
    const Tensor& grad,
    const Tensor& indices,
    // const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  // TODO: padding_idx
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return impl::_embedding_bag_dense_backward_dpcpp(
      grad,
      indices,
      // offsets,
      offset2bag,
      bag_size,
      maximum_indices,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights);
}

} // namespace AtenIpexTypeXPU
} // namespace at
