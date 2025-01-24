#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <core/Device.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <torch/torch.h>
#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include "utils/CustomOperatorRegistration.h"
#endif
#include <utils/DPCPP.h>

#include "BitonicMergeSort.h"
#include "MemoryAccess.h"
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "EmbeddingBackwardKernel.h"
#include "EmbeddingBagKernel.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

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
    sycl::nd_item<1> item) {
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

template <
    int vec_size,
    typename vec_t,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
struct EmbeddingBagUpdateOutputKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    vec_chunk_kernel_embeddingbag<
        vec_size,
        vec_t,
        scalar_t,
        accscalar_t,
        index_t>(
        mode,
        input,
        offset,
        weight,
        output,
        offset2bag,
        bag_size,
        per_sample_weights_defined,
        per_sample_weights,
        per_sample_weights_stride,
        max_indices,
        WGNumber,
        numBags,
        weight_total_elem,
        chunk_size,
        bag_chunk_num,
        bag_wi_num,
        bagsPerLoop,
        input_length,
        weight_stride0,
        weight_stride1,
        include_last_offset,
        padding_idx,
        if_align_vector,
        item);
  }
  EmbeddingBagUpdateOutputKernelFunctor(
      const int64_t mode_,
      index_t* input_,
      index_t* offset_,
      scalar_t* weight_,
      scalar_t* output_,
      index_t* offset2bag_,
      index_t* bag_size_,
      bool per_sample_weights_defined_,
      scalar_t* per_sample_weights_,
      int64_t per_sample_weights_stride_,
      index_t* max_indices_,
      int64_t WGNumber_,
      int64_t numBags_,
      int64_t weight_total_elem_,
      int64_t chunk_size_,
      int64_t bag_chunk_num_,
      int64_t bag_wi_num_,
      int64_t bagsPerLoop_,
      int64_t input_length_,
      int64_t weight_stride0_,
      int64_t weight_stride1_,
      bool include_last_offset_,
      index_t padding_idx_,
      bool if_align_vector_,
      sycl::nd_item<1> item_)
      : mode(mode_),
        input(input_),
        offset(offset_),
        weight(weight_),
        output(output_),
        offset2bag(offset2bag_),
        bag_size(bag_size_),
        per_sample_weights_defined(per_sample_weights_defined_),
        per_sample_weights(per_sample_weights_),
        per_sample_weights_stride(per_sample_weights_stride_),
        max_indices(max_indices_),
        WGNumber(WGNumber_),
        numBags(numBags_),
        weight_total_elem(weight_total_elem_),
        chunk_size(chunk_size_),
        bag_chunk_num(bag_chunk_num_),
        bag_wi_num(bag_wi_num_),
        bagsPerLoop(bagsPerLoop_),
        input_length(input_length_),
        weight_stride0(weight_stride0_),
        weight_stride1(weight_stride1_),
        include_last_offset(include_last_offset_),
        padding_idx(padding_idx_),
        if_align_vector(if_align_vector_),
        item(item_) {}

 private:
  const int64_t mode;
  index_t* input;
  index_t* offset;
  scalar_t* weight;
  scalar_t* output;
  index_t* offset2bag;
  index_t* bag_size;
  bool per_sample_weights_defined;
  scalar_t* per_sample_weights;
  int64_t per_sample_weights_stride;
  index_t* max_indices;
  int64_t WGNumber;
  int64_t numBags;
  int64_t weight_total_elem;
  int64_t chunk_size;
  int64_t bag_chunk_num;
  int64_t bag_wi_num;
  int64_t bagsPerLoop;
  int64_t input_length;
  int64_t weight_stride0;
  int64_t weight_stride1;
  const bool include_last_offset;
  const index_t padding_idx;
  const bool if_align_vector;
  sycl::nd_item<1> item;
};

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
    const index_t padding_idx,
    const bool ignore_offsets) {
  using accscalar_t = acc_type<scalar_t>;

  // vector size, query it according to machine, scalar_t and weight_data
  auto& queue = dpcppGetCurrentQueue();
  auto vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(weight_data));

  // determine per sample weights should be in calculation or not
  bool per_sample_weights_defined = per_sample_weights_data ? true : false;

  auto maxWGSize = dpcppMaxWorkGroupSize();

  auto gpuEuCount = dpcppGpuEuCount();

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
  auto WGNumber = gpuEuCount * 8 * 32 / maxWGSize * 512;

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
      EmbeddingBagUpdateOutputKernelFunctor<                                  \
          vec_size,                                                           \
          vec_t,                                                              \
          scalar_t,                                                           \
          accscalar_t,                                                        \
          index_t>                                                            \
          kfn(mode,                                                           \
              input,                                                          \
              offset,                                                         \
              weight,                                                         \
              output,                                                         \
              offset2bag,                                                     \
              bag_size,                                                       \
              per_sample_weights_defined,                                     \
              per_sample_weights,                                             \
              per_sample_weights_stride,                                      \
              max_indices,                                                    \
              WGNumber,                                                       \
              numBags,                                                        \
              weight_total_elem,                                              \
              chunk_size,                                                     \
              bag_chunk_num,                                                  \
              bag_wi_num,                                                     \
              bagsPerLoop,                                                    \
              input_length,                                                   \
              weight_stride0,                                                 \
              weight_stride1,                                                 \
              include_last_offset,                                            \
              padding_idx,                                                    \
              if_align_vector,                                                \
              item);                                                          \
      cgh.parallel_for<decltype(kfn)>(                                        \
          sycl::nd_range<1>(                                                  \
              sycl::range<1>(global_range), sycl::range<1>(local_range)),     \
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
  isOnSameDevice("embedding_bag_dpcpp", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  isOnSameDevice("embedding_bag_dpcpp", weight_arg, indices_arg);
  isOnSameDevice("embedding_bag_dpcpp", weight_arg, offsets_arg);

  bool ignore_offsets = indices.sizes().size() == 2;
  int64_t numIndices = indices.numel();
  int64_t numBags = ignore_offsets ? indices.size(0) : offsets.size(0);

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

#ifndef VEC_EMBBAG_KERNEL_OPT
#define EXTEND_EMBBAG_TEMPLATE(mode) \
  embedding_bag_##mode##_template(   \
      indices,                       \
      offsets,                       \
      weight,                        \
      per_sample_weights,            \
      output,                        \
      offset2bag,                    \
      bag_size,                      \
      max_indices,                   \
      numIndices,                    \
      numBags,                       \
      weight.stride(0),              \
      padding_idx,                   \
      ignore_offsets)

  switch (mode) {
    case MODE_SUM:
      EXTEND_EMBBAG_TEMPLATE(sum);
      break;
    case MODE_MEAN:
      EXTEND_EMBBAG_TEMPLATE(mean);
      break;
    case MODE_MAX:
      EXTEND_EMBBAG_TEMPLATE(max);
      break;
    default:
      TORCH_CHECK(0, "Invalid EmbeddingBag mode (max, sum, mean) ...");
  };
#else
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
                  padding_idx,
                  ignore_offsets);
            });
      });
#endif

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
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
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices.dim());
  if (indices.dim() == 1) {
    TORCH_CHECK(
        offsets.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets.dim());
  }
  TORCH_CHECK(
      weight.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight.dim());

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

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices.dim());
  if (indices.dim() == 1) {
    TORCH_CHECK(
        offsets.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets.dim());
  }
  TORCH_CHECK(
      weight.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight.dim());
  // See [Note: hacky wrapper removal for optional tensor]
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

} // namespace AtenIpexTypeXPU
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
wrapper_XPU___embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, weight, "wrapper_XPU___embedding_bag", "weight");
  c10::impl::check_and_update_common_device(
      common_device, indices, "wrapper_XPU___embedding_bag", "indices");
  c10::impl::check_and_update_common_device(
      common_device, offsets, "wrapper_XPU___embedding_bag", "offsets");
  c10::impl::check_and_update_common_device(
      common_device,
      per_sample_weights,
      "wrapper_XPU___embedding_bag",
      "per_sample_weights");
  const OptionalDeviceGuard device_guard(device_of(weight));

  return at::AtenIpexTypeXPU::_embedding_bag(
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

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
wrapper_XPU___embedding_bag_forward_only(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      weight,
      "wrapper_XPU___embedding_bag_forward_only",
      "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      indices,
      "wrapper_XPU___embedding_bag_forward_only",
      "indices");
  c10::impl::check_and_update_common_device(
      common_device,
      offsets,
      "wrapper_XPU___embedding_bag_forward_only",
      "offsets");
  c10::impl::check_and_update_common_device(
      common_device,
      per_sample_weights,
      "wrapper_XPU___embedding_bag_forward_only",
      "per_sample_weights");
  const OptionalDeviceGuard device_guard(device_of(weight));

  return at::AtenIpexTypeXPU::_embedding_bag_forward_only(
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
IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("_embedding_bag", TORCH_FN((&wrapper_XPU___embedding_bag)));
  m.impl(
      "_embedding_bag_forward_only",
      TORCH_FN((&wrapper_XPU___embedding_bag_forward_only)));
}

} // namespace
#endif
