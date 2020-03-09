#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/DPCPPTensorUtils.h>
#include <core/SYCL.h>
#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLContext.h>

#include <utils/Numerics.h>

#include <functions/Atomics.h>

using namespace at::native;
namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

constexpr int64_t NROWS_PER_THREAD = 10;
constexpr int64_t WARP_SIZE = 64;

DP_DEF_K1(partials_per_segment_sycl);
DP_DEF_K1(partial_segment_offset_sycl);
DP_DEF_K2(compute_grad_weight_bags_sycl, typename scalar_t);
DP_DEF_K2(compute_grad_weight_sycl, typename scalar_t);
DP_DEF_K2(sum_and_scatter_sycl, typename scalar_t);

DP_DEF_K2(EmbeddingbagSycl, typename scalar_t);
DP_DEF_K2(AccGradParametersKernel_max_Sycl, typename scalar_t);

void krn_partials_per_segment(int64_t *ret, const int64_t *segment_offsets,
                              int64_t num_of_segments, int64_t numel) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  int64_t group_size = 32;
  auto num_groups    = CeilDiv(num_of_segments, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc_ret = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, ret);
    auto acc_offsets = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, segment_offsets);
    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto ret_ptr = acc_ret.template get_pointer<int64_t>();
      auto offsets_ptr = acc_offsets.template get_pointer<int64_t>();
      int64_t id = item.get_global_id(0);
      if (id < num_of_segments) {
        const int64_t idx_start = offsets_ptr[id];
        const int64_t idx_end = (id == num_of_segments - 1)? numel : offsets_ptr[id+1];
        const int64_t size = idx_end - idx_start;
        ret_ptr[id] = CeilDiv(size, NROWS_PER_THREAD);
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(partials_per_segment_sycl)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

void krn_partial_segment_offset(
        int64_t *ret,
        const int64_t *partials_per_segment,
        const int64_t *partials_per_segment_offset,
        const int64_t *segment_offsets,
        int64_t num_of_segments) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  int64_t group_size = 32;
  auto num_groups    = CeilDiv(num_of_segments, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc_ret = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, ret);
    auto acc_partials_per_segment = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, partials_per_segment);
    auto acc_partials_per_segment_offset = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, partials_per_segment_offset);
    auto acc_segment_offsets = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, segment_offsets);
    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto ret_ptr = acc_ret.template get_pointer<int64_t>();
      auto partials_per_segment_ptr = acc_partials_per_segment.template get_pointer<int64_t>();
      auto partials_per_segment_offset_ptr = acc_partials_per_segment_offset.template get_pointer<int64_t>();
      auto segment_offsets_ptr = acc_segment_offsets.template get_pointer<int64_t>();

      int64_t id = item.get_global_id(0);
      if (id < num_of_segments) {
        int64_t idx = partials_per_segment_offset_ptr[id];
        const int64_t num_partials = partials_per_segment_ptr[id];
        const int64_t segment_offset = segment_offsets_ptr[id];
        for (int64_t i=0; i<num_partials; ++i) {
          ret_ptr[idx++] = segment_offset + i * NROWS_PER_THREAD;
        }
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(partial_segment_offset_sycl)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

int64_t exclusive_scan(int64_t * out, int64_t * in, int64_t num_of_segments) {
  static const auto write_mode = cl::sycl::access::mode::write;
  static const auto read_mode = cl::sycl::access::mode::read;
  auto in_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(in);
  auto out_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(out);
  auto acc_in = in_ptr.get_access<read_mode>();
  auto acc_out = out_ptr.get_access<write_mode>();
  acc_out[0] = 0;
  for (int64_t i = 1; i < num_of_segments; i++) {
    acc_out[i] = acc_in[i-1] + acc_out[i - 1];
  }
  return acc_out[num_of_segments-1] + acc_in[num_of_segments-1];
}


template <typename scalar_t>
void compute_grad_weight_bags(
    int64_t *indices, scalar_t *gradOutput,
    int64_t *offset2bag, int64_t *count, int64_t numel,
    int64_t stride, int mode_mean, const int64_t *bag_size,
    scalar_t* per_sample_weights, int64_t per_sample_weights_stride,
    int64_t* segment_offsets, int64_t num_of_segments,
    acc_type<scalar_t, true> *grad_weight_per_segment,
    bool scale_grad_by_freq, bool per_sample_weight_defined) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  int64_t stride_warped = CeilDiv(stride, WARP_SIZE)*WARP_SIZE;
  int64_t group_size = std::min(stride_warped, c10::sycl::syclMaxWorkGroupSize(queue));
  auto num_groups    = CeilDiv(num_of_segments*stride_warped, group_size);
  auto total_items   = num_groups * group_size;
  DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1)); 

  auto cgf = DP_Q_CGF(cgh) {
    auto acc_grad_weight_per_segment = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, grad_weight_per_segment);
    auto acc_indices = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, indices);
    auto acc_gradOutput = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, gradOutput);
    auto acc_offset2bag = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, offset2bag);
    auto acc_count = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, count);
    auto acc_bag_size = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, bag_size);
    auto acc_per_sample_weights = per_sample_weight_defined ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, per_sample_weights) :
                                                       c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer);
    auto acc_segment_offsets = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, segment_offsets);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto grad_weight_per_segment_ptr = acc_grad_weight_per_segment.template get_pointer<acc_type<scalar_t, true>>();
      auto indices_ptr = acc_indices.template get_pointer<int64_t>();
      auto gradOutput_ptr = acc_gradOutput.template get_pointer<scalar_t>();
      auto offset2bag_ptr = acc_offset2bag.template get_pointer<int64_t>();
      auto count_ptr = acc_count.template get_pointer<int64_t>();
      auto bag_size_ptr = acc_bag_size.template get_pointer<int64_t>();
      auto per_sample_weights_ptr = per_sample_weight_defined ? acc_per_sample_weights.template get_pointer<scalar_t>() : NULL;
      auto segment_offsets_ptr = acc_segment_offsets.template get_pointer<int64_t>();

      const int gid = item.get_global_id(0);
      const int id = gid / stride_warped;
      const int startFeature = gid % stride_warped;
      if (startFeature >= stride) {
        return;
      }
      if (id >= num_of_segments) {
        return;
      }

      const int idx_begin = segment_offsets_ptr[id];
      const int idx_end = (id == num_of_segments-1) ? numel : segment_offsets_ptr[id+1];

      acc_type<scalar_t, true> weight = 0;
      for (int idx = idx_begin; idx < idx_end; ++idx) {
        const int seq_number = offset2bag_ptr[idx];
        const int gradOutputRow = seq_number * stride;

        acc_type<scalar_t, true> scale = scale_grad_by_freq ? 1.0 / count_ptr[indices_ptr[idx]] : 1.0;
        if (per_sample_weight_defined) {
          scale *= per_sample_weights_ptr[idx * per_sample_weights_stride];
        }

        acc_type<scalar_t, true> gradient = gradOutput_ptr[gradOutputRow + startFeature];
        if (mode_mean) {
          gradient /= bag_size_ptr[seq_number];
        }
        weight += gradient * scale;
      }
      grad_weight_per_segment_ptr[id * stride + startFeature] = weight;
    };

    // kick off kernel
    cgh.parallel_for<DP_K(compute_grad_weight_bags_sycl, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf);  
}

template <typename scalar_t>
void compute_grad_weight(
    int64_t *indices,
    int64_t *sort,
    scalar_t *gradOutput,
    int64_t *count,
    ptrdiff_t numel,
    int64_t stride,
    int64_t* segment_offsets,
    int64_t num_of_segments,
    acc_type<scalar_t, true> *grad_weight_per_segment,
    int padding_idx,
    bool scale_grad_by_fred) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  int64_t stride_warped = CeilDiv(stride, WARP_SIZE)*WARP_SIZE;
  int64_t group_size = std::min(stride_warped, c10::sycl::syclMaxWorkGroupSize(queue));
  auto num_groups    = CeilDiv(num_of_segments*stride_warped, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc_grad_weight_per_segment = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, grad_weight_per_segment);
    auto acc_indices = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, indices);
    auto acc_sort = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, sort);
    auto acc_gradOutput = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, gradOutput);
    auto acc_count = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, count);
    auto acc_segment_offsets = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, segment_offsets);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto grad_weight_per_segment_ptr = acc_grad_weight_per_segment.template get_pointer<acc_type<scalar_t, true>>();
      auto indices_ptr = acc_indices.template get_pointer<int64_t>();
      auto sort_ptr = acc_sort.template get_pointer<int64_t>();
      auto gradOutput_ptr = acc_gradOutput.template get_pointer<scalar_t>();
      auto count_ptr = acc_count.template get_pointer<int64_t>();
      auto segment_offsets_ptr = acc_segment_offsets.template get_pointer<int64_t>();

      const int gid = item.get_global_id(0);
      const int id = gid / stride_warped;
      const int startFeature = gid % stride_warped;
      if (startFeature >= stride) {
        return;
      }
      if (id >= num_of_segments) {
        return;
      }
      const int idx_begin = segment_offsets_ptr[id];
      const int idx_end = (id == num_of_segments - 1) ? numel : segment_offsets_ptr[id + 1];

      acc_type<scalar_t, true> weight = 0;
      for (int idx = idx_begin; idx < idx_end; idx++) {
        const int64_t target_row = sort_ptr[idx];
        if (target_row != padding_idx) {
          const acc_type<scalar_t, true> scale = scale_grad_by_fred ? 1.0 / count_ptr[indices_ptr[idx]] : 1.0;
          weight += gradOutput_ptr[target_row * stride + startFeature] * scale;
        }
      }
      grad_weight_per_segment_ptr[id * stride + startFeature] = weight;
    };

    // kick off kernel
    cgh.parallel_for<DP_K(compute_grad_weight_sycl, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void sum_and_scatter(
    int64_t *input, scalar_t *gradWeight, int64_t stride,
    int64_t* segment_offsets, int64_t num_of_segments,
    const acc_type<scalar_t, true> *grad_weight_per_segment,
    const int64_t *segment_sizes_offsets, int64_t num_of_partial_segments) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  int64_t stride_warped = CeilDiv(stride, WARP_SIZE)*WARP_SIZE;
  int64_t group_size = std::min(stride_warped, c10::sycl::syclMaxWorkGroupSize(queue));;
  auto num_groups    = CeilDiv(num_of_segments*stride_warped, group_size);
  auto total_items   = num_groups * group_size;
  
  auto cgf = DP_Q_CGF(cgh) {
    auto acc_gradWeight = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, gradWeight);
    auto acc_input = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, input);
    auto acc_segment_offsets = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, segment_offsets);
    auto acc_grad_weight_per_segment = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, grad_weight_per_segment);
    auto acc_segment_sizes_offsets = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, segment_sizes_offsets);
    
    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto gradWeight_ptr = acc_gradWeight.template get_pointer<scalar_t>();
      auto input_ptr = acc_input.template get_pointer<int64_t>();
      auto segment_offsets_ptr = acc_segment_offsets.template get_pointer<int64_t>();
      auto grad_weight_per_segment_ptr = acc_grad_weight_per_segment.template get_pointer<acc_type<scalar_t, true>>();
      auto segment_sizes_offsets_ptr = acc_segment_sizes_offsets.template get_pointer<int64_t>();
      
      const int gid = item.get_global_id(0);
      const int id = gid / stride_warped;
      const int startFeature = gid % stride_warped;
      if (startFeature >= stride) {
        return;
      }
      if (id >= num_of_segments) {
        return;
      }

      const int idx_begin = segment_sizes_offsets_ptr[id];
      const int idx_end = (id == num_of_segments-1) ? num_of_partial_segments : segment_sizes_offsets_ptr[id + 1];
      acc_type<scalar_t, true> weight = 0;
      for (int idx = idx_begin; idx < idx_end; idx++) {
        weight += grad_weight_per_segment_ptr[idx * stride + startFeature];
      }
      const int weightRow = input_ptr[segment_offsets_ptr[id]] * stride;
      gradWeight_ptr[weightRow + startFeature] = weight;
    };

    // kick off kernel
    cgh.parallel_for<DP_K(sum_and_scatter_sycl, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf);  
}

Tensor embedding_bag_backward_sycl_kernel(
        const Tensor &grad,
        const Tensor &sorted_indices,
        const Tensor &ind_sort,
        const Tensor &count,
        const Tensor &segment_offset,
        int64_t num_weights,
        int64_t num_segments,
        int padding_idx,
        bool scale_grad_by_freq,
        bool mode_mean,
        const Tensor &offset2bag,
        const Tensor &bag_size,
        const Tensor &per_sample_weights) {

  const int64_t numel = sorted_indices.numel();
  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);

  auto partials_per_segment = at::empty({num_segments}, sorted_indices.options());

  krn_partials_per_segment(partials_per_segment.data_ptr<int64_t>(), segment_offset.data_ptr<int64_t>(), num_segments, numel);
  auto partials_per_segment_offset = at::empty({num_segments}, sorted_indices.options());
  
  // The total number of partial-segments is the sum of `partials_per_segment_offset`
  auto num_of_partial_segments = exclusive_scan(partials_per_segment_offset.data_ptr<int64_t>(), partials_per_segment.data_ptr<int64_t>(), num_segments);

  auto partial_segment_offset = at::empty({num_of_partial_segments}, sorted_indices.options());
  krn_partial_segment_offset(partial_segment_offset.data_ptr<int64_t>(),
            partials_per_segment.data_ptr<int64_t>(),
            partials_per_segment_offset.data_ptr<int64_t>(),
            segment_offset.data_ptr<int64_t>(),
            num_segments);

  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "embedding_bag_backward_sycl_compute_grad_weight", [&] {
      // For numerical stability, the dtype of `grad_weight_per_segment`
      // should match `acc_type`
      using partial_weight_t = acc_type<scalar_t, true>;
      TensorOptions op;
      if(grad.dtype() == at::kHalf) {
          op = grad.options().dtype(at::kFloat);
      } else {
          op = grad.options();
      }
      auto grad_weight_per_segment = at::empty({num_of_partial_segments, stride}, op);
      // Compute the sum of each partial-segment and handle bags
      if (offset2bag.defined()) {
            compute_grad_weight_bags<scalar_t>(
              sorted_indices.data_ptr<int64_t>(),
              grad.data_ptr<scalar_t>(),
              offset2bag.data_ptr<int64_t>(),
              count.data_ptr<int64_t>(), numel, stride,
              mode_mean, bag_size.data_ptr<int64_t>(),
              per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() : NULL,
              per_sample_weights.defined() ? per_sample_weights.stride(0) : 0,
              partial_segment_offset.data_ptr<int64_t>(),
              num_of_partial_segments, grad_weight_per_segment.data_ptr<partial_weight_t>(),
              scale_grad_by_freq, per_sample_weights.defined());
      } else {
            compute_grad_weight<scalar_t>(
              sorted_indices.data_ptr<int64_t>(),
              ind_sort.data_ptr<int64_t>(),
              grad.data_ptr<scalar_t>(),
              count.data_ptr<int64_t>(),
              numel, stride,
              partial_segment_offset.data_ptr<int64_t>(),
              num_of_partial_segments,
              grad_weight_per_segment.data_ptr<partial_weight_t>(),
              padding_idx,
              scale_grad_by_freq);
      }

      // Finally, we sum all the partial-sums and scatter them
      // into `grad_weight`.
      
          sum_and_scatter<scalar_t>(
            sorted_indices.data_ptr<int64_t>(),
            grad_weight.data_ptr<scalar_t>(),
            stride,
            segment_offset.data_ptr<int64_t>(),
            num_segments, grad_weight_per_segment.data_ptr<partial_weight_t>(),
            partials_per_segment_offset.data_ptr<int64_t>(),
            num_of_partial_segments);
      
  });

  return grad_weight;
}


// This kernel assumes that all input tensors except `weight` and
// per_sample_weights are contiguous.
template <typename scalar_t>
void EmbeddingBag_updateOutputKernel(
    int64_t *input, int64_t *offsets, scalar_t *weight, scalar_t *output,
    int64_t *offset2bag, int64_t numIndices, int64_t numBags,
    int64_t featureSize, int64_t weight_stide0, int64_t weight_stride1,
    int mode, int64_t *bag_size, int64_t *max_indices,
    scalar_t* per_sample_weights, int64_t per_sample_weights_stride) {

  // the strategy here is that each bag x feature is handled by a single thread
  
  using accscalar_t = acc_type<scalar_t, true>;
  auto queue = c10::sycl::syclGetCurrentQueue();
  int64_t chunksPerBag = CeilDiv(featureSize, (int64_t)64);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t kernel_range = 1024 * 64;
  bool per_sample_weights_defined = per_sample_weights ? true : false;
  DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1)); 

  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, input);
    auto offset_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, offsets);
    auto weight_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, weight);
    auto output_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, output);
    auto offset2bag_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, offset2bag);
    auto bag_size_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, bag_size);
    auto per_sample_weights_acc = per_sample_weights_defined ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, per_sample_weights) :
                                                               c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer);
    auto max_indices_acc = mode == MODE_MAX ? c10::sycl::SYCLAccessor<dp_w_mode>(cgh, max_indices) :
                                              c10::sycl::SYCLAccessor<dp_w_mode>(cgh, dummy_buffer);
    
    auto kfn = DP_Q_KFN(DP::nd_item<2>item) {
      auto input_ptr = in_acc.template get_pointer<int64_t>();
      auto offsets_ptr = offset_acc.template get_pointer<int64_t>();
      auto weight_ptr = weight_acc.template get_pointer<scalar_t>();
      auto output_ptr = output_acc.template get_pointer<scalar_t>();
      auto offset2bag_ptr = offset2bag_acc.template get_pointer<int64_t>();
      auto bag_size_ptr = bag_size_acc.template get_pointer<int64_t>();
      auto per_sample_weights_ptr = per_sample_weights_defined ? per_sample_weights_acc.template get_pointer<scalar_t>() : NULL;
      auto max_indices_ptr = mode == MODE_MAX ? max_indices_acc.template get_pointer<int64_t>() : NULL;

      int64_t chunkOffset = item.get_group()[0] * item.get_local_range()[1] + item.get_local_id()[1];

      for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
        int64_t featureDim = (chunk % chunksPerBag) * item.get_local_range(0) + item.get_local_id(0);
        if (featureDim < featureSize) {
          int64_t bag = chunk / chunksPerBag;
          scalar_t *weightFeat = weight_ptr + featureDim * weight_stride1;
          int64_t begin = offsets_ptr[bag];
          int64_t end = (bag < numBags - 1) ? (offsets_ptr[bag + 1]) : numIndices;
          
          accscalar_t weightFeatSum = 0;
          scalar_t weightFeatMax;

          int64_t bag_size_ = 0;
          int64_t maxWord = -1;
          for (int64_t emb = begin; emb < end; emb++) {
            const int64_t weightRow = input_ptr[emb] * weight_stide0;
            scalar_t weightValue = weightFeat[weightRow];

            if (mode == MODE_MAX) {
              if (emb == begin || weightValue > weightFeatMax) {
                weightFeatMax = weightValue;
                maxWord = input_ptr[emb];
              }
            } else {
              if (per_sample_weights_defined) {
                accscalar_t scaleWeightBy = static_cast<accscalar_t>(
                    per_sample_weights_ptr[emb * per_sample_weights_stride]);
                weightFeatSum += scaleWeightBy * static_cast<accscalar_t>(weightValue);
              } else {
                weightFeatSum += static_cast<accscalar_t>(weightValue);
              }
            }

            bag_size_++;
            if (featureDim == 0) {
              offset2bag_ptr[emb] = bag;
            }
          }
          if (mode == MODE_MEAN) {
            if (end == begin) {
              bag_size_ptr[bag] = 0;
            } else {
              weightFeatSum = weightFeatSum / static_cast<accscalar_t>(bag_size_);
              bag_size_ptr[bag] = bag_size_;
            }
          }

          if (mode == MODE_MEAN || mode == MODE_SUM) {
            output_ptr[bag * featureSize + featureDim] = static_cast<scalar_t>(weightFeatSum);
          }
          else if (mode == MODE_MAX) {
            if (end == begin) {
              // If bag is empty, set output to 0.
              weightFeatMax = 0;
            }
            max_indices_ptr[bag * featureSize + featureDim] = maxWord;
            output_ptr[bag * featureSize + featureDim] = weightFeatMax;
          }
        }
      }
    };

  // kick off kernel
  cgh.parallel_for<DP_K(EmbeddingbagSycl, scalar_t)>(
    DP::nd_range<2>(DP::range<2>(kernel_range, 4), DP::range<2>(64, 4)), kfn);
};
DP_Q_ASYNC_SUBMIT(queue, cgf); 
}

void compute_counts(int64_t * counts, int64_t * indice, int64_t indice_length) {
  static const auto write_mode = cl::sycl::access::mode::write;
  static const auto read_mode = cl::sycl::access::mode::read;
  auto in_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(indice);
  auto co_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(counts);
  auto acc_in = in_ptr.get_access<read_mode>();
  auto acc_co = co_ptr.get_access<write_mode>();
  for (int64_t i = 0; i < indice_length; i++)
    acc_co[acc_in[i]]++;
}

// counts_uniq stores the index of the NEXT unique element
// of the (sorted) indices vector.
//
// For example:
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [0, 3, 4, 6, 7]
//
// The unique indices can be found at index 0, 3, 4, 6.

int64_t compute_counts_uniq(int64_t * counts_uniq, int64_t * indice, int64_t * counts, int64_t indices_length) {
  static const auto write_mode = cl::sycl::access::mode::write;
  static const auto read_mode = cl::sycl::access::mode::read;
  auto in_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(indice);
  auto co_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(counts);
  auto out_ptr = c10::sycl::syclGetBufferMap().template get_buffer<int64_t>(counts_uniq);
  auto acc_in = in_ptr.get_access<read_mode>();
  auto acc_co = co_ptr.get_access<read_mode>();
  auto acc_out = out_ptr.get_access<write_mode>();
  int64_t o = 1;
  acc_out[0] = 0;
  for (int64_t i = 0; i < indices_length; i += acc_co[acc_in[i]]) {
    acc_out[o] = acc_co[acc_in[i]];
    if (o > 1) {
      acc_out[o] += acc_out[o -1];
    }
    o++;
  }
  return o;
}

Tensor embedding_bag_backward_sycl_sum_avg(
                                   const Tensor &grad,
                                   const Tensor &indices_,
                                   const Tensor& offsets_,
                                   const Tensor &offset2bag__,
                                   const Tensor &bag_size,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode,
                                   const Tensor& per_sample_weights__) {

  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);
  
  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  Tensor per_sample_weights;
  if (per_sample_weights__.defined()) {
    Tensor &per_sample_weights_ = const_cast<Tensor &>(per_sample_weights__);
    per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
  }

  Tensor counts = at::zeros({num_weights}, indices.options());
  int64_t numel = indices.numel();
  compute_counts(counts.data_ptr<int64_t>(), indices.data_ptr<int64_t>(), numel);

  Tensor next_unique_index_idx = at::empty_like(indices);
  int64_t num_segments;
  num_segments = compute_counts_uniq(next_unique_index_idx.data_ptr<int64_t>(), indices.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), numel);

  return embedding_bag_backward_sycl_kernel(grad, indices, ind_sort,
      counts, next_unique_index_idx, num_weights, num_segments,
       /* padding_idx= */ -1, scale_grad_by_freq,
      mode == MODE_MEAN, offset2bag, bag_size, 
      per_sample_weights__.defined() ? per_sample_weights : per_sample_weights__);
}

template <typename scalar_t>
void EmbeddingBag_accGradParametersKernel_max(
    int64_t *max_indices, scalar_t *gradOutput,
    scalar_t *gradWeight, int64_t stride, int64_t numBags) {

  using accscalar_t = acc_type<scalar_t, true>;
  auto queue = c10::sycl::syclGetCurrentQueue();
  int64_t chunksPerBag = CeilDiv(stride, (int64_t)64);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t kernel_range = 1024 * 64;
  
  auto cgf = DP_Q_CGF(cgh) {
    auto max_indices_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, max_indices);
    auto gradOutput_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, gradOutput);
    auto gradWeight_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, gradWeight);
    
    auto kfn = DP_Q_KFN(DP::nd_item<2>item) {
      auto max_indices_ptr = max_indices_acc.template get_pointer<int64_t>();
      auto gradOutput_ptr = gradOutput_acc.template get_pointer<scalar_t>();
      auto gradWeight_ptr = gradWeight_acc.template get_pointer<scalar_t>();
      
      int64_t chunkOffset = item.get_group()[0] * item.get_local_range()[1] + item.get_local_id()[1];

      for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
        int64_t featureDim = (chunk % chunksPerBag) * item.get_local_range(0) + item.get_local_id(0);
        if (featureDim < stride) {
          int64_t bag = chunk / chunksPerBag;

          int64_t word_idx = max_indices_ptr[bag * stride + featureDim];
          if (word_idx >= 0) {
            // If bag is empty, we have max_indices[idx] set to -1 in forward.
            atomicAdd(&(gradWeight_ptr[word_idx * stride + featureDim]),
                    gradOutput_ptr[bag * stride + featureDim]);
          }
        }
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(AccGradParametersKernel_max_Sycl, scalar_t)>(
      DP::nd_range<2>(DP::range<2>(kernel_range, 4), DP::range<2>(64, 4)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf); 
}


Tensor embedding_bag_backward_sycl_max(const Tensor &grad,
                                   const Tensor &max_indices,
                                   int64_t num_weights) {

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  int64_t stride = grad_weight.stride(0);

  int64_t numBags = grad.size(0);

  // for atomicAdd, only support float datatype.
  EmbeddingBag_accGradParametersKernel_max<float>(
      max_indices.data_ptr<int64_t>(), grad.data_ptr<float>(),
      grad_weight.data_ptr<float>(), stride, numBags);

  return grad_weight;
}


// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_sycl(const Tensor &weight, const Tensor &indices,
                   const Tensor &offsets, const bool scale_grad_by_freq,
                   const int64_t mode, bool sparse,
                   const Tensor& per_sample_weights) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag_sycl", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag_sycl", offsets_arg, kLong);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkSameDPCPP("embedding_bag_sycl", weight_arg, indices_arg);
  checkSameDPCPP("embedding_bag_sycl", weight_arg, offsets_arg);
  
  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);
  int64_t featureSize = weight.size(1);
  
  auto bag_size = at::zeros(offsets.sizes(), indices.options());
  auto offset2bag =
      at::zeros({indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]

  auto output = at::zeros({offsets.size(0), weight.size(1)}, weight.options());
  
  Tensor max_indices;
  
  max_indices = at::zeros({offsets.size(0), weight.size(1)}, indices.options());
  
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_bag_sycl", [&] {
    EmbeddingBag_updateOutputKernel<scalar_t>(
        indices.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        offset2bag.data_ptr<int64_t>(), numIndices, numBags, featureSize,
        weight.stride(0), weight.stride(1), mode, bag_size.data_ptr<int64_t>(),
        mode == MODE_MAX ? max_indices.data_ptr<int64_t>() : NULL,
        per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() : NULL,
        per_sample_weights.defined() ? per_sample_weights.stride(0) : 0);
  });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_sycl(const Tensor &grad_, const Tensor &indices,
                                   const Tensor &offsets,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size_,
                                   const Tensor &max_indices,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode,
                                   const Tensor& per_sample_weights) {
  Tensor grad = grad_.contiguous();                                   
  
  switch (mode) {
    case MODE_SUM:
    case MODE_MEAN:
      if (mode == MODE_MEAN)
        AT_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_sycl_sum_avg(grad, indices, offsets, offset2bag,
              bag_size_, num_weights, scale_grad_by_freq, mode, per_sample_weights);

    case MODE_MAX:
      AT_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_sycl_max(grad, max_indices, num_weights);

    default:
      AT_ERROR(
          "Unknown mode for embedding_bag_backward_sycl ", mode);
  }
}

} // impl

std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight,
    const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq,
    int64_t mode, bool sparse, const Tensor & per_sample_weights) {
return impl::_embedding_bag_sycl(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
}

Tensor _embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices,
    const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size,
    const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq,
    int64_t mode, const Tensor & per_sample_weights) {
return  impl::_embedding_bag_dense_backward_sycl(grad, indices, offsets, offset2bag, bag_size, maximum_indices,
            num_weights, scale_grad_by_freq, mode, per_sample_weights);
}


} // namespace AtenIpexTypeDPCPP
} // namespace at
