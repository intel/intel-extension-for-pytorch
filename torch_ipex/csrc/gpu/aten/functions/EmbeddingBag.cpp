#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/DPCPPTensorUtils.h>
#include <core/SYCL.h>
#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLContext.h>
#include <utils/Numerics.h>


namespace at {
namespace native {


constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

DP_DEF_K2(EmbeddingbagSycl, typename scalar_t);

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
  int64_t chunksPerBag = CeilDiv(featureSize, (int64_t)32);
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



}
}
