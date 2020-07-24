#include <ATen/ATen.h>

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/TensorImplUtils.h>
#include <core/detail/IndexUtils.h>
#include <utils/ATDispatch.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

#ifdef USE_USM
constexpr int CAT_ARRAY_BATCH_SIZE = 1024;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 3;

// Similar to any other IndexToOffset calculation for copying along a given
// dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline IndexType compute(
      const IndexType outputSize[Dims],
      const IndexType outputStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    // linearIndex is not really linear index, but instead the offset in
    // input tensor. If the input tensor is contiguous, then this offset
    // is the linear index, but if the input tensor is channels last, then
    // it is the linear index of the permuted contiguous tensor
    IndexType offset = 0;

#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : outputSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * outputStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * outputStride[0];
  }
};

template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template<typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
};

DPCPP_DEF_K2(CatArrayBatchKer, typename T, typename IndexType, int Dims);
/**
  * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a
  * grid-stride loop based off of the blockIdx.x, threadIdx.x for each input to
  * copy each element from each input tensor into the output.
  *
  * output: base pointer to the storage associated with the output tensor
  * inputs: GPU-allocated array of input metadata for each input to concatenate
  *         in the kernel
  * os: the size/stride vectors for the output tensor
  * concatDim: dimension along which we are concatenating
  * dimStride: the stride of the output tensor at the concatDim
  *
  * The most important assumption made is that the input tensors are contiguous.
  */
template <typename T, typename IndexType, int Dims>
void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride,
    int batchCounter) {

  auto queue = dpcppGetCurrentQueue();

  //Get grid where x dim fills half gpu and y dim is number of tensors.
  //This will have cating two tensors fill the entire grid, but prevent
  //many threads from needlessly load meta data if their sizes is small.

  auto numCU = queue.get_device().get_info<DPCPP::info::device::max_compute_units>();
  auto numWI = queue.get_device().get_info<DPCPP::info::device::max_work_group_size>();
  DPCPP::range<2> global_range(numCU * numWI / 2, batchCounter);
  DPCPP::range<2> local_range(numWI, 1);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item) {
      IndexType wg = item.get_group(0);
      IndexType wg_size = item.get_local_range(0);
      IndexType wi = item.get_local_id(0);
      IndexType tid = wg * wg_size + wi;
      IndexType in = item.get_group(1);

      IndexType nElements = inputs[in].nElements;

      if(tid >= nElements) return;

      T* data = inputs[in].input;
      IndexType offset = inputs[in].offset;
      IndexType dimSize = inputs[in].dimSize;
      IndexType dataOffset = offset * dimStride;

      IndexType stride = item.get_group_range(0) * wg_size;

      while (tid < nElements) {
        IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                      os.outputSize, os.outputStride, dimSize, concatDim, tid);
        output[dataOffset + elementOffset] = data[tid];

        tid += stride;
      }
    };
    cgh.parallel_for<DPCPP_K(CatArrayBatchKer, T, IndexType, Dims)>(
        DPCPP::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(queue, cgf)
}

template <typename scalar_t>
void parallel_cat(Tensor &out, const TensorList &inputs,
                  int64_t dimension, int nDims) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t *data = out.data_ptr<scalar_t>();

  // Kernel Parameter
  long tensorMetadataSize =
    sizeof(CatArrInputTensor<scalar_t, unsigned int>) * CAT_ARRAY_BATCH_SIZE;
  auto d_inputs_storage = at::empty(
    {tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(
    d_inputs_storage.data_ptr());

  OutputTensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> param;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  for (int i = 0; i < nDims; ++i) {
    param.outputSize[i] = at::native::size(out, i);
    param.outputStride[i] = out.stride(i);
  }

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size() ; i += CAT_ARRAY_BATCH_SIZE) {
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      auto stackInputs_storage = at::empty({tensorMetadataSize},
          out.options().dtype(at::kByte).device(at::kCPU));
      auto stackInputs =
        static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(
          stackInputs_storage.data_ptr());
      for (batchCounter = 0;
           batchCounter < CAT_ARRAY_BATCH_SIZE &&
             (i+batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize = at::native::size(inputs[i+batchCounter], dimension);

        stackInputs[batchCounter].input =
          inputs[i+batchCounter].data_ptr<scalar_t>();
        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements = inputs[i+batchCounter].numel();

        // update offset
        offset += dimSize;
      }
      d_inputs_storage.copy_(stackInputs_storage);
    }

#define HANDLE_CASE(DIMS) \
    CatArrayBatchedCopy<scalar_t, unsigned int, DIMS>(\
            data, d_inputs, param, dimension, param.outputStride[dimension],\
            batchCounter);
    switch (nDims) {
      case 1:
        HANDLE_CASE(1);
        break;
      case 2:
        HANDLE_CASE(2);
        break;
      case 3:
        HANDLE_CASE(3);
        break;
      default:
        break;
    }
#undef HANDLE_CASE
  }
}
#endif

void check_shape_except_dim(Tensor& first, Tensor& second, int dimension) {
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(
      first_dims == second_dims, "Tensors must have same number of dimensions");
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.size(dim);
    int64_t second_dim_size = second.size(dim);
    TORCH_CHECK(
        first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension");
  }
}

static void cat(
    Tensor& result,
    TensorList inputs,
    int numInputs,
    int dimension) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible
  // to cat empty tensors unless all the other tensors were 1-dimensional, so we
  // allowed these tensors
  // to be "skipped".  We maintain this behavior for backwards compatibility,
  // but only for this specific
  // size (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  int i, j;
  int64_t offset;
  bool hasSkippedInput = false;
  Tensor notSkippedTensor; // non-owning reference
  auto should_skip = [](const Tensor& t) {
    return !t.defined() && t.dim() == 1;
  };
  int nDims = 0;

  for (i = 0; i < numInputs; i++) {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i].dim();
    notSkippedTensor = inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (!notSkippedTensor.defined()) {
    return;
  }

  TORCH_CHECK(numInputs > 0, "invalid number of inputs");
  TORCH_CHECK(dimension >= 0, "invalid dimension");

  std::vector<int64_t> size(nDims);

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    Tensor tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    check_shape_except_dim(notSkippedTensor, tensor, dimension);
    cat_dim_size += tensor.size(dimension);
  }

  // Compute the size of the result
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor.size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  result.resize_(size);

  const bool all32BitIndexable = std::all_of(inputs.begin(), inputs.end(),
    [] (const Tensor& t) {
      return dpcpp::detail::canUse32BitIndexMath(t);
    });
  const bool allContiguous = std::all_of(inputs.begin(), inputs.end(),
    [](const Tensor& t) {
      return !t.defined() || t.is_contiguous();
    });

#ifdef USE_USM
  if (inputs.size() > 1 &&
      !hasSkippedInput &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      dpcpp::detail::canUse32BitIndexMath(result) &&
      allContiguous &&
      all32BitIndexable) {
    IPEX_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
        result.scalar_type(), "cat_dpcpp", [&]() {
      parallel_cat<scalar_t>(result, inputs, dimension, nDims);
    });
  } else {
#endif
    offset = 0;
    for (j = 0; j < numInputs; j++) {
      if (should_skip(inputs[j]))
        continue;
      int64_t dimSize = inputs[j].size(dimension);
      Tensor nt = at::narrow(result, dimension, offset, dimSize);
      nt.copy_(inputs[j]);
      offset += dimSize;
    }
#ifdef USE_USM
  }
#endif
}

} // namespace impl

Tensor& _cat_out(Tensor& out, TensorList tensors, int64_t dim) {
  impl::cat(out, tensors, tensors.size(), dim);
  return out;
}

Tensor _cat(TensorList tensors, int64_t dim) {
  auto out = at::empty({0}, tensors[0].options());
  return at::AtenIpexTypeDPCPP::_cat_out(out, tensors, dim);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
