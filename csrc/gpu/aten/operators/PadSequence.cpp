#include <ATen/ATen.h>
#include <aten/core/HostAllocator.h>
#include <core/detail/IndexUtils.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Memory.h>
#include <runtime/Utils.h>
#include <torch/custom_class.h>
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

constexpr int PAD_ARRAY_BATCH_SIZE = 1024;
constexpr int PAD_ARRAY_MAX_INPUT_DIMS = 4;

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

template <typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
};

/**
 * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses
 a
 * grid-stride loop based off of the blockIdx.x, threadIdx.x for each input
 to
 * copy each element from each input tensor into the output.
 *
 * output: base pointer to the storage associated with the output tensor
 * inputs: GPU-allocated array of input metadata for each input to
 concatenate
 *         in the kernel
 * os: the size/stride vectors for the output tensor
 * concatDim: dimension along which we are concatenating
 * dimStride: the stride of the output tensor at the concatDim
 *
 * The most important assumption made is that the input tensors are
 contiguous.
 */
template <typename T, typename IndexType, int Dims>
void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, PAD_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride,
    int batchCounter) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  // Get grid where x dim fills half gpu and y dim is number of tensors.
  // This will have cating two tensors fill the entire grid, but prevent
  // many threads from needlessly load meta data if their sizes is small.
  auto numWI = dpcppMaxWorkGroupSize(dev_id);

  // We set limited numWG to prevent over schedule.
  // numWG = 512 EUs * 8 threads * SIMD lanes 32 / max_compute_units
  // (1024 on PVC).
  // When input tensors less than 32, we choose 128 numWG to handle a tensor,
  // then we have one tile per tensor.
  // When input tensors more than 32, we choose 64 numWG to handle a tensor,
  // half tile per tensor, the other half is occupied by next input tensor.
  int64_t numWG;
  if (batchCounter > 32)
    numWG = 64;
  else
    numWG = 128;
  sycl::range<2> global_range(batchCounter, numWG * numWI);
  sycl::range<2> local_range(1, numWI);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      IndexType tid = item.get_global_id(1);
      IndexType in = item.get_group(0);

      IndexType nElements = inputs[in].nElements;

      if (tid >= nElements)
        return;

      T* data = inputs[in].input;
      IndexType offset = inputs[in].offset;
      IndexType dimSize = inputs[in].dimSize;
      IndexType dataOffset = offset * dimStride;

      IndexType stride = item.get_global_range(1);

      while (tid < nElements) {
        IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
            os.outputSize, os.outputStride, dimSize, concatDim, tid);
        output[dataOffset + elementOffset] = data[tid];

        tid += stride;
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf)
}

template <typename scalar_t>
void parallel_pad(
    const Tensor& out,
    const TensorList& inputs,
    int nDims,
    bool batch_first) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t* data = out.data_ptr<scalar_t>();

  // Kernel Parameter
  long tensorMetadataSize =
      sizeof(CatArrInputTensor<scalar_t, unsigned int>) * PAD_ARRAY_BATCH_SIZE;
  auto d_inputs_storage =
      at::empty({tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_t, unsigned int>*>(
      d_inputs_storage.data_ptr());

  OutputTensorSizeStride<unsigned int, PAD_ARRAY_MAX_INPUT_DIMS> param;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  for (int i = 0; i < nDims; ++i) {
    param.outputSize[i] = at::native::size(out, i);
    param.outputStride[i] = out.stride(i);
  }

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  int dimension = 0;
  for (int i = 0; i < inputs.size(); i += PAD_ARRAY_BATCH_SIZE) {
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      CatArrInputTensor<scalar_t, unsigned int>* stackInputs;
      stackInputs = (CatArrInputTensor<scalar_t, unsigned int>*)
                        xpu::dpcpp::HostAllocator::Instance()
                            ->raw_allocate(tensorMetadataSize);
      for (batchCounter = 0; batchCounter < PAD_ARRAY_BATCH_SIZE &&
           (i + batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize = at::native::size(out, dimension + 1);

        stackInputs[batchCounter].input =
            inputs[i + batchCounter].data_ptr<scalar_t>();
        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements = inputs[i + batchCounter].numel();

        // update offset
        offset += dimSize;
      }
      xpu::dpcpp::memcpyHostToDevice(
          d_inputs, stackInputs, tensorMetadataSize, /* async= */ true);
      xpu::dpcpp::HostAllocator::Instance()->release(stackInputs);
    }

#define HANDLE_CASE(DIMS)                            \
  CatArrayBatchedCopy<scalar_t, unsigned int, DIMS>( \
      data,                                          \
      d_inputs,                                      \
      param,                                         \
      dimension,                                     \
      param.outputStride[dimension + 1],             \
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
      case 4:
        HANDLE_CASE(4);
        break;
      default:
        break;
    }
#undef HANDLE_CASE
  }
}
} // namespace impl

Tensor pad_sequence(
    TensorList sequences,
    bool batch_first,
    double padding_value) {
  const int64_t sequences_size = sequences.size();
  TORCH_CHECK(sequences_size > 0, "received an empty list of sequences");
  IntArrayRef max_size = sequences[0].sizes();
  IntArrayRef trailing_dims = max_size.slice(1);
  int64_t max_len = std::max_element(
                        sequences.begin(),
                        sequences.end(),
                        [](const Tensor& a, const Tensor& b) {
                          return a.size(0) < b.size(0);
                        })
                        ->size(0);

  DimVector out_dims;
  if (batch_first) {
    out_dims = {sequences_size, max_len};
  } else {
    out_dims = {max_len, sequences_size};
  }

  out_dims.insert(out_dims.end(), trailing_dims.begin(), trailing_dims.end());

  // full with padding_value in out tensor (B*T OR T*B) * X
  Tensor out = at::full(out_dims, padding_value, sequences[0].options());

  ScalarType firstType = sequences[0].scalar_type();
  bool allSameType = std::all_of(
      sequences.begin(), sequences.end(), [firstType](const Tensor& t) {
        return t.scalar_type() == firstType;
      });

  bool isBlockfmt =
      std::any_of(sequences.begin(), sequences.end(), [](const Tensor& t) {
        return xpu::oneDNN::is_onednn_layout(t);
      });
  const bool all32BitIndexable =
      std::all_of(sequences.begin(), sequences.end(), [](const Tensor& t) {
        return xpu::dpcpp::detail::canUse32BitIndexMath(t);
      });
  const bool allContiguous =
      std::all_of(sequences.begin(), sequences.end(), [](const Tensor& t) {
        return !t.defined() || t.is_contiguous();
      });

  if (sequences.size() > 1 && allSameType && !isBlockfmt && all32BitIndexable &&
      allContiguous && out.dim() <= impl::PAD_ARRAY_MAX_INPUT_DIMS &&
      batch_first) {
    int nDims = out.dim();
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "pad_dpcpp",
        [&]() {
          impl::parallel_pad<scalar_t>(out, sequences, nDims, batch_first);
        });
  } else {
    for (const auto i : c10::irange(sequences_size)) {
      const Tensor currseq = sequences[i];
      const int64_t length_i = currseq.size(0);
      // use index notation to prevent duplicate references to the tensor
      if (batch_first) {
        out.select(0, i).narrow(0, 0, length_i).copy_(currseq);
      } else {
        out.narrow(0, 0, length_i).select(1, i).copy_(currseq);
      }
    }
  }
  return out;
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("pad_sequence", TORCH_FN(pad_sequence));
}

} // namespace AtenIpexTypeXPU
} // namespace at
