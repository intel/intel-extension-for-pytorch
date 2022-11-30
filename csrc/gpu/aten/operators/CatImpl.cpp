#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TypeProperties.h>

#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/CachingHostAllocator.h>
#include <runtime/Memory.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"

#include <core/Memory.h>
#include <core/MemoryFormat.h>
#include <tensor/Context.h>
#include "CatImpl.h"
#include "comm/Numerics.h"
#include "comm/zmath.h"

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// The best performance is achieved for parallel computing with 1024 batch sizes
// at a time.
constexpr int CAT_ARRAY_BATCH_SIZE = 1024;

// Maximum parallel dimension to supporte
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;

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
  float scale_in;
  int zero_point_in;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template <typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  float scale_out;
  int zero_point_out;
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
};

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
template <
    typename Tout,
    typename underlying_out_t,
    typename Tin,
    typename underlying_in_t,
    typename IndexType,
    int Dims,
    bool is_quantized_out = false,
    bool is_quantized_in = false>
void CatArrayBatchedCopy(
    Tout* output,
    CatArrInputTensor<Tin, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
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

      Tin* data = inputs[in].input;
      IndexType offset = inputs[in].offset;
      IndexType dimSize = inputs[in].dimSize;
      IndexType dataOffset = offset * dimStride;

      IndexType stride = item.get_global_range(1);

      while (tid < nElements) {
        IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
            os.outputSize, os.outputStride, dimSize, concatDim, tid);
        if constexpr (is_quantized_out) {
          // this path handles case of qtype/float inputs + qtype output
          auto out_float = static_cast<float>(
              round_impl<float>(
                  (static_cast<float>(data[tid]) - inputs[in].zero_point_in) *
                  inputs[in].scale_in / os.scale_out) +
              os.zero_point_out);
          auto out = Numerics<float>::min(
              Numerics<float>::max(
                  out_float,
                  static_cast<float>(
                      std::numeric_limits<underlying_out_t>::min())),
              static_cast<float>(std::numeric_limits<underlying_out_t>::max()));
          output[dataOffset + elementOffset] = static_cast<Tout>(out);
        } else if constexpr (is_quantized_in) {
          // this path handles cases of qtype input + float output
          output[dataOffset + elementOffset] =
              (static_cast<float>(data[tid]) - inputs[in].zero_point_in) *
              inputs[in].scale_in;
        } else {
          // this path handles cases of non-qtype
          output[dataOffset + elementOffset] = data[tid];
        }
        tid += stride;
      }
    };
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf)
}

template <
    typename scalar_out_t,
    typename underlying_out_t,
    typename scalar_in_t,
    typename underlying_in_t,
    bool is_quantized_out = false,
    bool is_quantized_in = false>
void parallel_cat(
    const Tensor& out,
    const MaterializedITensorListRef& inputs,
    int64_t dimension,
    int nDims) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_out_t* data = static_cast<scalar_out_t*>(out.data_ptr());

  // Kernel Parameter
  long tensorMetadataSize =
      sizeof(CatArrInputTensor<scalar_in_t, unsigned int>) *
      CAT_ARRAY_BATCH_SIZE;
  auto d_inputs_storage =
      at::empty({tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_in_t, unsigned int>*>(
      d_inputs_storage.data_ptr());

  OutputTensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> param;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  param.zero_point_out = out.is_quantized() ? out.q_zero_point() : 0;
  param.scale_out = AsignOneDnnQuantizeScale(out, 1.0f, param.zero_point_out);
  if ((at::kQInt8 == out.scalar_type() || at::kQUInt8 == out.scalar_type()) &&
      (128 == param.zero_point_out)) {
    param.zero_point_out = 0;
  }

  for (int i = 0; i < nDims; ++i) {
    param.outputSize[i] = at::native::size(out, i);
    param.outputStride[i] = out.stride(i);
  }

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size(); i += CAT_ARRAY_BATCH_SIZE) {
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      CatArrInputTensor<scalar_in_t, unsigned int>* stackInputs;
      CachingHostAllocator::Instance()->malloc(
          (void**)&stackInputs, tensorMetadataSize);
      for (batchCounter = 0; batchCounter < CAT_ARRAY_BATCH_SIZE &&
           (i + batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize =
            at::native::size(inputs[i + batchCounter].get(), dimension);

        stackInputs[batchCounter].input = static_cast<scalar_in_t*>(
            inputs[i + batchCounter].get().data_ptr());
        stackInputs[batchCounter].zero_point_in =
            inputs[i + batchCounter].get().is_quantized()
            ? inputs[i + batchCounter].get().q_zero_point()
            : 0;
        stackInputs[batchCounter].scale_in = AsignOneDnnQuantizeScale(
            inputs[i + batchCounter].get(),
            1.0f,
            stackInputs[batchCounter].zero_point_in);
        if ((at::kQInt8 == inputs[i + batchCounter].get().scalar_type() ||
             at::kQUInt8 == inputs[i + batchCounter].get().scalar_type()) &&
            (128 == stackInputs[batchCounter].zero_point_in)) {
          stackInputs[batchCounter].zero_point_in = 0;
        }

        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements =
            inputs[i + batchCounter].get().numel();

        // update offset
        offset += dimSize;
      }
      xpu::dpcpp::memcpyHostToDevice(
          d_inputs, stackInputs, tensorMetadataSize, /* async= */ true);
      CachingHostAllocator::Instance()->release(stackInputs);
    }

#define HANDLE_CASE(DIMS)            \
  CatArrayBatchedCopy<               \
      scalar_out_t,                  \
      underlying_out_t,              \
      scalar_in_t,                   \
      underlying_in_t,               \
      unsigned int,                  \
      DIMS,                          \
      is_quantized_out,              \
      is_quantized_in>(              \
      data,                          \
      d_inputs,                      \
      param,                         \
      dimension,                     \
      param.outputStride[dimension], \
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
    MaterializedITensorListRef inputs,
    int numInputs,
    int dimension,
    bool allSameType) {
  int i, j;
  int64_t offset;
  bool hasSkippedInput = false;
  Tensor notSkippedTensor; // non-owning reference
  auto should_skip = [](const Tensor& t) {
    return !t.defined() && t.dim() == 1;
  };
  int nDims = 0;

  // Check for type promotion
  TORCH_CHECK(
      canCast(at::native::result_type(inputs), result.scalar_type()),
      "input types ",
      " can't be cast to the desired output type ",
      result.scalar_type());

  for (i = 0; i < numInputs; i++) {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i].get().dim();
    notSkippedTensor = inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (!notSkippedTensor.defined()) {
    return;
  }

  TORCH_CHECK(numInputs > 0, "invalid number of inputs");
  TORCH_CHECK(dimension >= 0, "invalid dimension");

  Tensor first_tensor = inputs[0];
  auto ft_smf = cat_compute_output_memory_format(inputs);

  std::vector<int64_t> size(nDims);

  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    Tensor tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    check_shape_except_dim(notSkippedTensor, tensor, dimension);
    cat_dim_size += tensor.size(dimension);
  }

  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor.size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }

  if (CHANNELSLAST1D_DPCPP == ft_smf) {
    result.resize_(size, at::MemoryFormat::Contiguous);
    result = convert_tensor_to_channels_last_1d(result);
  } else {
    result.resize_(size, ft_smf);
  }

  const bool all32BitIndexable =
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& t) {
        return xpu::dpcpp::detail::canUse32BitIndexMath(t);
      });
  const bool allContiguous =
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& t) {
        return !t.defined() || t.is_contiguous();
      });

  if (inputs.size() > 1 && !hasSkippedInput &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      xpu::dpcpp::detail::canUse32BitIndexMath(result) && allContiguous &&
      all32BitIndexable && allSameType &&
      (inputs[0].get().scalar_type() == result.scalar_type()) &&
      (!inputs[0].get().is_quantized())) {
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        result.scalar_type(),
        "cat_dpcpp",
        [&]() {
          parallel_cat<
              scalar_t,
              scalar_t,
              scalar_t,
              scalar_t,
              std::false_type::value,
              std::false_type::value>(result, inputs, dimension, nDims);
        });
  } else if (
      inputs.size() > 1 && !hasSkippedInput &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      xpu::dpcpp::detail::canUse32BitIndexMath(result) && allContiguous &&
      all32BitIndexable && allSameType && inputs[0].get().is_quantized()) {
    // dispatch code for quantization(kQInt8, kQUInt8) tensor
    IPEX_DISPATCH_QTYPE_WITH_UNDERLYING(
        result.scalar_type(), "cat_dpcpp", 0, [&]() {
          IPEX_DISPATCH_QTYPE_WITH_UNDERLYING(
              inputs[0].get().scalar_type(), "cat_dpcpp", 1, [&]() {
                parallel_cat<
                    scalar_t_0,
                    underlying_t_0,
                    scalar_t_1,
                    underlying_t_1,
                    bool_t_0::value,
                    bool_t_1::value>(result, inputs, dimension, nDims);
              });
        });
  } else {
    offset = 0;
    for (j = 0; j < numInputs; j++) {
      if (should_skip(inputs[j]))
        continue;
      int64_t dimSize = inputs[j].get().size(dimension);
      Tensor nt = at::narrow(result, dimension, offset, dimSize);
      nt.copy_(inputs[j]);
      offset += dimSize;
    }
  }
}

} // namespace impl

double AsignOneDnnQuantizeScale(
    const Tensor& t,
    const double default_scale,
    const int64_t zero_point) {
  if (!t.is_quantized()) {
    return default_scale;
  }
  double res = t.q_scale();
  auto q_ctx = DPCPPTensorContext::get_tensor_ctx(t);
  res = ((q_ctx.is_plain() ? get_onednn_dtype(t) : q_ctx.meta().data_type()) ==
             memory::data_type::u8 &&
         zero_point == 128)
      ? static_cast<float>(res / 2)
      : static_cast<float>(res);
  return res;
}

void cat_(const ITensorListRef& container, int64_t dim, Tensor& out) {
  auto tensors = container.materialize();
  // Inputs cannot alias the output tensor
  size_t i = 0;
  for (const Tensor& t : tensors) {
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
    i++;
  }
  dim = at::legacy_cat_wrap_dim(dim, tensors);

  // Inputs cannot alias the output tensor
  for (const auto i : c10::irange(tensors.size())) {
    auto lap = at::get_overlap_status(out, tensors[i]);
    TORCH_CHECK(
        lap != at::MemOverlapStatus::Partial &&
            lap != at::MemOverlapStatus::Full,
        0,
        "unsupported operation: the input tensors cannot refer to any of the "
        "output memory locations. Found overlap in input tensor ",
        i);
  }
  at::assert_no_internal_overlap(out);

  ScalarType firstType = tensors[0].get().scalar_type();
  bool allSameType =
      std::all_of(tensors.begin(), tensors.end(), [firstType](const Tensor& t) {
        return t.scalar_type() == firstType;
      });

  bool isBlockfmt =
      std::any_of(tensors.begin(), tensors.end(), [](const Tensor& t) {
        return xpu::oneDNN::is_onednn_layout(t);
      });

  bool isQuant =
      std::any_of(tensors.begin(), tensors.end(), [](const Tensor& t) {
        return t.is_quantized();
      });

  // when satify none of the input tensors is block fmt
  // cat will go to DPCPP path, all the other cases will go to oneDNN path
  if (isQuant || !isBlockfmt) {
    auto atens = at::AtenIpexTypeXPU::to_plain_if_needed(tensors);
    impl::cat(
        out,
        ITensorListRef(atens).materialize(),
        atens.size(),
        dim,
        allSameType);
  } else {
    xpu::oneDNN::concat(out, tensors, dim);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
