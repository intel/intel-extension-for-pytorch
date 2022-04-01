#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TypeProperties.h>

#include <core/TensorImplUtils.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"

#include <core/Memory.h>
#include <core/MemoryFormat.h>
#include <intrinsic/ipex_intrinsic.h>
#include <oneDNN/oneDNN.h>
#include <tensor/Context.h>

#include "Cat.h"

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

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

template <typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
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
template <typename T, typename IndexType, int Dims>
void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride,
    int batchCounter) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  // Get grid where x dim fills half gpu and y dim is number of tensors.
  // This will have cating two tensors fill the entire grid, but prevent
  // many threads from needlessly load meta data if their sizes is small.

  auto numCU = dpcppMaxComputeUnitSize(dev_id);
  auto numWI = dpcppMaxWorkGroupSize(dev_id);
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

      if (tid >= nElements)
        return;

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
    cgh.parallel_for(DPCPP::nd_range<2>(global_range, local_range), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf)
}

template <typename scalar_t>
void parallel_cat(
    Tensor& out,
    const TensorList& inputs,
    int64_t dimension,
    int nDims) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t* data = out.data_ptr<scalar_t>();

  // Kernel Parameter
  long tensorMetadataSize =
      sizeof(CatArrInputTensor<scalar_t, unsigned int>) * CAT_ARRAY_BATCH_SIZE;
  auto d_inputs_storage =
      at::empty({tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_t, unsigned int>*>(
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
  for (int i = 0; i < inputs.size(); i += CAT_ARRAY_BATCH_SIZE) {
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      auto stackInputs_storage = at::empty(
          {tensorMetadataSize},
          out.options().dtype(at::kByte).device(at::kCPU));
      auto stackInputs =
          static_cast<CatArrInputTensor<scalar_t, unsigned int>*>(
              stackInputs_storage.data_ptr());
      for (batchCounter = 0; batchCounter < CAT_ARRAY_BATCH_SIZE &&
           (i + batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize = at::native::size(inputs[i + batchCounter], dimension);

        stackInputs[batchCounter].input =
            inputs[i + batchCounter].data_ptr<scalar_t>();
        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements = inputs[i + batchCounter].numel();

        // update offset
        offset += dimSize;
      }
      d_inputs_storage.copy_(stackInputs_storage);
    }

#define HANDLE_CASE(DIMS)                            \
  CatArrayBatchedCopy<scalar_t, unsigned int, DIMS>( \
      data,                                          \
      d_inputs,                                      \
      param,                                         \
      dimension,                                     \
      param.outputStride[dimension],                 \
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

  // Inputs cannot alias the output tensor
  for (int i = 0; i < inputs.size(); i++) {
    auto lap = at::get_overlap_status(result, inputs[i]);
    TORCH_CHECK(
        lap != at::MemOverlapStatus::PARTIAL &&
            lap != at::MemOverlapStatus::FULL,
        "unsupported operation: the input tensors cannot refer to any "
        "of the output memory locations. Found overlap in input "
        "tensor ",
        i);
  }
  at::assert_no_internal_overlap(result);

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

  Tensor first_tensor = inputs[0];
  auto ft_smf = first_tensor.suggest_memory_format();

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
  result.resize_(size, ft_smf);

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
      all32BitIndexable && allSameType) {
    IPEX_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        result.scalar_type(),
        "cat_dpcpp",
        [&]() { parallel_cat<scalar_t>(result, inputs, dimension, nDims); });
  } else {
    offset = 0;
    for (j = 0; j < numInputs; j++) {
      if (should_skip(inputs[j]))
        continue;
      int64_t dimSize = inputs[j].size(dimension);
      Tensor nt = at::narrow(result, dimension, offset, dimSize);
      nt.copy_(inputs[j]);
      offset += dimSize;
    }
  }
}

void dnnl_cat(Tensor& output, TensorList inputs, int numInputs, int dimension) {
  int i, j;
  int64_t offset;
  Tensor notSkippedTensor; // non-owning reference
  auto should_skip = [](const Tensor& t) {
    return !t.defined() && t.dim() == 1;
  };

  int nDims = 0;
  for (i = 0; i < numInputs; i++) {
    if (should_skip(inputs[i])) {
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

  bool first_tensor_is_channels_last = false;
  Tensor first_tensor = inputs[0];
  auto ft_ndim = first_tensor.ndimension();
  auto ft_smf = first_tensor.suggest_memory_format();
  first_tensor_is_channels_last = is_smf_channels_last(first_tensor);

  std::vector<Tensor> cat_tensors;
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    Tensor tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    check_shape_except_dim(notSkippedTensor, tensor, dimension);
    cat_dim_size += tensor.size(dimension);
    auto ndim = tensor.ndimension();
    if (true == first_tensor_is_channels_last) {
      Tensor cl_tensor = tensor.contiguous(get_cl_tag_by_ndim(ndim));
      cat_tensors.push_back(cl_tensor);
    } else {
      // only support ndim == 1, 2, 3, 6 for nchw format
      // if first tensor is plain format, convert all the following tensors to
      // nchw format to algin pytorch behavior
      Tensor nchw_tensor = tensor.contiguous();
      cat_tensors.push_back(nchw_tensor);
    }
  }

  std::vector<int64_t> size(nDims);
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor.size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  memory::dims output_dims = size;
  output.resize_(size, ft_smf);

  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  std::vector<memory::desc> cat_tensors_md;
  std::vector<memory> cat_tensors_mem;
  for (size_t i = 0; i < cat_tensors.size(); i++) {
    std::vector<int64_t> dims;
    for (size_t j = 0; j < cat_tensors[i].dim(); j++) {
      dims.push_back(cat_tensors[i].size(j));
    }

    memory::dims input_tz = dims;
    auto data_t = get_onednn_dtype(cat_tensors[i]);
    auto ndim = cat_tensors[i].ndimension();
    bool is_channels_last = is_smf_channels_last(first_tensor);
    auto format_plain = get_dnnl_default_format(ndim, is_channels_last);

    memory::desc tensor_md;
    if (!Settings::I().is_layout_opt_enabled()) {
      tensor_md = memory::desc({input_tz}, data_t, format_plain);
    } else {
      auto input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(
          cat_tensors[i]);
      tensor_md = input_ctx.is_plain()
          ? memory::desc({input_tz}, data_t, format_plain)
          : input_ctx.meta();
    }
    cat_tensors_md.push_back(tensor_md);

    auto input_usr_memory =
        dpcpp_onednn_memory(tensor_md, engine, cat_tensors[i].data_ptr());
    cat_tensors_mem.push_back(input_usr_memory);
  }

  auto data_t = get_onednn_dtype(cat_tensors[0]);
  auto format_plain =
      get_dnnl_default_format(ft_ndim, first_tensor_is_channels_last);
  auto output_md = memory::desc(output_dims, data_t, format_plain);
  auto concat_pd = concat::primitive_desc(
      output_md, static_cast<int>(dimension), cat_tensors_md, engine);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, output_dims, static_cast<int>(dimension), cat_tensors_md);
#endif

  // Tensor output;
  memory output_usr_memory;
  if (!Settings::I().is_layout_opt_enabled()) {
    output_usr_memory =
        dpcpp_onednn_memory(output_md, engine, output.data_ptr());
  } else {
    auto expected_output_md = concat_pd.dst_desc();
    if (output_md != expected_output_md) {
      // reallocate memory for some blk fmt
      output = at::AtenIpexTypeXPU::empty_opaque_tensor(
          expected_output_md, cat_tensors[0].options(), c10::nullopt);
      output_usr_memory =
          dpcpp_onednn_memory(expected_output_md, engine, output.data_ptr());
    } else {
      output_usr_memory =
          dpcpp_onednn_memory(output_md, engine, output.data_ptr());
    }
  }

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_DST, output_usr_memory},
  };
  for (int i = 0; i < (int)cat_tensors.size(); i++) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, cat_tensors_mem[i]});
  }

  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  auto concat_p = fetch_or_create_m<dnnl::concat>(key, concat_pd);
#else
  auto concat_p = dnnl::concat(concat_pd);
#endif

  DPCPP_ONEDNN_EXEC(concat_p, strm, args);
}

} // namespace impl

Tensor& _cat_out(Tensor& out, TensorList tensors, int64_t dim) {
  bool skip_dnnl_cat = false;
  for (int i = 0; i < tensors.size(); i++) {
    Tensor tensor = tensors[i];
    if (tensor.defined()) {
      if (tensor.scalar_type() == ScalarType::Bool ||
          tensor.scalar_type() == ScalarType::Short ||
          tensor.scalar_type() == ScalarType::Double ||
          tensor.scalar_type() == ScalarType::Long) {
        skip_dnnl_cat = true;
      }
      break;
    }
  }

  ScalarType firstType = tensors[0].scalar_type();
  bool allSameType =
      std::all_of(tensors.begin(), tensors.end(), [firstType](const Tensor& t) {
        return t.scalar_type() == firstType;
      });
  allSameType = allSameType && (out.scalar_type() == firstType);

  // DNNL cat does not support double datatype now.
  if (!allSameType || skip_dnnl_cat) {
    impl::cat(out, tensors, tensors.size(), dim, allSameType);
  } else {
    impl::dnnl_cat(out, tensors, tensors.size(), dim);
  }
  return out;
}

Tensor _cat(TensorList tensors, int64_t dim) {
  auto high_type = at::native::result_type(tensors);
  auto out = at::empty({0}, tensors[0].options().dtype(high_type));
  return at::AtenIpexTypeXPU::_cat_out(out, tensors, dim);
}

} // namespace AtenIpexTypeXPU
} // namespace at
