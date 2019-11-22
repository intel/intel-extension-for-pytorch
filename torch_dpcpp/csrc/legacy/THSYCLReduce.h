#ifndef THSYCL_REDUCE_INC
#define THSYCL_REDUCE_INC

#include <c10/dpcpp/SYCLUtils.h>
#include <THDP/THSYCLGeneral.h>
#include <THDP/THSYCLDeviceUtils.h>
#include <THDP/THSYCLTensorTypeUtils.h>

DP_DEF_K2(reducedim,
          typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          int ADims,
          int BDims);

// Kernel that handles an entire reduction of a slice of a tensor
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          int ADims, int BDims>
void kernelReduceDim(TensorInfo<T, IndexType> out,
                     TensorInfo<T, IndexType> in,
                     IndexType reductionStride,
                     IndexType reductionSize,
                     int64_t outElements,
                     AccT init,
                     ModifyOp modifyOp,
                     ReduceOp reduceOp,
                     FinalizeOp finalizeOp) {
  auto queue = c10::sycl::syclGetCurrentQueue();
  int64_t group_size =  c10::sycl::syclMaxWorkGroupSize(queue);

  // TODO: how to handle the huge outElements larger than work item limitation?
  auto num_groups = THSYCLCeilDiv(outElements, group_size);
  auto total_items = num_groups * group_size;

  // command group function
  auto cgf = DP_Q_CGF(cgh) {
    auto out_acc = c10::sycl::SYCLAccessor<dp_w_mode, AccT>(cgh, out.data).get_access();
    auto in_acc = c10::sycl::SYCLAccessor<dp_r_mode, T>(cgh, in.data).get_access();

    auto kfn = DP_Q_KFN(DP::nd_item<1> item) {
      int64_t global_id = item.get_global_linear_id();
      if (global_id >= outElements) { return; }

      dp_global_ptr_pt<T> out_ptr = GET_ACC_PTR(out_acc, T);
      dp_global_ptr_cpt<T> in_ptr = GET_ACC_PTR(in_acc, T);

      IndexType out_off = IndexToOffset<T, IndexType, ADims>::get(global_id, out);
      IndexType in_off = IndexToOffset<T, IndexType, BDims>::get(global_id, in);

      AccT r = init;
      for (IndexType i = 0; i < reductionSize; i++) {
        const AccT val = scalar_cast<AccT>(in_ptr[in_off]);
        r = reduceOp(r, modifyOp(val));
        in_off += reductionStride;
      }

      out_ptr[out_off] = scalar_cast<T>(finalizeOp(r));
    };

    // kick off kernel
    cgh.parallel_for<DP_K(
        reducedim, T, IndexType, AccT, ModifyOp, ReduceOp, FinalizeOp, ADims, BDims)>(
          DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  // submit to SYCL queue
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
template <typename ScalarType,
          typename TensorType,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          typename AccT>
bool THSYCL_reduceDim(THSYCLState* state,
                      TensorType* out,
                      TensorType* in,
                      const ModifyOp modifyOp,
                      const ReduceOp reduceOp,
                      const FinalizeOp finalizeOp,
                      AccT init,
                      int dim,
                      int keepdim) {
  auto inElements = THSYCLTensor_nElement(state, in);

  auto inDims = THSYCLTensor_nDimensionLegacyNoScalars(state, in);
  auto outDims = THSYCLTensor_nDimensionLegacyNoScalars(state, out);
  if (inDims == 0) { return true; /* Zero-dim tensor; do nothing */ }
  if (inDims > MAX_SYCLTORCH_DIMS || outDims > MAX_SYCLTORCH_DIMS) { return false; }

  auto reductionSize = THSYCLTensor_sizeLegacyNoScalars(state, in, dim);
  auto reductionStride = THSYCLTensor_strideLegacyNoScalars(state, in, dim);
  auto outElements = inElements / reductionSize;

  // Preserve noncontiguities by unsqueezing out if necessary
  THSYCLTensor_preserveReduceDimSemantics(
      state, out, THSYCLTensor_nDimensionLegacyAll(state, in), dim, keepdim);

  // Resize out to correspond to the reduced size with keepdim=True.
  auto sizes = THTensor_sizesLegacyNoScalars(in);
  sizes[dim] = 1;
  THSYCLTensor_resize(state, out, sizes, {});

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, OUT, IN)                                                          \
  kernelReduceDim<ScalarType, TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp, OUT, IN>(         \
      outInfo, inInfo, reductionStride, reductionSize, outElements, init,                   \
      modifyOp, reduceOp, finalizeOp);

#define HANDLE_IN_CASE(TYPE, OUT, IN)                     \
  {                                                       \
    switch (IN) {                                         \
      case 1:                                             \
        HANDLE_CASE(TYPE, OUT, 1);                        \
        break;                                            \
      case 2:                                             \
        HANDLE_CASE(TYPE, OUT, 2);                        \
        break;                                            \
      default:                                            \
        HANDLE_CASE(TYPE, OUT, -1);                       \
        break;                                            \
    }                                                     \
  }

#define HANDLE_OUT_CASE(TYPE, OUT, IN)                    \
  {                                                       \
    switch (OUT) {                                        \
      case 1:                                             \
        HANDLE_IN_CASE(TYPE, 1, IN);                      \
        break;                                            \
      case 2:                                             \
        HANDLE_IN_CASE(TYPE, 2, IN);                      \
        break;                                            \
      default:                                            \
        HANDLE_IN_CASE(TYPE, -1, IN);                     \
        break;                                            \
    }                                                     \
  }

  if(THSYCLTensor_canUse32BitIndexMath(state, out) &&
     THSYCLTensor_canUse32BitIndexMath(state, in)) {
    TensorInfo<ScalarType, unsigned int> outInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, out);
    outInfo.collapseDims();

    TensorInfo<ScalarType, unsigned int> inInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();
    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  } else {
    TensorInfo<ScalarType, uint64_t> outInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, out);
    outInfo.collapseDims();

    TensorInfo<ScalarType, uint64_t> inInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (outInfo.dims == 1 && inInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1);
    }
  }

#undef HANDLE_CASE
#undef HANDLE_IN_CASE
#undef HANDLE_OUT_CASE

  if (!keepdim) { THSYCLTensor_squeeze1d(state, out, out, dim); }
  return true;
}

#endif // THSYCL_REDUCE_INC
