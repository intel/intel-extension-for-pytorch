#ifndef THSYCL_REDUCEALL_INC
#define THSYCL_REDUCEALL_INC

#include <core/SYCLUtils.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLDeviceUtils.h>
#include <legacy/THSYCLTensorTypeUtils.h>

DP_DEF_K2(reduceall,
          typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          bool IsTwoPass,
          int ADims);



// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
DP_HOST static inline bool isTwoPassReductionSize(ptrdiff_t elements) {
  return (elements > (c10::sycl::syclMaxWorkGroupSize() * 2));
}

// reduce all elements in local buffer with local barrier
template <typename T,
        typename LocID,
        typename LocPtr,
        typename ReduceOp,
        typename Item>
DP_DEVICE static inline void reduceWorkgroup(T locTotalElements,
                                             LocID locID,
                                             LocPtr locPtr,
                                             ReduceOp reduceOp,
                                             Item item) {
  auto local_total_size = locTotalElements;
  while (local_total_size > 1) {
    auto local_reduce_size = THSYCLCeilDiv(local_total_size, static_cast<T>(2));
    auto reduceId = locID + local_reduce_size;

    item.barrier(dp_local_fence);
    if (reduceId < local_total_size) {
      locPtr[locID] = reduceOp(locPtr[locID], locPtr[reduceId]);
    }
    local_total_size = local_reduce_size;
  }
}

template <typename T,
        typename IndexType,
        typename AccT,
        typename ModifyOp,
        typename ReduceOp,
        bool IsTwoPass,
        int ADims>
void kernelReduceAll(THSYCLState* state,
                     TensorInfo<T, IndexType> in,
                     int64_t totalElements,
                     AccT init,
                     const ModifyOp modifyOp,
                     const ReduceOp reduceOp,
                     void* devOut) {
  auto queue = c10::sycl::syclGetCurrentQueue();
  int64_t group_size =  c10::sycl::syclMaxWorkGroupSize(queue);

  // the range of work-items should be [1, group_size]
  auto num_groups = IsTwoPass ? THSYCLMin(THSYCLCeilDiv(
          THSYCLCeilDiv(totalElements, group_size), 2L), group_size) : 1;
  auto total_items = num_groups * group_size;

  // write through to devOut directly if not two pass reduction
  auto buffer_ptr = devOut;
  at::DataPtr buffer;
  if (IsTwoPass) {
    buffer = state->syclDeviceAllocator->allocate(num_groups * sizeof(AccT));
    buffer_ptr = buffer.get();
  }

  using GroupIDType = uint32_t; //the atomic op only supports 32 bits
  GroupIDType ready = 0;
  auto ready_buf = cl::sycl::buffer<GroupIDType, /*dim=*/1>(&ready, 1);

  // command group function
  auto cgf = DP_Q_CGF(cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<dp_r_mode, T>(cgh, in.data).get_access();
    auto buf_acc = c10::sycl::SYCLAccessor<dp_rw_mode, AccT>(cgh, buffer_ptr).get_access();
    auto out_acc = c10::sycl::SYCLAccessor<dp_w_mode, AccT>(cgh, devOut).get_access();
    auto loc_acc = dp_local_acc_t<AccT, dp_rw_mode>(group_size, cgh);
    auto twop_flag_acc = dp_local_acc_t<bool, dp_rw_mode>(1, cgh);
    auto ready_acc = ready_buf.template get_access<dp_rw_mode>(cgh);

    // kernel function per work-item
    auto kfn = DP_Q_KFN(DP::nd_item<1> item) {
      int64_t id = item.get_local_linear_id();
      int64_t group_id = item.get_group_linear_id();
      int64_t global_id = item.get_global_linear_id();

      dp_global_ptr_cpt<T> in_ptr = GET_ACC_PTR(in_acc, T);
      dp_global_ptr_pt<AccT> buf_ptr = GET_ACC_PTR(buf_acc, AccT);
      dp_local_ptr_pt<AccT> loc_ptr = GET_ACC_PTR(loc_acc, AccT);
      dp_local_ptr_pt<bool> last_wg_done = GET_ACC_PTR(twop_flag_acc, bool);
      auto ready_ptr = GET_ACC_PTR(ready_acc, GroupIDType);
      dp_multi_ptr<uint32_t, dp_global_space> ready_atomic_ptr(ready_ptr);

      // reduce elements per total_items-wise stride
      AccT result = init;
      for (int64_t i = global_id; i < totalElements; i += total_items) {
        auto next_offset = IndexToOffset<T, IndexType, ADims>::get(i, in);
        result = reduceOp(result, modifyOp(in_ptr[next_offset]));
      }
      loc_ptr[id] = result;
      // Reduce the number of the group size elements in local buffer
      reduceWorkgroup<IndexType, IndexType, dp_local_ptr_pt<AccT>, ReduceOp, DP::nd_item<1>>(
              THSYCLMin(totalElements, group_size), id, loc_ptr, reduceOp, item);
      if (id == 0) {
        buf_ptr[group_id] = loc_ptr[0];
      }

      // reduce all temp results in one workgroup
      if (IsTwoPass) {
        if (id == 0) {
          // workitem0 add 1 to ready
          auto finished = DP::atomic<GroupIDType>(ready_atomic_ptr).fetch_add(GroupIDType(1));

          last_wg_done[0] = (finished == (num_groups - 1)) ? true : false;
        }

        item.barrier(dp_local_fence);

        // the last work group keep working
        if (!last_wg_done[0]) { return; }

        // load temp results to local buffer
        if (id < num_groups) { loc_ptr[id] = buf_ptr[id];}

        reduceWorkgroup<IndexType, IndexType, dp_local_ptr_pt<AccT>, ReduceOp, DP::nd_item<1>>(
                num_groups, id, loc_ptr, reduceOp, item);
      }

      if (id == 0) {
        dp_global_ptr_pt<AccT> out_ptr = GET_ACC_PTR(out_acc, AccT);
        out_ptr[0] = loc_ptr[0];
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(reduceall, T, IndexType, AccT, ModifyOp, ReduceOp, IsTwoPass, ADims)>(
            DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  // submit to SYCL queue
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename ScalarType,
        typename TensorType,
        typename ModifyOp,
        typename ReduceOp,
        typename AccT>
bool THSYCL_reduceAll(THSYCLState* state,
                      TensorType* in,
                      const ModifyOp& modifyOp,
                      const ReduceOp& reduceOp,
                      AccT init,
                      AccT* out,
                      int outOnDevice) {
  auto inElements = THSYCLTensor_nElement(state, in);
  auto isTwoPass = isTwoPassReductionSize(inElements);

  auto dims = THSYCLTensor_nDimensionLegacyNoScalars(state, in);
  if (dims > MAX_SYCLTORCH_DIMS) { return false; }

  if (dims == 0) {
    if (!outOnDevice) {
      *out = init;
    } else {
      c10::sycl::syclMemcpy(out, &init, sizeof(AccT), c10::sycl::HostToDevice);
    }
    return true;
  }

  AccT *devOut = out;
  at::DataPtr devOutData;
  if (!outOnDevice) {
    devOutData = state->syclDeviceAllocator->allocate(sizeof(AccT));
    devOut = static_cast<AccT*>(devOutData.get());
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, TwoPass, IN)                                            \
  kernelReduceAll<ScalarType, TYPE, AccT, ModifyOp, ReduceOp, TwoPass, IN>(       \
                  state, inInfo, inElements, init, modifyOp, reduceOp, devOut);

#define HANDLE_PASS_CASE(TYPE, IN)                        \
  if (isTwoPass) {                                        \
    HANDLE_CASE(TYPE, true, IN);                          \
  } else {                                                \
    HANDLE_CASE(TYPE, false, IN);                         \
  }

#define HANDLE_IN_CASE(TYPE, IN)                          \
  {                                                       \
    switch (IN) {                                         \
      case 1:                                             \
        HANDLE_PASS_CASE(TYPE, 1);                        \
        break;                                            \
      case 2:                                             \
        HANDLE_PASS_CASE(TYPE, 2);                        \
        break;                                            \
      default:                                            \
        HANDLE_PASS_CASE(TYPE, -1);                       \
        break;                                            \
    }                                                     \
  }

  if (THSYCLTensor_canUse32BitIndexMath(state, in)) {
    TensorInfo<ScalarType, unsigned int> inInfo =
            getTensorInfo<ScalarType, TensorType, unsigned int>(state, in);
    inInfo.collapseDims();

    HANDLE_IN_CASE(unsigned int, inInfo.dims);
  } else {
    TensorInfo<ScalarType, uint64_t> inInfo =
            getTensorInfo<ScalarType, TensorType, uint64_t>(state, in);
    inInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (inInfo.dims == 1) {
      HANDLE_IN_CASE(uint64_t, 1);
    } else {
      HANDLE_IN_CASE(uint64_t, -1);
    }
  }

#undef HANDLE_CASE
#undef HANDLE_PASS_CASE
#undef HANDLE_IN_CASE

  if (!outOnDevice) {
    c10::sycl::syclMemcpy(out, devOut, sizeof(AccT), c10::sycl::DeviceToHost);
  }

  return true;
}

#endif // THSYCL_REDUCEALL_INC
