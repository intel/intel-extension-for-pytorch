#include <core/Context.h>
#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  inline void operator()(T& v) const {
    v = val;
  }

  const T val;
};

// calculate shift where we should start processing on current item
template <
    typename _NDItemId,
    typename _GlobalIdx,
    typename _SizeNIter,
    typename _SizeN>
_SizeN calc_shift(
    const _NDItemId __item_id,
    const _GlobalIdx __global_idx,
    _SizeNIter& __n_iter,
    const _SizeN __n) {
  auto __global_range_size = __item_id.get_global_range().size();

  auto __start = __n_iter * __global_idx;
  auto __global_shift = __global_idx + __n_iter * __global_range_size;
  if (__n_iter > 0 && __global_shift > __n) {
    __start += __n % __global_range_size - __global_idx;
  } else if (__global_shift < __n) {
    __n_iter++;
  }
  return __start;
}

// write data from local memory to global
template <
    typename _Inclusive,
    typename _NDItemId,
    typename _GlobalIdx,
    typename _Size,
    typename _AccLocal,
    typename _InAcc,
    typename _OutAcc,
    typename _Tp,
    typename _Fp,
    typename _BinaryOp,
    typename _UnaryOp>
void write_to_global(
    const _NDItemId __item_id,
    const _GlobalIdx __global_idx,
    const _Size __n,
    const _AccLocal& __local_mem,
    const _InAcc& __input,
    const _OutAcc& __result,
    _Tp __init,
    _Fp __f,
    _BinaryOp __bin_op,
    _UnaryOp __unary_op) {
  auto __local_idx = __item_id.get_local_id(0);
  auto __global_range_size = __item_id.get_global_range().size();
  auto __n_iter = __n / __global_range_size;
  auto __start = calc_shift(__item_id, __global_idx, __n_iter, __n);
  auto __shifted_global_idx = __global_idx + __start;

  _Tp __shift_for_true = __init;
  if (__local_idx != 0)
    __shift_for_true = __local_mem[__local_idx - 1];
  _Tp __shift_for_false = __shifted_global_idx - __shift_for_true;

  // inclusive scan branch
  if (_Inclusive()) {
    for (decltype(__n_iter) __i = 0; __i < __n_iter; ++__i) {
      auto __unary_op__result = __unary_op(__shifted_global_idx + __i, __input);
      __shift_for_true = __bin_op(__shift_for_true, __unary_op__result);
      __shift_for_false = __bin_op(__shift_for_false, 1 - __unary_op__result);

      __f(__shift_for_true,
          __shift_for_false,
          __shifted_global_idx + __i,
          __input,
          __result);
    }
  }
  // exclusive scan branch
  else {
    for (decltype(__n_iter) __i = 0; __i < __n_iter; ++__i) {
      __f(__shift_for_true,
          __shift_for_false,
          __shifted_global_idx + __i,
          __input,
          __result);

      auto __unary_op_result = __unary_op(__shifted_global_idx + __i, __input);
      __shift_for_true = __bin_op(__shift_for_true, __unary_op_result);
      __shift_for_false = __bin_op(__shift_for_false, 1 - __unary_op_result);
    }
  }
}

// Scan on local memory
template <
    typename _Inclusive,
    typename _BinaryOperation,
    typename _UnaryOp,
    typename _Assigner,
    typename _Tp>
struct scan {
  _BinaryOperation __bin_op;
  _UnaryOp __unary_op;
  _Assigner __f;

  template <
      typename _NDItemId,
      typename _GlobalIdx,
      typename _Size,
      typename _AccLocal,
      typename _InAcc,
      typename _OutAcc>
  void operator()(
      const _NDItemId __item_id,
      const _GlobalIdx __global_idx,
      const _Size __n,
      const _AccLocal& __local_mem,
      const _InAcc& __input,
      const _OutAcc& __result,
      _Tp __init) const {
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_size = __item_id.get_local_range().size();
    auto __old_init = __init;
    if (__local_idx == 0) {
      __local_mem[0] = __bin_op(__init, __local_mem[0]);
    }
    // 1. reduce
    decltype(__group_size) __k = 1;
    do {
      __item_id.barrier(DPCPP::access::fence_space::local_space);
      if (__local_idx % (2 * __k) == 0 && __local_idx + __k < __group_size &&
          __global_idx < __n && __global_idx + __k < __n) {
        __local_mem[__local_idx + 2 * __k - 1] = __bin_op(
            __local_mem[__local_idx + __k - 1],
            __local_mem[__local_idx + 2 * __k - 1]);
      }
      __k *= 2;
    } while (__k < __group_size);
    __item_id.barrier(DPCPP::access::fence_space::local_space);

    // 2. scan
    auto __partial_sums = __local_mem[__local_idx];
    __k = 2;
    do {
      auto __shifted_local_idx = __local_idx - __local_idx % __k - 1;
      if (__shifted_local_idx >= 0 && __local_idx % (2 * __k) >= __k &&
          __local_idx % (2 * __k) < 2 * __k - 1 && __global_idx < __n) {
        __partial_sums =
            __bin_op(__local_mem[__shifted_local_idx], __partial_sums);
      }
      __k *= 2;
    } while (__k < __group_size);
    __item_id.barrier(DPCPP::access::fence_space::local_space);
    __local_mem[__local_idx] = __partial_sums;
    __item_id.barrier(DPCPP::access::fence_space::local_space);

    // 4. Write result to global memory
    write_to_global<_Inclusive>(
        __item_id,
        __global_idx,
        __n,
        __local_mem,
        __input,
        __result,
        __old_init,
        __f,
        __bin_op,
        __unary_op);
  }
};

template <typename _BinaryOperation1, typename _Tp>
struct reduce {
  _BinaryOperation1 __bin_op1;

  template <
      typename _NDItemId,
      typename _GlobalIdx,
      typename _Size,
      typename _AccLocal>
  _Tp operator()(
      const _NDItemId __item_id,
      const _GlobalIdx __global_idx,
      const _Size __n,
      const _AccLocal& __local_mem) const {
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_size = __item_id.get_local_range().size();

    decltype(__group_size) __k = 1;
    do {
      __item_id.barrier(DPCPP::access::fence_space::local_space);
      if (__local_idx % (2 * __k) == 0 && __local_idx + __k < __group_size &&
          __global_idx < __n && __global_idx + __k < __n) {
        __local_mem[__local_idx] =
            __bin_op1(__local_mem[__local_idx], __local_mem[__local_idx + __k]);
      }
      __k *= 2;
    } while (__k < __group_size);
    return __local_mem[__local_idx];
  }
};

template <typename _Operation1, typename _Operation2>
struct transform_init {
  _Operation1 __binary_op;
  _Operation2 __unary_op;

  template <
      typename _NDItemId,
      typename _Acc,
      typename _GlobalIdx,
      typename _Size,
      typename _AccLocal>
  void operator()(
      const _NDItemId __item_id,
      const _GlobalIdx __global_idx,
      const _Acc& __acc,
      _Size __n,
      const _AccLocal& __local_mem) const {
    auto __local_idx = __item_id.get_local_id(0);
    auto __global_range_size = __item_id.get_global_range().size();
    auto __n_iter = __n / __global_range_size;
    auto __start = calc_shift(__item_id, __global_idx, __n_iter, __n);
    auto __shifted_global_idx = __global_idx + __start;

    typename _AccLocal::value_type res;
    if (__global_idx < __n) {
      res = __unary_op(__shifted_global_idx, __acc);
    }
    // Add neighbour to the current __local_mem
    for (decltype(__n_iter) __i = 1; __i < __n_iter; ++__i) {
      res = __binary_op(res, __unary_op(__shifted_global_idx + __i, __acc));
    }
    if (__global_idx < __n) {
      __local_mem[__local_idx] = res;
    }
  }
};

// get mask without predicate application
template <typename _Tp, const int N>
struct get_mask {
  template <typename _Idx, typename _Input>
  _Tp operator()(const _Idx __idx, const _Input& __input) const {
    return _Tp(__input[__idx]);
  }
};

template <typename... T>
class __init_kernel_name_1 {};
template <typename... T>
class __init_kernel_name_2 {};
// returns the last partial sum
template <
    typename InputType,
    typename OutputType,
    typename IndexType,
    typename _BinaryOperation,
    typename _Transform,
    typename _Reduce,
    typename _Scan>
IndexType parallel_transform_scan(
    DPCPP::queue& queue,
    TensorInfo<InputType, IndexType>& input,
    TensorInfo<OutputType, IndexType>& output,
    IndexType num_elements,
    _BinaryOperation __binary_op,
    IndexType __init,
    _Transform __brick_transform,
    _Reduce __brick_reduce,
    _Scan __brick_scan) {
  auto __target_buffer = input.data;
  auto __result_buffer = output.data;
  auto __wgroup_size = dpcppMaxWorkGroupSize(queue);
  auto __mcu = dpcppMaxComputeUnitSize(queue);
  auto __n = num_elements;

  auto __n_groups = (__n - 1) / __wgroup_size + 1;
  __n_groups = std::min(decltype(__n_groups)(__mcu), __n_groups);
  // TODO: try to change __n_groups with another formula for more perfect load
  // balancing
  // TODO: try to replace with int8_t
  using _AtomicType = int32_t;
  // 0. Create temporary global buffer to store temporary value
  auto& allocator = *at::dpcpp::getDPCPPDeviceAllocator();
  auto __local_sums = allocator.allocate(
      sizeof(IndexType) * __n_groups); // temporary storage for global atomic
  auto __ready_flags = allocator.allocate(
      sizeof(_AtomicType) * __n_groups); // temporary storage for global atomic

  // 1. Initialize temp buffer
  auto cgf_1 = DPCPP_Q_CGF(cgh) {
    auto __ready_flags_acc =
        DPCPPAccessor<dpcpp_rw_mode>(cgh, __ready_flags.get());

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto __ready_flags_ptr =
          __ready_flags_acc.template get_pointer<_AtomicType>();
      int64_t gid = item_id.get_linear_id();

      __ready_flags_ptr[gid] = 0;
    };

    cgh.parallel_for<__init_kernel_name_1<InputType, OutputType, IndexType>>(
        DPCPP::range</*dim=*/1>(__n_groups), kfn);
  };

  // launch kernel
  DPCPP_Q_SYNC_SUBMIT(queue, cgf_1);

  uint32_t __for_dynamic_id = 0;
  auto __dynamic_id_buf = DPCPP::buffer<uint32_t, /*dim=*/1>(
      &__for_dynamic_id, 1); // temporary storage for group_id atomic
  // Main parallel_for
  auto cgf_2 = DPCPP_Q_CGF(cgh) {
    auto __acc = DPCPPAccessor<dpcpp_r_mode>(cgh, __target_buffer);
    auto __dynamic_id_acc =
        __dynamic_id_buf.template get_access<dpcpp_rw_mode>(cgh);
    auto __local_sums_acc =
        DPCPPAccessor<dpcpp_rw_mode>(cgh, __local_sums.get());
    auto __result_acc = DPCPPAccessor<dpcpp_w_mode>(cgh, __result_buffer);
    auto __ready_flags_acc =
        DPCPPAccessor<dpcpp_rw_mode>(cgh, __ready_flags.get());

    // create local accessors
    DPCPP::accessor<uint32_t, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        __group_id_local(1, cgh);
    DPCPP::accessor<IndexType, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        __transform_local(__wgroup_size, cgh);
    DPCPP::accessor<IndexType, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        __reduce_local_mem(__wgroup_size, cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item</*dim=*/1> item_id) {
      auto __local_idx = item_id.get_local_id(0);
      auto __group_size = item_id.get_local_range().size();
      auto __acc_ptr = __acc.template get_pointer<InputType>();
      auto __result_ptr = __result_acc.template get_pointer<OutputType>();
      auto __dynamic_id_ptr = GET_ACC_PTR(__dynamic_id_acc, uint32_t);
      auto __ready_flags_ptr =
          __ready_flags_acc.template get_pointer<_AtomicType>();
      auto __local_sums_ptr =
          __local_sums_acc.template get_pointer<IndexType>();
      dpcpp_multi_ptr<_AtomicType, dpcpp_global_space> __ready_flags_atomic_ptr(
          __ready_flags_ptr);
      dpcpp_multi_ptr<uint32_t, dpcpp_global_space> __dynamic_id_atomic_ptr(
          __dynamic_id_ptr);

      // dynamic group_id
      if (__local_idx == 0) {
        // add 1 to __dynamic_id_acc atomically
        __group_id_local[0] = DPCPP::atomic<uint32_t>(__dynamic_id_atomic_ptr)
                                  .fetch_add(uint32_t(1));
      }
      item_id.barrier(DPCPP::access::fence_space::local_space);
      auto __group_id = __group_id_local[0];
      auto __global_idx = (__group_id * __group_size) + __local_idx;

      // 2. Initialization (transform part). Fill local memory
      __brick_transform(
          item_id, __global_idx, __acc_ptr, __n, __transform_local);

      // copy to another memory to save the state
      __reduce_local_mem[__local_idx] = __transform_local[__local_idx];
      item_id.barrier(DPCPP::access::fence_space::local_space);

      // TODO: think about the model Scan-Add. It will help us to get rid of 2
      // reduce calls
      // and __reduce_local_mem won't be needed
      // 3. local reduce
      auto __local_reduce =
          __brick_reduce(item_id, __global_idx, __n, __reduce_local_mem);
      if (__group_id == 0 && __local_idx == 0) {
        // the next 2 lines might be swapped
        __local_sums_ptr[0] = __binary_op(__init, __local_reduce);
        DPCPP::atomic<_AtomicType>(__ready_flags_atomic_ptr).store(1);
      }
      item_id.barrier(DPCPP::access::fence_space::local_space);

      // 4. get reduced value from the previous work group
      IndexType __new_init = __init;
      if (__group_id != 0 && __local_idx == 0) {
        _AtomicType __temp;
        // wait for updating atomic from the previous work group
        while ((__temp = DPCPP::atomic<_AtomicType>(
                             __ready_flags_atomic_ptr + __group_id - 1)
                             .load()) == 0) {
        }
        auto __new_res =
            __binary_op(__local_sums_ptr[__group_id - 1], __local_reduce);
        // the next 2 lines might be swapped
        __local_sums_ptr[__group_id] = __new_res;
        DPCPP::atomic<_AtomicType>(__ready_flags_atomic_ptr + __group_id)
            .store(1);
        __new_init = __local_sums_ptr[__group_id - 1];
      }
      item_id.barrier(DPCPP::access::fence_space::local_space);

      // 5. local scan and putting down to __result
      __brick_scan(
          item_id,
          __global_idx,
          __n,
          __transform_local,
          __acc_ptr,
          __result_ptr,
          __new_init);
    };

    cgh.parallel_for<__init_kernel_name_2<InputType, OutputType, IndexType>>(
        DPCPP::nd_range</*dim=*/1>(
            DPCPP::range</*dim=*/1>(__n_groups * __wgroup_size),
            DPCPP::range</*dim=*/1>(__wgroup_size)),
        kfn);
  };

  // launch kernel
  DPCPP_Q_SYNC_SUBMIT(queue, cgf_2);

  auto sb =
      dpcppGetBufferMap().template get_buffer<IndexType>(__local_sums.get());
  auto host_acc = sb.template get_access<dpcpp_r_mode>();
  auto __last_reduced_value = host_acc[__n_groups - 1];
  return __last_reduced_value;
}

template <
    typename InputType,
    typename OutputType,
    typename IndexType,
    typename _CreateMaskOp,
    typename _CopyByMaskOp>
IndexType pattern_scan_copy(
    DPCPP::queue& queue,
    TensorInfo<InputType, IndexType>& input,
    TensorInfo<OutputType, IndexType>& output,
    IndexType num_elements,
    _CreateMaskOp __create_mask_op,
    _CopyByMaskOp __copy_by_mask_op) {
  using _ReduceOp = std::plus<IndexType>;
  using _GetMaskOp = get_mask<IndexType, 1>;

  auto __reduce_op = _ReduceOp{};
  auto __get_mask_op = _GetMaskOp{};

  return parallel_transform_scan(
      queue,
      input,
      output,
      num_elements,
      __reduce_op,
      IndexType{0},
      transform_init<_ReduceOp, _CreateMaskOp>{__reduce_op, __create_mask_op},
      reduce<_ReduceOp, IndexType>{__reduce_op},
      scan<
          /*inclusive*/ std::true_type,
          _ReduceOp,
          _GetMaskOp,
          _CopyByMaskOp,
          IndexType>{__reduce_op, __get_mask_op, __copy_by_mask_op});
}

// create mask
template <typename _Pred, typename _Tp>
struct create_mask {
  _Pred __pred;

  template <typename _Idx, typename _Input>
  _Tp operator()(const _Idx __idx, const _Input& __input) const {
    using std::get;
    // 1. apply __pred
    auto __temp = __pred(__input[__idx]);
    // 2. initialize mask
    __input[__idx] = __temp;
    return _Tp(__temp);
  }
};

// copy values by mask to ouput with scanned shift
template <typename IndexType>
struct idx_functor {
  int dims;
  IndexType sz[MAX_DPCPPTORCH_DIMS];
  IndexType st[MAX_DPCPPTORCH_DIMS];

  template <typename T>
  idx_functor(const TensorInfo<T, IndexType>& t_info)
      : idx_functor(t_info, std::make_index_sequence<MAX_DPCPPTORCH_DIMS>{}) {}

  template <typename _Value, typename _Idx, typename _InAcc, typename _OutAcc>
  void operator()(
      const _Value& __out_shift,
      const _Value&,
      const _Idx __global_idx,
      const _InAcc& __input,
      const _OutAcc& __output) const {
    if (__input[__global_idx]) {
      auto linear_idx = __global_idx;
      auto base_out_ptr = __output + (__out_shift + (-1)) * dims;
      for (int i = 0; i < dims; i++) {
        base_out_ptr[i] = linear_idx / st[i];
        linear_idx = linear_idx % st[i];
      }
    }
  }

 private:
  template <typename T, std::size_t... I>
  idx_functor(const TensorInfo<T, IndexType>& t_info, std::index_sequence<I...>)
      : dims(t_info.dims), sz{t_info.sizes[I]...}, st{t_info.strides[I]...} {}
};

template <
    typename InputType,
    typename OutputType,
    typename IndexType,
    typename Predicate,
    typename CopyMaskOp>
IndexType pattern_scan(
    DPCPP::queue& queue,
    TensorInfo<InputType, IndexType>& input,
    TensorInfo<OutputType, IndexType>& output,
    IndexType num_elements,
    Predicate __pred,
    CopyMaskOp __copy_by_mask_op) {
  auto __create_mask_op = create_mask<Predicate, IndexType>{__pred};

  auto __result = pattern_scan_copy(
      queue, input, output, num_elements, __create_mask_op, __copy_by_mask_op);

  return __result;
}
