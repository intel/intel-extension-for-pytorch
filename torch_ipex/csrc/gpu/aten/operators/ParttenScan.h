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
    if (__input[__idx])
      return _Tp(1);
    else
      return _Tp(0);
  }
};

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
//    __input[__idx] = __temp;
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
