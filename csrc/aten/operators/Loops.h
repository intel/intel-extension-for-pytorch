#pragma once
#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <core/detail/OffsetCalculator.h>
#include <core/detail/TensorInfo.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include "MemoryAccess.h"

#define UNROLLED_ELEM_PER_WORK_ITEM 4

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

#define MAX_INPUT_TENSOR_NUM 3
#define MAX_TOTAL_TENSOR_NUM 4
// DPCPP suggest: it’s possible (and even desirable) to oversubscribe tasks to
// device;
constexpr int OVER_SUBSCRIBE_DSS_FACTOR = 16;

// Work around for passing the offsets to the dpcpp kernel instead of using
// OffsetCalculator.
// Need to change it back to OffsetCalculator with dpcpp
template <typename IndexType = uint32_t>
struct SyclOffsetCal {
  int dims;
  // Make the information to be basic data types to avoid compiler issue.
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];

  IndexType get(IndexType linear_idx) const {
    IndexType offset = 0;

    for (int dim = 0; dim < dims; ++dim) {
      // Make the code as naive as possible.
      offset += (linear_idx % sizes[dim]) * strides[dim];
      linear_idx = linear_idx / sizes[dim];
    }
    return offset;
  }
};

template <typename IndexType>
static SyclOffsetCal<IndexType> make_offset_calculator(
    const TensorIterator& iter,
    int n) {
  SyclOffsetCal<IndexType> offset;
  if (n < iter.ntensors()) {
    auto dims = iter.ndim();
    offset.dims = dims;
    auto strides = iter.strides(n);
    auto sizes = iter.shape();
    for (int i = 0; i < dims; i++) {
      offset.sizes[i] = sizes[i];
      offset.strides[i] = strides[i];
    }
  } else {
    offset.dims = 0;
  }
  return offset;
}

template <int N>
static OffsetCalculator<N> make_offset_calculator(const TensorIterator& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
}

template <int N>
static OffsetCalculator<N> make_input_offset_calculator(
    const TensorIterator& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(
    const TensorIterator& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <typename traits, typename ptr_t, std::size_t... INDEX>
typename traits::ArgsTuple dereference_impl(
    ptr_t data[],
    const int64_t* strides,
    int64_t i,
    std::index_sequence<INDEX...>) {
  return std::make_tuple(*(typename traits::template arg<
                           INDEX>::type*)(data[INDEX] + i * strides[INDEX])...);
}

template <typename traits, typename ptr_t>
typename traits::ArgsTuple dereference(
    ptr_t data[],
    const int64_t* strides,
    int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

template <int vec_size, typename T>
struct vectorized_args_tuple {};

template <int vec_size, typename... Args>
struct vectorized_args_tuple<vec_size, std::tuple<Args...>> {
  using vec_args_t = std::tuple<sycl::vec<
      typename native::Memory::aligned_element<sizeof(Args)>::element_type,
      vec_size>...>;
};

template <int vec_size, typename T>
struct vectorized_return_type {
  using vec_ret_t = sycl::vec<
      typename native::Memory::aligned_element<sizeof(T)>::element_type,
      vec_size>;
};

template <
    int unroll_index,
    typename Result,
    class F,
    class TupleVector,
    typename policy_t,
    std::size_t... I>
constexpr void apply_fun_impl(
    Result&& results,
    F& f,
    TupleVector& t,
    const policy_t& policy,
    std::index_sequence<I...>) {
  using traits = function_traits<F>;
  using result_t = std::decay_t<decltype(results[unroll_index])>;
  if (policy.check_inbounds(unroll_index)) {
    auto ret = std::__invoke(
        std::forward<F>(f),
        at::native::Memory::detail::bitwise_cast<
            typename traits::template arg<I>::type>(
            std::get<I>(std::forward<TupleVector>(t))[unroll_index])...);

    results[unroll_index] =
        at::native::Memory::detail::bitwise_cast<result_t>(ret);
  }
}

template <int unroll_index>
struct apply_func_helper {
  template <
      typename Result,
      typename F,
      typename TupleVector,
      typename policy_t>
  static void apply(
      Result&& results,
      F&& f,
      TupleVector&& t,
      policy_t&& policy) {
    using Indices = std::make_index_sequence<
        std::tuple_size<std::decay_t<TupleVector>>::value>;
    apply_fun_impl<unroll_index>(
        std::forward<Result>(results),
        std::forward<F>(f),
        std::forward<TupleVector>(t),
        std::forward<policy_t>(policy),
        Indices{});
  }
};

template <int vec_size, typename func_t, typename policy_t>
inline void vec_elementwise_kernel_helper(func_t f, policy_t& policy) {
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  using vectorized_args_t =
      typename vectorized_args_tuple<vec_size, args_t>::vec_args_t;
  using return_t = typename traits::result_type;
  using vectorized_ret_t =
      typename vectorized_return_type<vec_size, return_t>::vec_ret_t;

  vectorized_ret_t results;
  vectorized_args_t args;

  // load
  policy.template load<args_t>(args);

  // unroll the compute multiple times
  native::Memory::detail::static_unroll<apply_func_helper, vec_size>::with_args(
      results, f, args, policy);

  // store
  policy.template store<return_t>(results);
}

template <typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  return_t results[THREAD_WORK_SIZE];
  args_t args[THREAD_WORK_SIZE];

  // load
  policy.load(args);

  // compute
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = c10::guts::apply(f, args[i]);
    }
  }

  // store
  policy.store(results);
}

template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void unrolled_elementwise_kernel(
    DPCPP::item<1> item_id,
    int numel,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int thread_idx = item_id.get_linear_id();

  int remaining = numel - thread_idx * vec_size;
  auto policy = at::native::Memory::policies::
      vec_unroll<vec_size, array_t, inp_calc_t, out_calc_t, loader_t, storer_t>(
          data, remaining, ic, oc, l, s, thread_idx);
  vec_elementwise_kernel_helper<vec_size>(f, policy);
}

template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using ret_t = typename traits::result_type;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int thread_num = (N + vec_size - 1) / vec_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      unrolled_elementwise_kernel<vec_size>(item_id, N, f, data, ic, oc, l, s);
    };

    cgh.parallel_for(DPCPP::range</*dim=*/1>(thread_num), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int vec_size, typename func_t, typename array_t>
void vectorized_elementwise_kernel(
    DPCPP::item<1> item_id,
    int numel,
    const func_t& fn,
    array_t data) {
  using traits = function_traits<func_t>;
  int thread_idx = item_id.get_linear_id();
  int remaining = numel - vec_size * thread_idx;

  if (remaining < vec_size) { // if this thread handles the remaining, just do a
                              // naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = at::native::Memory::LoadWithoutCast();
    auto storer = at::native::Memory::StoreWithoutCast();
    auto policy = at::native::Memory::policies::unroll<
        vec_size,
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        at::native::Memory::LoadWithoutCast,
        at::native::Memory::StoreWithoutCast>(
        data, remaining, input_calc, output_calc, loader, storer, thread_idx);
    elementwise_kernel_helper(fn, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
    // vectorized memory access
    auto policy = at::native::Memory::policies::vectorized<vec_size, array_t>(
        data, thread_idx);
    vec_elementwise_kernel_helper<vec_size>(fn, policy);
  }
}

// Assumption:
// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& fn,
    array_t data) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto vec_size = at::native::Memory::can_vectorize_up_to<func_t>(
      getDeviceIdOfCurrentQueue(), data);
  auto thread_num = (N + vec_size - 1) / vec_size;

#define VEC_LOOPS_KERNEL(vec_size)                                        \
  {                                                                       \
    auto cgf = DPCPP_Q_CGF(cgh) {                                         \
      cgh.parallel_for(                                                   \
          DPCPP::range<1>(thread_num), [=](DPCPP::item<1> itemId) {       \
            vectorized_elementwise_kernel<vec_size>(itemId, N, fn, data); \
          });                                                             \
    };                                                                    \
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);                                     \
  }

  switch (vec_size) {
    case 16: {
      VEC_LOOPS_KERNEL(16);
      break;
    }
    case 8: {
      VEC_LOOPS_KERNEL(8);
      break;
    }
    case 4: {
      VEC_LOOPS_KERNEL(4);
      break;
    }
    case 2: {
      VEC_LOOPS_KERNEL(2);
      break;
    }
    case 1: {
      VEC_LOOPS_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size", vec_size);
  }

#undef VEC_LOOPS_KERNEL
}

template <typename func_t>
void dpcpp_loops_kernel(TensorIterator& iter, const func_t f) {
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = at::native::needs_dynamic_casting<func_t>::check(iter);

  if (!dynamic_casting) {
    if (contiguous) {
      launch_vectorized_kernel(numel, f, data);
    } else {
      auto input_offset_calculator =
          make_input_offset_calculator<traits::arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      auto loader = at::native::Memory::LoadWithoutCast();
      auto storer = at::native::Memory::StoreWithoutCast();
      launch_unrolled_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    }
  } else {
    at::detail::Array<ScalarType, traits::arity> dtypes;
    for (int i = 0; i < traits::arity; i++) {
      dtypes[i] = iter.tensor(i + 1).scalar_type();
    }
    auto loader = at::native::Memory::LoadWithCast<traits::arity>(dtypes);
    auto storer =
        at::native::Memory::StoreWithCast(iter.tensor(0).scalar_type());

    if (contiguous) {
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    } else {
      auto input_offset_calculator =
          make_input_offset_calculator<traits::arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      launch_unrolled_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    }
  }
}

template <typename func_t>
void dpcpp_kernel_for_tensor_iter(TensorIterator& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).type() == at::kXPU);
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_kernel_for_tensor_iter(sub_iter, f);
    }
    return;
  }

  dpcpp_loops_kernel(iter, f);
}

template <typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  AUnaryFunctor(func_t f_, arg1_t a_) : f(f_), a(a_) {}

 private:
  func_t f;
  arg1_t a;
};

template <typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  BUnaryFunctor(func_t f_, arg2_t b_) : f(f_), b(b_) {}

 private:
  func_t f;
  arg2_t b;
};

template <typename func_t>
void dpcpp_kernel_with_scalars(TensorIterator& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "dpcpp_kernel_with_scalars only supports two input arguments");

  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<func_t> af(f, iter.scalar_value<arg1_t>(1));
    iter.remove_operand(1);
    const OptionalDeviceGuard device_guard(device_of(iter.tensor(1)));
    dpcpp_kernel_for_tensor_iter(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<func_t> bf(f, iter.scalar_value<arg2_t>(2));
    iter.remove_operand(2);
    dpcpp_kernel_for_tensor_iter(iter, bf);
  } else {
    dpcpp_kernel_for_tensor_iter(iter, f);
  }
}

template <typename func_t>
void dpcpp_small_index_kernel_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  auto numel = iter.numel();
  auto indices_size = iter.tensor(2).size(-1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_group_num = dpcppMaxDSSNum(dev_id) * OVER_SUBSCRIBE_DSS_FACTOR;

  auto total_index_iter = numel / indices_size;
  max_group_num = std::min(int64_t(total_index_iter / 2), max_group_num);

  // process the tail
  auto group_index_iter =
      (total_index_iter + max_group_num - 1) / max_group_num;
  auto group_num_tail = group_index_iter * max_group_num - total_index_iter;
  auto group_num = max_group_num - group_num_tail;
  auto group_numel = group_index_iter * indices_size;
  auto group_numel_tail = (group_index_iter - 1) * indices_size;

  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(group_numel), wgroup_size);
  auto global_size = max_group_num * wgroup_size;

  size_t num_non_indices = non_index_size.size();
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> src_sizes(0);
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> src_strides(0);
  for (size_t i = 0; i < num_non_indices; ++i) {
    src_sizes[i] = non_index_size[i];
    src_strides[i] = non_index_stride[i];
  }
  auto src_strides0 = non_index_stride[0];

  size_t num_indices = index_size.size();
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  int64_t element_size_bytes = iter.tensor(1).element_size();
  int64_t indice_size_bytes = iter.tensor(2).element_size();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = (char*)iter.data_ptr(0);
    auto in_data = (char*)iter.data_ptr(1);
    using index_buf_type = decltype((char*)iter.data_ptr(0));
    at::detail::Array<index_buf_type, MAX_TENSORINFO_DIMS> index_ptrs;
    for (size_t i = 0; i < num_indices; i++) {
      index_ptrs[i] = (char*)iter.data_ptr(i + 2);
    }

    using local_accessor_t = DPCPP::accessor<
        int64_t,
        1,
        DPCPP::access::mode::read_write,
        DPCPP::access::target::local>;
    auto local_offset = local_accessor_t(indices_size, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto local_id = item_id.get_local_id(0);
      auto group_id = item_id.get_group(0);

      // construct a indices_size table on SLM
      for (int64_t local_index = local_id; local_index < indices_size;
           local_index += wgroup_size) {
        int64_t offset = 0;
        for (size_t i = 0; i < num_indices; i++) {
          int64_t index =
              *(int64_t*)(index_ptrs[i] + local_index * indice_size_bytes);
          // if (index >= -sizes[i] && index < sizes[i]) {
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
          //} else {
          //  DPCPP_K_PRINT("index %ld out of bounds, expected [%ld, %ld)\n",
          //  index, -sizes[i], sizes[i]);
          //}
        }
        local_offset[local_index] = offset;
      }

      // calculate the number of workloads on each group
      auto group_linear_id = group_id * group_numel;
      auto group_numel_range = group_numel;
      if (group_num_tail && group_id >= group_num) {
        group_linear_id =
            group_num * group_numel + (group_id - group_num) * group_numel_tail;
        group_numel_range = group_numel_tail;
      }
      auto out_ptr = out_data;
      auto in_ptr = in_data;
      item_id.barrier(DPCPP::access::fence_space::local_space);

      // compute the in/out/indices offsets and perform memory copy
      for (int64_t local_index = local_id; local_index < group_numel_range;
           local_index += wgroup_size) {
        auto linear_id = group_linear_id + local_index;
        auto out_offset = linear_id * element_size_bytes;
        auto src_linear_id = linear_id / indices_size;
        int64_t in_offset = 0;
        for (int i = num_non_indices - 1; i > 0; --i) {
          in_offset += (src_linear_id % src_sizes[i]) * src_strides[i];
          src_linear_id /= src_sizes[i];
        }
        in_offset += src_linear_id * src_strides0;

        auto offset = local_offset[local_index % indices_size];
        f(out_ptr + out_offset, in_ptr + in_offset, offset);
      }
    };
    __cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(wgroup_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
void dpcpp_index_kernel_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    const func_t f) {
  size_t num_indices = index_size.size();
  auto numel = iter.numel();
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = (char*)iter.data_ptr(0);
    auto in_data = (char*)iter.data_ptr(1);
    using index_buf_type = decltype((char*)iter.data_ptr(0));
    at::detail::Array<index_buf_type, MAX_TENSORINFO_DIMS> index_ptrs;
    for (size_t i = 0; i < num_indices; i++) {
      index_ptrs[i] = (char*)iter.data_ptr(i + 2);
    }

    auto offset_calc = make_offset_calculator<3>(iter);
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto linear_idx = item_id.get_linear_id();
      auto offsets = offset_calc.get(linear_idx);
      auto out_ptr = out_data + offsets[0];
      auto in_ptr = in_data + offsets[1];
      int64_t offset = 0;
      //#pragma unroll
      for (size_t i = 0; i < num_indices; i++) {
        int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
        if (index >= -sizes[i] && index < sizes[i]) {
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        } else {
          DPCPP_K_PRINT(
              "index %ld out of bounds, expected [%ld, %ld)\n",
              index,
              -sizes[i],
              sizes[i]);
        }
      }
      f(out_ptr, in_ptr, offset);
    };
    __cgh.parallel_for(DPCPP::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
void dpcpp_index_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  auto numel = iter.numel();

  if (numel == 0) {
    return;
  }

  size_t num_indices = index_size.size();
  TORCH_INTERNAL_ASSERT(num_indices == index_stride.size());
  TORCH_INTERNAL_ASSERT(
      num_indices == static_cast<size_t>(iter.ntensors()) - 2);
  TORCH_INTERNAL_ASSERT(num_indices <= MAX_TENSORINFO_DIMS);

  // the dpcpp_small_index_kernel_impl is applied for last several successive
  // dims indexing of an input tensor Taking 3-dims tensor input
  // (input.shape=[x,y,z]) for example: input[:,:,idx] or input[:,idx1,idx2]
  // when input tensor satisfies the following conditions, the
  // small_index_kernel path will be selected: 1.there are common indices such
  // as input[:,:,idx] and input[:,idx1,idx2] instead of
  //   input[idx0,idx1,idx2], input[idx0,idx1,:], input[idx0,:,idx2],
  //   input[idx0,:,:], input[:,idx1,:]
  // 2.the common indices numel should larger than 2 times of the
  // dpcppMaxComputeUnitSize (then we can get memory access benifit) 3.the
  // workloads in each group should larger than the maximum number of workitem
  // (ensure all the workitem activate) 4.the indices_table size should
  // satisfied the SLM limit condition

  // check whether the current case satisfying the condition 1
  // Taking input[idx0,:,idx2] for example, the indices_sizes=[sz,1,sz]
  // While the satified case is input[:,idx1,idx2], indices_sizes=[1,sz,sz]
  bool small_index = non_index_size.size() != 0;
  auto indices_sizes = iter.tensor(2).sizes();
  for (size_t i = 1; i < num_indices; ++i) {
    if (indices_sizes[i - 1] > indices_sizes[i]) {
      small_index = false;
      break;
    }
  }
  if (small_index) {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t max_group_num = dpcppMaxDSSNum(dev_id);
    auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
    auto indices_size = iter.tensor(2).size(-1);
    auto total_index_iter = numel / indices_size;
    auto local_index = numel / max_group_num;

    // the max_local_mem_size = 65536B (64KB)
    auto max_local_mem_size = dpcppLocalMemSize(dev_id);
    auto indice_table_size = indices_size * sizeof(int64_t);

    // check whether the current case satisfying conditions 2,3,4
    small_index =
        (total_index_iter > 2 * max_group_num && local_index > wgroup_size &&
         indice_table_size < max_local_mem_size * 0.5);
    if (small_index) {
      dpcpp_small_index_kernel_impl<func_t>(
          iter, index_size, index_stride, non_index_size, non_index_stride, f);
      return;
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_index_kernel(
          sub_iter, index_size, index_stride, IntArrayRef{}, IntArrayRef{}, f);
    }
    return;
  }

  dpcpp_index_kernel_impl<func_t>(iter, index_size, index_stride, f);
}

} // namespace AtenIpexTypeXPU
} // namespace at
