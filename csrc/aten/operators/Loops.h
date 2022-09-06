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
#include "comm/Load.h"

#define UNROLLED_ELEM_PER_WORK_ITEM 4
#define LOOPS_UNROLL_WORK_SIZE 16

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides>
make_input_offset_calculator(const TensorIteratorBase& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(
      N <= iter.ntensors() - iter.noutputs()); // TODO: using N == ...
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N, uint32_t, signed_strides>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1, bool signed_strides = false>
static OffsetCalculator<num_outputs, uint32_t, signed_strides>
make_output_offset_calculator(const TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs, uint32_t, signed_strides>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int WORK_SIZE, typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  return_t results[WORK_SIZE];
  args_t args[WORK_SIZE];

  // load
  policy.load(args);

  // compute
#pragma unroll
  for (int i = 0; i < WORK_SIZE; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = c10::guts::apply(f, args[i]);
    }
  }

  // store
  policy.store(results);
}

template <
    int ITEM_WORK_SIZE,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void unrolled_elementwise_kernel(
    sycl::nd_item<1>& item,
    int numel,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int group_items = item.get_local_range(0);
  int thread_idx = item.get_local_id(0);
  int group_idx = item.get_group(0);
  int remaining = numel - ITEM_WORK_SIZE * group_items * group_idx;
  auto policy = at::native::Memory::policies::unroll<
      ITEM_WORK_SIZE,
      array_t,
      inp_calc_t,
      out_calc_t,
      loader_t,
      storer_t>(
      data, remaining, ic, oc, l, s, thread_idx, group_idx, group_items);
  elementwise_kernel_helper<ITEM_WORK_SIZE>(f, policy);
}

template <
    int ITEM_WORK_SIZE,
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
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int group_items = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int group_work_size = ITEM_WORK_SIZE * group_items;
  int num_groups = (N + group_work_size - 1) / group_work_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      unrolled_elementwise_kernel<ITEM_WORK_SIZE>(
          item, N, f, data, ic, oc, l, s);
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * group_items),
            sycl::range<1>(group_items)),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    int ITEM_WORK_SIZE,
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t>
void vectorized_elementwise_kernel(
    sycl::nd_item<1>& item,
    int numel,
    func_t fn,
    array_t data,
    inp_calc_t input_calc) {
  int group_items = item.get_local_range(0);
  int thread_idx = item.get_local_id(0);
  int group_idx = item.get_group(0);
  int group_work_size = ITEM_WORK_SIZE * group_items;
  int remaining = numel - group_idx * group_work_size;
  if (remaining <
      group_work_size) { // if this thread handles the remaining, just do a
    // naive unrolled loop
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = at::native::Memory::LoadWithoutCast();
    auto storer = at::native::Memory::StoreWithoutCast();
    auto policy = at::native::Memory::policies::unroll<
        ITEM_WORK_SIZE,
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        at::native::Memory::LoadWithoutCast,
        at::native::Memory::StoreWithoutCast>(
        data,
        remaining,
        input_calc,
        output_calc,
        loader,
        storer,
        thread_idx,
        group_idx,
        group_items);
    elementwise_kernel_helper<ITEM_WORK_SIZE>(fn, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
    // vectorized memory access
    auto policy = at::native::Memory::policies::
        vectorized<ITEM_WORK_SIZE, vec_size, array_t, inp_calc_t>(
            data, input_calc, thread_idx, group_idx, group_items);
    elementwise_kernel_helper<ITEM_WORK_SIZE>(fn, policy);
  }
}

constexpr int max_scalar_size_(std::tuple<>) {
  return 0;
}

template <typename scalar_t, typename... types>
constexpr int max_scalar_size_(std::tuple<scalar_t, types...>) {
  return std::max<int>(
      sizeof(scalar_t), max_scalar_size_(std::tuple<types...>{}));
}

template <typename func_t>
constexpr static inline int max_scalar_size() {
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  constexpr auto size = max_scalar_size_(args_t{});
  using return_t = typename traits::result_type;
  return std::max<int>(sizeof(return_t), size);
}

template <typename scalar_t>
constexpr inline bool check_double() {
  return std::is_same<scalar_t, double>::value ||
      std::is_same<scalar_t, c10::complex<double>>::value;
}

constexpr bool has_double_arg_(std::tuple<>) {
  return false;
}

template <typename scalar_t, typename... types>
constexpr bool has_double_arg_(std::tuple<scalar_t, types...>) {
  return check_double<scalar_t>() || has_double_arg_(std::tuple<types...>{});
}

template <typename func_t>
static inline bool has_double_arg(TensorIteratorBase& iter) {
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  using return_t = typename traits::result_type;
  for (int i = 0; i < iter.ntensors(); i++) {
    if (iter.tensor(i).scalar_type() == at::kDouble ||
        iter.tensor(i).scalar_type() == at::kComplexDouble)
      return true;
  }
  return check_double<return_t>() || has_double_arg_(args_t{});
}

// Assumption:
// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t, typename inp_calc_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& fn,
    array_t data,
    inp_calc_t input_calc,
    int vec_size) {
  constexpr auto max_scalar_bytes = max_scalar_size<func_t>();
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto group_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());

#define VEC_LOOPS_KERNEL(vec_size)                                    \
  {                                                                   \
    TORCH_CHECK(max_scalar_bytes* vec_size <= 16);                    \
    if constexpr (max_scalar_bytes * vec_size <= 16) {                \
      auto cgf = DPCPP_Q_CGF(cgh) {                                   \
        int group_work_size = group_size * vec_size;                  \
        int num_groups = (N + group_work_size - 1) / group_work_size; \
        cgh.parallel_for(                                             \
            sycl::nd_range<1>(                                        \
                sycl::range<1>(num_groups * group_size),              \
                sycl::range<1>(group_size)),                          \
            [=](sycl::nd_item<1> itemId) {                            \
              {                                                       \
                vectorized_elementwise_kernel<vec_size, vec_size>(    \
                    itemId, N, fn, data, input_calc);                 \
              }                                                       \
            });                                                       \
      };                                                              \
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);                               \
    }                                                                 \
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

template <int vec_size, typename func_t>
static inline void elementwise_kernel(
    sycl::nd_item<1>& item,
    int N,
    func_t f,
    int group_size) {
  int group_work_size = group_size * vec_size;
  int idx = group_work_size * item.get_group(0) + item.get_local_id(0);
#pragma unroll
  for (int i = 0; i < vec_size; i++) {
    if (idx < N) {
      f(idx);
      idx += group_size;
    }
  }
}

template <int vec_size, typename func_t>
static inline void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto max_group_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  auto num_groups =
      (N + max_group_size * vec_size - 1) / (max_group_size * vec_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      elementwise_kernel<vec_size, func_t>(item_id, N, f, max_group_size);
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * max_group_size),
            sycl::range<1>(max_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
typename traits::result_type invoke_impl(
    const func_t& f,
    char* const DPCPP_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(
      at::AtenIpexTypeXPU::load<typename traits::template arg<INDEX>::type>(
          data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
typename traits::result_type invoke(
    const func_t& f,
    char* const DPCPP_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <
    bool REMOVE_DOUBLE,
    typename traits,
    typename func_t,
    typename index_t,
    size_t... I>
typename traits::result_type invoke_with_cast_impl(
    const func_t& f,
    char* const DPCPP_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  if constexpr (REMOVE_DOUBLE) {
    return f(at::native::Memory::no_double_fetch_and_cast<
             typename traits::template arg<I>::type>(
        dtypes[I], data[I] + i * strides[I])...);
  } else {
    return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
        dtypes[I], data[I] + i * strides[I])...);
  }
}

template <
    bool REMOVE_DOUBLE,
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
typename traits::result_type invoke_with_cast(
    const func_t& f,
    char* const DPCPP_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_with_cast_impl<REMOVE_DOUBLE, traits>(
      f, data, strides, dtypes, i, Indices{});
}

template <typename func_t, typename data_t>
static inline bool can_use_broadcast_vectorize(
    TensorIteratorBase& iter,
    const data_t& data,
    int& vec_size) {
  if (iter.is_contiguous() || !iter.has_contiguous_first_dim() ||
      !iter.tensor(0).is_contiguous())
    return false;
  vec_size = at::native::Memory::can_vectorize_up_to_loop<func_t>(
      getDeviceIdOfCurrentQueue(), data);
  if (vec_size <= 1)
    return false;
  int last_compute_dim_size = iter.shape()[0];
  while (last_compute_dim_size % vec_size)
    vec_size >>= 1;
  for (int i = 0; i < iter.ntensors(); i++) {
    auto strides = iter.strides(i);
    for (int dim = 1; dim < strides.size(); dim++) {
      while (strides[dim] % (strides[0] * vec_size))
        vec_size >>= 1;
    }
  }
  return vec_size > 1;
}

template <typename func_t, bool signed_strides = false, bool fast_mode = false>
void dpcpp_loops_kernel(TensorIteratorBase& iter, const func_t f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() >= traits::arity);
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
      int vec_size = at::native::Memory::can_vectorize_up_to_loop<func_t>(
          getDeviceIdOfCurrentQueue(), data);
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      launch_vectorized_kernel(
          numel, f, data, input_offset_calculator, vec_size);
    } else {
      if constexpr (fast_mode) {
        int vec_size;
        if (can_use_broadcast_vectorize<func_t>(iter, data, vec_size) &&
            !signed_strides) {
          auto input_offset_calculator =
              make_input_offset_calculator<traits::arity, signed_strides>(iter);
          launch_vectorized_kernel(
              numel, f, data, input_offset_calculator, vec_size);
          return;
        }
      }
      auto offset_calc =
          make_offset_calculator<traits::arity + 1, signed_strides>(iter);
      constexpr int unroll_factor = 16 / sizeof(arg0_t);
      launch_legacy_kernel<unroll_factor>(numel, [=](int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      });
    }
  } else {
    at::detail::Array<ScalarType, traits::arity> dtypes;
    for (int i = 0; i < traits::arity; i++) {
      dtypes[i] = iter.tensor(i + 1).scalar_type();
    }

#define HANDLE_DYNAMIC_CAST(REMOVE_DOUBLE)                                     \
  {                                                                            \
    if (contiguous) {                                                          \
      auto loader =                                                            \
          at::native::Memory::LoadWithCast<traits::arity, REMOVE_DOUBLE>(      \
              dtypes);                                                         \
      auto storer = at::native::Memory::StoreWithCast<REMOVE_DOUBLE>(          \
          iter.tensor(0).scalar_type());                                       \
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>(); \
      auto output_offset_calculator = TrivialOffsetCalculator<1>();            \
      launch_unrolled_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(                     \
          numel,                                                               \
          f,                                                                   \
          data,                                                                \
          input_offset_calculator,                                             \
          output_offset_calculator,                                            \
          loader,                                                              \
          storer);                                                             \
    } else {                                                                   \
      at::detail::Array<ScalarType, ntensors> dtypes;                          \
      for (int i = 0; i < ntensors; i++) {                                     \
        dtypes[i] = iter.dtype(i);                                             \
      }                                                                        \
      auto offset_calc =                                                       \
          make_offset_calculator<traits::arity + 1, signed_strides>(iter);     \
      launch_legacy_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(numel, [=](int idx) {  \
        auto offsets = offset_calc.get(idx);                                   \
        void* out = data[0] + offsets[0];                                      \
        arg0_t result = invoke_with_cast<REMOVE_DOUBLE>(                       \
            f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);           \
        if constexpr (REMOVE_DOUBLE)                                           \
          at::native::Memory::no_double_cast_and_store<arg0_t>(                \
              dtypes[0], out, result);                                         \
        else                                                                   \
          c10::cast_and_store<arg0_t>(dtypes[0], out, result);                 \
      });                                                                      \
    }                                                                          \
  }

    if constexpr (fast_mode) {
      if (!has_double_arg<func_t>(iter)) {
        HANDLE_DYNAMIC_CAST(true)
        return;
      }
    }

    HANDLE_DYNAMIC_CAST(false)

#undef HANDLE_DYNAMIC_CAST
  }
}

template <typename func_t, bool signed_strides = false>
void dpcpp_kernel_for_tensor_iter(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).type() == at::kXPU,
        "argument ",
        arg,
        ": expected a XPU device but found ",
        iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_kernel_for_tensor_iter<func_t, signed_strides>(sub_iter, f);
    }
    return;
  }

  dpcpp_loops_kernel<func_t, signed_strides, false>(iter, f);
}

template <typename func_t, bool signed_strides = false>
void dpcpp_fast_mode_kernel_for_tensor_iter(
    TensorIteratorBase& iter,
    const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).type() == at::kXPU,
        "argument ",
        arg,
        ": expected a XPU device but found ",
        iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_fast_mode_kernel_for_tensor_iter<func_t, signed_strides>(
          sub_iter, f);
    }
    return;
  }

  dpcpp_loops_kernel<func_t, signed_strides, true>(iter, f);
}

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  // NB: scalar is stored in higher precision!
  AUnaryFunctor(func_t f_, opmath_arg1_t a_) : f(f_), a(a_) {}

 private:
  func_t f;
  opmath_arg1_t a;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  // NB: scalar is stored in higher precision!
  BUnaryFunctor(func_t f_, opmath_arg2_t b_) : f(f_), b(b_) {}

 private:
  func_t f;
  opmath_arg2_t b;
};

// Though seemingly noop, this inserts casts from arg1_t to func_t's type
// (which may be higher precision), as well as casts to return_t
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_) : f(f_) {}

 private:
  func_t f;
};

// Unlike gpu_kernel_with_scalars, this allows you to pass a func_t which
// accepts inputs at higher precision (typically opmath_t), but then
// ensure that we load from memory at the correct precision (scalar_t)
// to avoid expensive loads.  For the whole sordid story see
// https://dev-discuss.pytorch.org/t/cuda-loops-case-study-code-generation-vs-templates/302
template <
    typename arg1_t,
    typename arg2_t = arg1_t,
    typename return_t = arg1_t,
    typename func_t,
    bool fast_mode>
void opmath_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(
        f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(device_of(iter.tensor(1)));
    if constexpr (fast_mode)
      dpcpp_fast_mode_kernel_for_tensor_iter(iter, af);
    else
      dpcpp_kernel_for_tensor_iter(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(
        f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    if constexpr (fast_mode)
      dpcpp_fast_mode_kernel_for_tensor_iter(iter, bf);
    else
      dpcpp_kernel_for_tensor_iter(iter, bf);
  } else {
    if constexpr (fast_mode)
      dpcpp_fast_mode_kernel_for_tensor_iter(
          iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
    else
      dpcpp_kernel_for_tensor_iter(
          iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

template <typename func_t>
void dpcpp_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "dpcpp_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t, false>(
      iter, f);
}

template <typename func_t>
void dpcpp_fast_mode_kernel_with_scalars(
    TensorIteratorBase& iter,
    const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "dpcpp_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t, true>(
      iter, f);
}

} // namespace AtenIpexTypeXPU
} // namespace at
