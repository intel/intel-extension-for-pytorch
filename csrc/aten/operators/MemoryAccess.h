#pragma once

#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <cstdint>
#include <type_traits>

namespace at {
namespace native {
namespace Memory {

namespace detail {

template <typename TO, typename FROM>
inline std::decay_t<TO> bitwise_cast(FROM&& value) {
  static_assert(
      sizeof(TO) == sizeof(FROM), "in-compatible type size in bitwise_cast.");
  std::decay_t<TO> transport_bits = *((std::decay_t<TO>*)&value);
  return transport_bits;
}

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template <template <int i> typename func, int end, int current = 0>
struct static_unroll {
  template <typename... Args>
  static inline void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current + 1>::with_args(args...);
  }
};

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
  template <typename... Args>
  static inline void with_args(Args... args) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one
template <int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t, typename offset_t>
  static void apply(
      policy_t& self,
      args_t* args,
      offset_t offset,
      int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    auto ptr = reinterpret_cast<arg_t*>(self.data[arg_index + num_outputs]) +
        offset[arg_index];
    auto args_accessor = [&args](int thread_unroll_idx) -> arg_t& {
      return std::get<arg_index>(args[thread_unroll_idx]);
    };
    self.load_single_arg(args_accessor, ptr);
  }
};

template <int arg_index>
struct unroll_load_helper {
  template <
      typename args_t,
      typename policy_t,
      typename offset_t,
      typename loader_t>
  static void apply(
      policy_t& self,
      args_t* args,
      offset_t offset,
      loader_t loader,
      int j,
      int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(
        self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template <int ntensors, int num_outputs, typename... Args>
  static void apply(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<uint32_t, num_outputs> offsets,
      std::tuple<Args...> ret) {
    using T = typename std::tuple_element<current, std::tuple<Args...>>::type;
    T* to = reinterpret_cast<T*>(data[current]) + offsets[current];
    *to = std::get<current>(ret);
  }
};

} // namespace detail

struct LoadWithoutCast {
  template <typename scalar_t, typename offset_t>
  scalar_t load(char* base_ptr, offset_t offset, int arg) {
    return *(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }
};

template <int N>
struct LoadWithCast {
  using array_t = at::detail::Array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = at::detail::Array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  template <typename array_t_>
  LoadWithCast(array_t_ dtypes) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      this->dtypes[i] = dtypes[i];
      element_sizes[i] = c10::elementSize(dtypes[i]);
    }
  }

  template <typename scalar_t, typename offset_t>
  scalar_t load(char* base_ptr, offset_t offset, int arg) {
    void* ptr = base_ptr + element_sizes[arg] * offset;
    return c10::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
  }
};

struct StoreWithoutCast {
  template <typename scalar_t, typename offset_t>
  void store(scalar_t value, char* base_ptr, offset_t offset) {
    *(reinterpret_cast<scalar_t*>(base_ptr) + offset) = value;
  }
};

struct StoreWithCast {
  at::ScalarType dtype;
  uint32_t element_size;
  StoreWithCast(at::ScalarType dtype)
      : dtype(dtype), element_size(c10::elementSize(dtype)) {}
  template <typename scalar_t, typename offset_t>
  void store(scalar_t value, char* base_ptr, offset_t offset) {
    void* ptr = base_ptr + element_size * offset;
    c10::cast_and_store<scalar_t>(dtype, ptr, value);
  }
};

// aligned vector generates vectorized load/store on XPU
template <int N_BYTES>
struct aligned_element {};
template <>
struct aligned_element<1> {
  using element_type = uint8_t;
};

template <>
struct aligned_element<2> {
  using element_type = uint16_t;
};

template <>
struct aligned_element<4> {
  using element_type = uint32_t;
};

template <>
struct aligned_element<8> {
  using element_type = uint64_t;
};

template <typename scalar_t, int vec_size>
struct aligned_vector {
  using element_type = typename aligned_element<sizeof(scalar_t)>::element_type;
  using type = DPCPP::vec<element_type, vec_size>;
};

// aligned vector generates vectorized load/store on XPU
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector_loop {
  scalar_t val[vec_size];

  scalar_t& operator[](int index) {
    return val[index];
  }

  scalar_t const& operator[](int index) const {
    return val[index];
  }
};

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template <
    int vec_size,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int num_outputs = 1>
struct unroll {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  int thread_idx;

  unroll(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s,
      int thread_idx)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        loader(l),
        storer(s),
        thread_idx(thread_idx) {}

  inline bool check_inbounds(int thread_work_elem) const {
    return (thread_work_elem < remaining);
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      if (i >= remaining) {
        return;
      }
      int linear_idx = thread_idx * vec_size + i;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(
          *this, args, offset, loader, i, num_outputs);
    }
  }

  template <typename scalar_t>
  inline void store(scalar_t* from) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      if (i >= remaining) {
        return;
      }
      int linear_idx = thread_idx * vec_size + i;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
    }
  }
};

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
// Note:
// Functions in vectorized policy does not do boundary check. It assumes the
// whole block has its job to do. So the reminders should be handled by the the
// caller manually.
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template <
    int vec_size,
    typename data_t,
    typename input_offset_calc,
    typename output_offset_calc,
    int num_outputs = 1>
struct vectorized {
  data_t data;
  input_offset_calc& input_offset_calculator;
  output_offset_calc& output_offset_calculator;
  int thread_idx;
  int vec_idx;

  vectorized(
      data_t data,
      input_offset_calc& ic,
      output_offset_calc& oc,
      int thread_idx)
      : data(data),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        thread_idx(thread_idx),
        vec_idx(thread_idx * vec_size) {}

  inline constexpr bool check_inbounds(int thread_work_elem) const {
    return true;
  }

  template <typename accessor_t, typename scalar_t>
  inline void load_single_arg(accessor_t to, scalar_t* from) {
    using vec_t = aligned_vector_loop<scalar_t, vec_size>;
    vec_t v = reinterpret_cast<vec_t*>(from)[0];
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      to(i) = v.val[i];
    }
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    auto offset = input_offset_calculator.get(vec_idx);
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(
        *this, args, offset, num_outputs);
  }

  template <typename scalar_t>
  inline void store(scalar_t* from) {
    using vec_t = aligned_vector_loop<scalar_t, vec_size>;
    auto offset = output_offset_calculator.get(vec_idx);
    auto ptr = reinterpret_cast<scalar_t*>(data[0]) + offset[0];
    vec_t* to = reinterpret_cast<vec_t*>(ptr);
    vec_t v;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      v.val[j] = from[j];
    }
    *to = v;
  }
};

} // namespace policies

static inline int preferred_vector_width(DeviceId dev_id, int elem_sz) {
  size_t ret;
  switch (elem_sz) {
    case 1:
      static_assert(sizeof(char) == 1, "the char size is not 2 bytes");
      ret = xpu::dpcpp::dpcppPrefVectorWidth<char>(dev_id);
      break;
    case 2:
      static_assert(sizeof(short) == 2, "the short size is not 2 bytes");
      ret = xpu::dpcpp::dpcppPrefVectorWidth<short>(dev_id);
      break;
    case 4:
      ret = xpu::dpcpp::dpcppPrefVectorWidth<int>(dev_id);
      static_assert(sizeof(int) == 4, "the long size is not 4 bytes");
      break;
    case 8:
      static_assert(sizeof(long) == 8, "the long size is not 8");
      ret = xpu::dpcpp::dpcppPrefVectorWidth<long>(dev_id);
      break;
    default:
      // no vectorize
      ret = 1;
  }
  return ret;
}

// This is only used in host, but we will wrap this into some templates
// which is , so we have to make this
// in order to compile
template <typename scalar_t>
inline int can_vectorize_up_to(DeviceId dev_id, char* pointer) {
  int elem_size = sizeof(scalar_t);
  int preferred_width = preferred_vector_width(dev_id, elem_size);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 2>::type>::value;
  constexpr int vec4_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 4>::type>::value;
  constexpr int vec8_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 8>::type>::value;
  constexpr int vec16_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 16>::type>::value;
  if (address % vec16_alignment == 0) {
    return std::min<int>(preferred_width, 16);
  } else if (address % vec8_alignment == 0) {
    return std::min<int>(preferred_width, 8);
  } else if (address % vec4_alignment == 0) {
    return std::min<int>(preferred_width, 4);
  } else if (address % vec2_alignment == 0) {
    return std::min<int>(preferred_width, 2);
  }
  return 1;
}

template <int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static void apply(int& result, DeviceId dev_id, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...],
    // so we need a +1 offset to get the input
    result = std::min<int>(
        result, can_vectorize_up_to<arg_t>(dev_id, pointers[i + 1]));
  }
};

template <typename func_t, typename array_t>
inline int can_vectorize_up_to(DeviceId dev_id, array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(dev_id, pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(
      result, dev_id, pointers, traits());
  return result;
}

// This is only used in host, but we will wrap this into some templates
// which is , so we have to make this
// in order to compile
template <typename scalar_t>
inline int can_vectorize_up_to_loop(DeviceId dev_id, char* pointer) {
  int elem_size = sizeof(scalar_t);
  int preferred_width = preferred_vector_width(dev_id, elem_size);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 2>>::value;
  constexpr int vec4_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 4>>::value;
  constexpr int vec8_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 8>>::value;
  constexpr int vec16_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 16>>::value;
  if (address % vec16_alignment == 0) {
    return std::min<int>(preferred_width, 16);
  } else if (address % vec8_alignment == 0) {
    return std::min<int>(preferred_width, 8);
  } else if (address % vec4_alignment == 0) {
    return std::min<int>(preferred_width, 4);
  } else if (address % vec2_alignment == 0) {
    return std::min<int>(preferred_width, 2);
  }
  return 1;
}

template <int i>
struct can_vectorize_up_to_helper_loop {
  template <typename array_t, typename traits>
  static void apply(int& result, DeviceId dev_id, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...],
    // so we need a +1 offset to get the input
    result = std::min<int>(
        result, can_vectorize_up_to_loop<arg_t>(dev_id, pointers[i + 1]));
  }
};

template <typename func_t, typename array_t>
inline int can_vectorize_up_to_loop(DeviceId dev_id, array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to_loop<return_t>(dev_id, pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper_loop, arity>::with_args(
      result, dev_id, pointers, traits());
  return result;
}

} // namespace Memory
} // namespace native
} // namespace at
