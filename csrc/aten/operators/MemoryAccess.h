#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>


namespace at { namespace native { namespace Memory {

namespace detail {

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

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline  void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline  void with_args(Args... args) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static  void apply(policy_t &self, args_t *args) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]);
    auto args_accessor = [&args]  (int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
    self.load_single_arg(args_accessor, ptr);
  }
};

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static  void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<int ntensors, int num_outputs, typename ...Args>
   static void apply(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<uint32_t, num_outputs> offsets,
      std::tuple<Args...> ret) {
    using T = typename std::tuple_element<current, std::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = std::get<current>(ret);
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
   scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return *(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

template <int N>
struct LoadWithCast {
  using array_t = at::detail::Array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = at::detail::Array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  template<typename array_t_>
  LoadWithCast(array_t_ dtypes) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      this->dtypes[i] = dtypes[i];
      element_sizes[i] = c10::elementSize(dtypes[i]);
    }
  }

  template<typename scalar_t>
   scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    return c10::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
   void store(scalar_t value, char *base_ptr, uint32_t offset) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

struct StoreWithCast {
  at::ScalarType dtype;
  uint32_t element_size;
  StoreWithCast(at::ScalarType dtype): dtype(dtype), element_size(c10::elementSize(dtype)) {}
  template<typename scalar_t>
   void store(scalar_t value, char *base_ptr, uint32_t offset) {
    void *ptr = base_ptr + element_size * offset;
    c10::cast_and_store<scalar_t>(dtype, ptr, value);
  }
};

// aligned vector generates vectorized load/store on XPU
template<int N_BYTES>
struct aligned_element {};
template<>
struct aligned_element<1> {
  using element_type = uint8_t ;
};

template<>
struct aligned_element<2> {
  using element_type = uint16_t ;
};

template<>
struct aligned_element<4> {
  using element_type = uint32_t ;
};

template<>
struct aligned_element<8> {
  using element_type = uint64_t ;
};

template<typename scalar_t, int vec_size>
struct aligned_vector {
  using element_type = typename aligned_element<sizeof(scalar_t)>::element_type;
  using type = DPCPP::vec<element_type , vec_size>;
};

template <typename TO, typename FROM>
inline TO bitwise_cast(FROM value) {
  static_assert(sizeof(TO) == sizeof(FROM), "in-compatible type size in bitwise_cast.");
  TO transport_bits = *((TO*)&value);
  return transport_bits;
}

namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t, int num_outputs = 1>
struct unroll {

  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  int thread_idx;

   unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s, int thread_idx):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s), thread_idx(thread_idx) {}

   inline bool check_inbounds(int thread_work_elem) {
     return ((/*thread_idx*THREAD_WORK_SIZE + */thread_work_elem) < remaining);
  }

  template<typename args_t>
   inline void load(args_t *args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if (i >= remaining) {
        return;
      }
      int linear_idx = thread_idx*THREAD_WORK_SIZE + i;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
    }
  }

  template<typename scalar_t>
   inline void store(scalar_t *from) {
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
      if (i >= remaining) {
        return;
      }
      int linear_idx = thread_idx*THREAD_WORK_SIZE + i;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
    }
  }
};

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
// Note:
// Functions in vectorized policy does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the the caller manually.
template <int vec_size, typename data_t>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(THREAD_WORK_SIZE % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = THREAD_WORK_SIZE / vec_size;

  data_t data;
  int thread_idx;

  vectorized(data_t data, int thread_idx) : data(data), thread_idx(thread_idx) {}

  inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template<typename accessor_t, typename scalar_t>
   inline void load_single_arg(accessor_t to, scalar_t *from) {
    using vec_t = typename aligned_vector<scalar_t, vec_size>::type;
    vec_t *from_ = reinterpret_cast<vec_t *>(from);
    for (int i = 0; i < loop_size; i++) {
      int linear_idx = thread_idx + i;
      vec_t v = from_[linear_idx];
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = bitwise_cast<scalar_t>(v[j]);
      }
    }
  }

  template<typename args_t>
   inline void load(args_t *args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args);
  }

  template<typename scalar_t>
   inline void store(scalar_t *from) {
    using vec_t = typename aligned_vector<scalar_t, vec_size>::type;
    using element_t = typename aligned_vector<scalar_t, vec_size>::element_type;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0])/* + THREAD_WORK_SIZE * thread_idx*/;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    element_t *from_ = reinterpret_cast<element_t *>(from);
    for (int i = 0; i < loop_size; i++) {
      int linear_idx = thread_idx + i;
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v[j] = from_[vec_size * i + j];
      }
      to_[linear_idx] = v;
    }
  }
};

}  // namespace policies

// This is only used in host, but we will wrap this into some templates
// which is , so we have to make this 
// in order to compile
template<typename scalar_t>
inline  int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static  void apply(int &result, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
  return result;
}

}}} // namespace at::native::Memory
