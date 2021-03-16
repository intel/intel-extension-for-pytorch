#pragma once

namespace torch_ipex {
namespace autocast {

using at::IntArrayRef;
using at::Tensor;
using at::TensorList;
using namespace c10;

#define INT8_DTYPE_PRIORITY 2 //INT8
#define BF16_DTYPE_PRIORITY 1 //BF16
#define FP32_DTYPE_PRIORITY 0 //FP32

enum class DtypeCastPolicy : uint8_t {
  user_defined_dtype=0,
  fp32,  // Cast all inputs to at::kHalf before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //   1. we'd like to run in fp32 and
                      //   2. have a c10::optional<ScalarType> arg that controls the output type.
                      // fp32_set_opt_dtype wrappers' policy is:  if the output type is already set,
                      // don't touch it, otherwise, set it to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //   1. we'd like to run in fp32 and
                     //   2. have some overloads that accept an output type and other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't have an output dtype.
                     // The wrapper policy is:  append at::kFloat to the args, and redispatch to the
                     // type-aware overload.
  promote, // Run in the widest dtype among several args.
};

bool is_autocast_enabled();
void set_autocast_enabled(bool new_enabled);
at::ScalarType get_autocast_dtype();
void set_autocast_dtype(at::ScalarType dtype);
int autocast_increment_nesting();
int autocast_decrement_nesting();
void clear_autocast_cache();

Tensor cpu_cached_cast(at::ScalarType to_type, const Tensor& arg);

inline c10::optional<Tensor> cpu_cached_cast(at::ScalarType to_type, const c10::optional<Tensor>& arg) {
  if (arg.has_value()) {
    return cpu_cached_cast(to_type, *arg);
  } else {
    return c10::nullopt;
  }
}

inline std::vector<Tensor> cpu_cached_cast(at::ScalarType to_type, const TensorList& arg) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cpu_cached_cast(to_type, t));
  }
  return vec;
}

template <typename T>
inline T cpu_cached_cast(at::ScalarType to_type, T arg) {
  return arg;
}

/****************************************************
Logic to apply cached casting to any Tensor argument.
****************************************************/
inline bool is_eligible_cpu(const Tensor& arg) {
  return (arg.defined() && arg.is_floating_point() &&
          (arg.scalar_type() != at::kDouble));
}

template<typename T>
std::map<int, T> flip_map(std::map<T, int> input) {
  std::map<int, T> reversed;
  for (typename std::map<T, int>::iterator i = input.begin(); i != input.end(); ++i)
    reversed[i->second] = i->first;
  return reversed;
}

}  // namespace autocast
}  // namespace torch_ipex
