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

#define MKLDNN_LAYOUT_PRIORITY 1 //MKLDNN
#define STRIDED_LAYOUT_PRIORITY 0 //STRIDED

enum class DtypeCastPolicy : uint8_t {
  fp32 = 0,  // Cast all inputs to at::kFloat before running the op.
  bf16,      // Cast all inputs to at::bfloat16 before running the op.
  int8,
};

enum class LayoutCastPolicy : uint8_t {
  strided = 0,
  mkldnn,
};

bool is_autocast_enabled();
void set_autocast_enabled(bool new_enabled);
at::ScalarType get_autocast_dtype();
void set_autocast_dtype(at::ScalarType dtype);
at::Layout get_autocast_layout();
void set_autocast_layout(at::Layout layout);
int autocast_increment_nesting();
int autocast_decrement_nesting();
void clear_autocast_cache();

Tensor cpu_cached_cast(DtypeCastPolicy cast_policy,
                       LayoutCastPolicy layout_policy, const Tensor& arg);

inline c10::optional<Tensor> cpu_cached_cast(DtypeCastPolicy cast_policy,
                                             LayoutCastPolicy layout_policy,
                                             const c10::optional<Tensor>& arg) {
  if (arg.has_value()) {
    return cpu_cached_cast(cast_policy, layout_policy, *arg);
  } else {
    return c10::nullopt;
  }
}

inline std::vector<Tensor> cpu_cached_cast(DtypeCastPolicy cast_policy,
                                           LayoutCastPolicy layout_policy,
                                           const TensorList& arg) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cpu_cached_cast(cast_policy, layout_policy, t));
  }
  return vec;
}

template <typename T>
inline T cpu_cached_cast(DtypeCastPolicy cast_policy,
                         LayoutCastPolicy layout_policy, T arg) {
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
