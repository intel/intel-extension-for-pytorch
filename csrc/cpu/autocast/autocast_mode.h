#pragma once

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <Macros.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/library.h>

namespace torch_ipex {
namespace autocast {

using at::IntArrayRef;
using at::Tensor;
using at::TensorList;
using namespace c10;

enum class IPEX_API DtypeCastPolicy : uint8_t {
  user_defined_dtype = 0,
  fp32, // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //   1. we'd like to run in fp32 and
                      //   2. have a c10::optional<ScalarType> arg that controls
                      //   the output type.
                      // fp32_set_opt_dtype wrappers' policy is:  if the output
                      // type is already set, don't touch it, otherwise, set it
                      // to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //   1. we'd like to run in fp32 and
                     //   2. have some overloads that accept an output type and
                     //   other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't
                     // have an output dtype. The wrapper policy is:  append
                     // at::kFloat to the args, and redispatch to the type-aware
                     // overload.
  promote, // Run in the widest dtype among several args.
  fallthrough, // Do not cast inputs.
};

IPEX_API at::ScalarType get_autocast_dtype();

Tensor cpu_cached_cast(at::ScalarType to_type, const Tensor& arg);

inline c10::optional<Tensor> cpu_cached_cast(
    at::ScalarType to_type,
    const c10::optional<Tensor>& arg) {
  if (arg.has_value()) {
    return cpu_cached_cast(to_type, *arg);
  } else {
    return c10::nullopt;
  }
}

inline std::vector<Tensor> cpu_cached_cast(
    at::ScalarType to_type,
    const at::ITensorListRef& arg) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cpu_cached_cast(to_type, t));
  }
  return vec;
}

inline std::vector<Tensor> cpu_cached_cast(
    at::ScalarType to_type,
    const TensorList& arg) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cpu_cached_cast(to_type, t));
  }
  return vec;
}

inline std::vector<Tensor> cpu_cached_cast(
    at::ScalarType to_type,
    const std::vector<at::Tensor>& arg) {
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

// Overload to catch Tensor args.
// If nextArg is floating-point, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
inline at::ScalarType prioritize(
    at::ScalarType current,
    const Tensor& nextArg) {
  TORCH_CHECK(
      current != at::kDouble,
      "promote type is double in at::autocast::prioritize");
  at::ScalarType lower_precision_fp =
      at::autocast::get_lower_precision_fp_from_device_type(
          c10::DeviceType::CPU);
  if (at::autocast::is_autocast_eligible(nextArg, c10::DeviceType::CPU)) {
    auto next = nextArg.scalar_type();
    if (next == at::kDouble) {
      return current; // ignores double tensors
    } else if (current == at::kFloat || next == at::kFloat) {
      return at::kFloat; // prioritizes float over bfloat16
    } else if (current == lower_precision_fp && next == lower_precision_fp) {
      return lower_precision_fp;
    } else {
      AT_ERROR("Unexpected floating ScalarType in at::autocast::prioritize");
      return current;
    }
  } else {
    return current;
  }
}

// Overload to catch TensorList args (for e.g. cat, stack).
// Reuses the overload above to process each Tensor in the list.
inline at::ScalarType prioritize(
    at::ScalarType current,
    const TensorList& list) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor);
  }
  return current;
}

inline at::ScalarType prioritize(
    at::ScalarType current,
    const std::vector<Tensor>& list) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor);
  }
  return current;
}

inline at::ScalarType prioritize(
    at::ScalarType current,
    const at::ITensorListRef& list) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor);
  }
  return current;
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template <typename T>
inline at::ScalarType prioritize(at::ScalarType current, T nextArg) {
  return current;
}

// Overload for the tail case.
inline at::ScalarType promote_type(at::ScalarType current) {
  return current;
}

// Unpack args and determine if incoming bfloat16 tensors need to be promoted to
// float32. Non-Tensor arguments are ignored.
template <typename Arg0, typename... Args>
inline at::ScalarType promote_type(
    at::ScalarType current,
    Arg0 arg0,
    Args... args) {
  auto new_current = prioritize(current, arg0);
  return promote_type(new_current, args...);
}

} // namespace autocast
} // namespace torch_ipex
