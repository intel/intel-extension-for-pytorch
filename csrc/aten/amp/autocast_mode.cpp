#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

#include <exception>
#include <iostream>

namespace at {
namespace autocast {

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  lower_precision_fp =
      0, // Cast all inputs to lower_precision_fp before running the op.
         // Currently, lower_precision_fp is bf16 for AutocastXPU, and is
         // defined by user(default bf16) for AutocastXPU.
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
};

/********************************************************************************************************
Templates to provide wrapper functions

I'm copying the pattern used in core/boxing/impl/WrapFunctionIntoFunctor.h to
extract args and return type. (see also
https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)

This strategy uses an exterior "WrapFunction" that extracts arguments on behalf
of (in my case several specializations of) an interior "WrapFunction_". Interior
WrapFunction_ specializations are defined for each CastPolicy.
********************************************************************************************************/

// Base template for WrapFunction_, which is specialized to contain a "call"
// method each CastPolicy
template <
    CastPolicy policy,
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::lower_precision_fp,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(
        get_lower_precision_fp_from_device_type(device_type),
        args,
        device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::fp32,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::fp32_set_opt_dtype DeviceType::XPU
template <class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::fp32_set_opt_dtype,
    DeviceType::XPU,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastXPU);
    if (firstarg_is_eligible(args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype,
      // because setting opt dtype explicitly may interfere with internal
      // implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype DeviceType::XPU
template <class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::fp32_append_dtype,
    DeviceType::XPU,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::AutocastXPU);
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::promote,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type = promote_type(
        get_lower_precision_fp_from_device_type(device_type),
        device_type,
        args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating
// core/boxing/impl/WrapFunctionIntoFunctor.h)
template <
    CastPolicy policy,
    DeviceType device_type,
    class Registered, // The signature for which we're registering.  The
                      // dispatcher's calling code invokes our registered
                      // functions with arguments matching Registered, so we
                      // register WrapFunction_::call methods with a matching
                      // signature to properly field those arguments.
                      // guts::function_traits below extracts return_type and
                      // parameter_types from Registered, which WrapFunction_
                      // templates above use to declare their call methods.
    class Redispatch, // The signature for the function we're redispatching to.
                      // In most cases this is the same as Registered, but for
                      // some ops (for example, ops where we append a dtype)
                      // it's useful to redispatch to a function with a
                      // different signature.
    Redispatch* F> // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<
      policy,
      device_type,
      Redispatch,
      F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

#define ADD_NS(RAW_OP) at::RAW_OP

#define KERNEL_XPU(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  m.impl(                                                  \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),        \
      &WrapFunction<                                       \
          CastPolicy::POLICY,                              \
          DeviceType::XPU,                                 \
          SIGNATURE,                                       \
          SIGNATURE,                                       \
          &FUNC>::type::call);

// Less-common but still useful case: redispatching to a function with a new
// signature (e.g. appending a dtype)
#define KERNEL_XPU_DIFFERENT_REDISPATCH_SIGNATURE(  \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  m.impl(                                           \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
      &WrapFunction<                                \
          CastPolicy::POLICY,                       \
          DeviceType::XPU,                          \
          REGISTER_SIGNATURE,                       \
          REDISPATCH_SIGNATURE,                     \
          &REDISPATCH_FUNC>::type::call);

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, AutocastXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) {
  // lower_precision_fp cast policy
  KERNEL_XPU(
      ADD_NS(_convolution),
      "_convolution.deprecated",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t,
          bool,
          bool,
          bool),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(_convolution),
      "_convolution",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t,
          bool,
          bool,
          bool,
          bool),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv1d),
      "conv1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv2d),
      "conv2d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv3d),
      "conv3d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(convolution),
      "convolution",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(addmm),
      "addmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(matmul),
      "matmul",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(mm),
      "mm",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(linear),
      "linear",
      Tensor(const Tensor&, const Tensor&, const c10::optional<Tensor>&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(addbmm),
      "addbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(baddbmm),
      "baddbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(bmm),
      "bmm",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  m.impl(
      "_thnn_fused_gru_cell",
      TORCH_FN((&WrapFunction<
                CastPolicy::lower_precision_fp,
                DeviceType::XPU,
                std::tuple<Tensor, Tensor>(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                std::tuple<Tensor, Tensor>(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                &ADD_NS(_thnn_fused_gru_cell)>::type::call)));

  // fp32
  KERNEL_XPU(
      ADD_NS(binary_cross_entropy),
      "binary_cross_entropy",
      Tensor(
          const Tensor&, const Tensor&, const c10::optional<Tensor>&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(binary_cross_entropy_with_logits),
      "binary_cross_entropy_with_logits",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(cross_entropy_loss),
      "cross_entropy_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t,
          double),
      fp32)
  KERNEL_XPU(
      ADD_NS(log_softmax),
      "log_softmax.int",
      Tensor(const Tensor&, int64_t, c10::optional<ScalarType>),
      fp32)
  KERNEL_XPU(
      ADD_NS(log_softmax),
      "log_softmax.Dimname",
      Tensor(const Tensor&, Dimname, c10::optional<ScalarType>),
      fp32)
  KERNEL_XPU(
      ADD_NS(nll_loss),
      "nll_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(nll_loss_nd),
      "nll_loss_nd",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ifft),
      "fft_ifft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_fft2),
      "fft_fft2",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ifft2),
      "fft_ifft2",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_fftn),
      "fft_fftn",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          c10::optional<at::IntArrayRef>,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ifftn),
      "fft_ifftn",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          c10::optional<at::IntArrayRef>,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_rfft),
      "fft_rfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_irfft),
      "fft_irfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_rfft2),
      "fft_rfft2",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_irfft2),
      "fft_irfft2",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_rfftn),
      "fft_rfftn",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          c10::optional<at::IntArrayRef>,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_irfftn),
      "fft_irfftn",
      Tensor(
          const Tensor&,
          c10::optional<at::IntArrayRef>,
          c10::optional<at::IntArrayRef>,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_hfft),
      "fft_hfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ihfft),
      "fft_ihfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)

  // promote
  KERNEL_XPU(ADD_NS(cat), "cat", Tensor(TensorList, int64_t), promote)
  KERNEL_XPU(ADD_NS(stack), "stack", Tensor(TensorList, int64_t), promote)
}
} // namespace autocast
} // namespace at
