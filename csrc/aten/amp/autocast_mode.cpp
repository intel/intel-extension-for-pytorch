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
  lower_precision =
      0, // Cast all inputs to lower_precision before running the op.
         // Currently, lower_precision is bf16 / fp16 for AutocastXPU, and is
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

// CastPolicy::lower_precision General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::lower_precision,
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
  // lower_precision cast policy
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
      lower_precision)
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
      lower_precision)
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
      lower_precision)
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
      lower_precision)
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
      lower_precision)
  KERNEL_XPU(
      ADD_NS(conv_tbc),
      "conv_tbc",
      Tensor(const Tensor&, const Tensor&, const Tensor&, int64_t),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(conv_transpose1d),
      "conv_transpose1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(conv_transpose2d),
      "conv_transpose2d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(conv_transpose3d),
      "conv_transpose3d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision)
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
      lower_precision)
  KERNEL_XPU(
      ADD_NS(prelu),
      "prelu",
      Tensor(const Tensor&, const Tensor&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(addmm),
      "addmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(addmv),
      "addmv",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(addr),
      "addr",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(matmul),
      "matmul",
      Tensor(const Tensor&, const Tensor&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(mm), "mm", Tensor(const Tensor&, const Tensor&), lower_precision)
  KERNEL_XPU(
      ADD_NS(mv), "mv", Tensor(const Tensor&, const Tensor&), lower_precision)
  KERNEL_XPU(
      ADD_NS(linear),
      "linear",
      Tensor(const Tensor&, const Tensor&, const c10::optional<Tensor>&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(addbmm),
      "addbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(baddbmm),
      "baddbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision)
  KERNEL_XPU(
      ADD_NS(bmm), "bmm", Tensor(const Tensor&, const Tensor&), lower_precision)
  KERNEL_XPU(
      ADD_NS(chain_matmul), "chain_matmul", Tensor(TensorList), lower_precision)
  KERNEL_XPU(
      ADD_NS(linalg_multi_dot),
      "linalg_multi_dot",
      Tensor(TensorList),
      lower_precision)
  // The macro doesn't like these (I think it chokes on commas inside <>) so
  // write them manually
  m.impl(
      "_thnn_fused_gru_cell",
      TORCH_FN((&WrapFunction<
                CastPolicy::lower_precision,
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
  m.impl(
      "gru_cell",
      TORCH_FN((&WrapFunction<
                CastPolicy::lower_precision,
                DeviceType::XPU,
                Tensor(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                Tensor(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                &ADD_NS(gru_cell)>::type::call)));
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
      ADD_NS(nll_loss2d),
      "nll_loss2d",
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
  KERNEL_XPU(ADD_NS(acos), "acos", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(asin), "asin", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(cosh), "cosh", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(erfinv), "erfinv", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(exp), "exp", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(expm1), "expm1", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(log), "log", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(log10), "log10", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(log2), "log2", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(log1p), "log1p", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(reciprocal), "reciprocal", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(rsqrt), "rsqrt", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(sinh), "sinh", Tensor(const Tensor&), fp32)
  KERNEL_XPU(ADD_NS(tan), "tan", Tensor(const Tensor&), fp32)
  KERNEL_XPU(
      ADD_NS(pow),
      "pow.Tensor_Scalar",
      Tensor(const Tensor&, const Scalar&),
      fp32)
  KERNEL_XPU(
      ADD_NS(pow),
      "pow.Tensor_Tensor",
      Tensor(const Tensor&, const Tensor&),
      fp32)
  KERNEL_XPU(
      ADD_NS(pow), "pow.Scalar", Tensor(const Scalar&, const Tensor&), fp32)
  KERNEL_XPU(
      ADD_NS(softplus),
      "softplus",
      Tensor(const Tensor&, const Scalar&, const Scalar&),
      fp32)
  KERNEL_XPU(
      ADD_NS(group_norm),
      "group_norm",
      Tensor(
          const Tensor&,
          int64_t,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          double,
          bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(frobenius_norm), "frobenius_norm", Tensor(const Tensor&), fp32)
  KERNEL_XPU(
      ADD_NS(frobenius_norm),
      "frobenius_norm.dim",
      Tensor(const Tensor&, IntArrayRef, bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(nuclear_norm), "nuclear_norm", Tensor(const Tensor&, bool), fp32)
  KERNEL_XPU(
      ADD_NS(nuclear_norm),
      "nuclear_norm.dim",
      Tensor(const Tensor&, IntArrayRef, bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(cosine_similarity),
      "cosine_similarity",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL_XPU(
      ADD_NS(poisson_nll_loss),
      "poisson_nll_loss",
      Tensor(const Tensor&, const Tensor&, bool, bool, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(cosine_embedding_loss),
      "cosine_embedding_loss",
      Tensor(const Tensor&, const Tensor&, const Tensor&, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(hinge_embedding_loss),
      "hinge_embedding_loss",
      Tensor(const Tensor&, const Tensor&, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(kl_div),
      "kl_div",
      Tensor(const Tensor&, const Tensor&, int64_t, bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(l1_loss),
      "l1_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(smooth_l1_loss),
      "smooth_l1_loss",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL_XPU(
      ADD_NS(huber_loss),
      "huber_loss",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL_XPU(
      ADD_NS(mse_loss),
      "mse_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(margin_ranking_loss),
      "margin_ranking_loss",
      Tensor(const Tensor&, const Tensor&, const Tensor&, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(multilabel_margin_loss),
      "multilabel_margin_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(soft_margin_loss),
      "soft_margin_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(triplet_margin_loss),
      "triplet_margin_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          double,
          double,
          double,
          bool,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(multi_margin_loss),
      "multi_margin_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&,
          const c10::optional<Tensor>&,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(dist),
      "dist",
      Tensor(const Tensor&, const Tensor&, const Scalar&),
      fp32)
  KERNEL_XPU(ADD_NS(pdist), "pdist", Tensor(const Tensor&, double), fp32)
  KERNEL_XPU(
      ADD_NS(cdist),
      "cdist",
      Tensor(const Tensor&, const Tensor&, double, c10::optional<int64_t>),
      fp32)
  KERNEL_XPU(
      ADD_NS(renorm),
      "renorm",
      Tensor(const Tensor&, const Scalar&, int64_t, const Scalar&),
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
  KERNEL_XPU(
      ADD_NS(addcdiv),
      "addcdiv",
      Tensor(const Tensor&, const Tensor&, const Tensor&, const Scalar&),
      promote)
  KERNEL_XPU(
      ADD_NS(addcmul),
      "addcmul",
      Tensor(const Tensor&, const Tensor&, const Tensor&, const Scalar&),
      promote)
  KERNEL_XPU(
      ADD_NS(atan2), "atan2", Tensor(const Tensor&, const Tensor&), promote)
  KERNEL_XPU(
      ADD_NS(bilinear),
      "bilinear",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&),
      promote)
  KERNEL_XPU(
      ADD_NS(cross),
      "cross",
      Tensor(const Tensor&, const Tensor&, c10::optional<int64_t>),
      promote)
  KERNEL_XPU(ADD_NS(dot), "dot", Tensor(const Tensor&, const Tensor&), promote)
  KERNEL_XPU(
      ADD_NS(grid_sampler),
      "grid_sampler",
      Tensor(const Tensor&, const Tensor&, int64_t, int64_t, bool),
      promote)
  KERNEL_XPU(
      ADD_NS(index_put),
      "index_put",
      Tensor(
          const Tensor&,
          const torch::List<c10::optional<Tensor>>&,
          const Tensor&,
          bool),
      promote)
  KERNEL_XPU(
      ADD_NS(tensordot),
      "tensordot",
      Tensor(const Tensor&, const Tensor&, IntArrayRef, IntArrayRef),
      promote)
  KERNEL_XPU(
      ADD_NS(scatter_add),
      "scatter_add",
      Tensor(const Tensor&, int64_t, const Tensor&, const Tensor&),
      promote)
}
} // namespace autocast
} // namespace at
